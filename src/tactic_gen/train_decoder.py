from typing import Optional, Any

import os
import time
import csv
import sys
import argparse
from pathlib import Path
import shutil
import json
import uuid
from coqpyt.coq.base_file import CoqFile
from coqpyt.coq.exceptions import InvalidAddException
from coqpyt.lsp.structs import (
    VersionedTextDocumentIdentifier,
    TextDocumentContentChangeEvent,
    Position
)

from peft import LoraConfig, get_peft_model
import transformers
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    BitsAndBytesConfig,
    Trainer,
)
import torch
from typing import Optional, Any

import os
import time
import csv
import sys
import argparse
from pathlib import Path
import shutil
import json
import uuid
from coqpyt.coq.base_file import CoqFile
from coqpyt.coq.exceptions import InvalidAddException
from coqpyt.lsp.structs import (
    VersionedTextDocumentIdentifier,
    TextDocumentContentChangeEvent,
)

from peft import LoraConfig, get_peft_model
import transformers
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    BitsAndBytesConfig,
    Trainer,
)
import torch

from trl import GRPOTrainer

from util.train_utils import (
    get_optional_arg,
    get_required_arg,
    get_grpo_training_args,
    load_config,
    make_output_dir,
    copy_configs,
    get_train_val_path,
    TrainType
)
from util.util import set_rango_logger
from util.constants import RANGO_LOGGER
from data_management.splits import Split
from data_management.jsonl_utils import ExampleDB
from tactic_gen.tactic_data import (
    LmDataset,
    LmProcessedDataset,
    TacticDataConf,
    example_collator_conf_from_yaml,
    example_collator_from_conf,
    get_tokenizer,
)

from torch.utils.data import Subset
import logging

_logger = logging.getLogger(RANGO_LOGGER)
# {file_path: workspace_path}
valid_files = {}

def init_valid_files(repo_path: Path) -> set[Path]:
    for repo in Path(os.path.join(repo_path, "repos")).iterdir():
        if repo.is_dir():
            if (repo / "valid_files.csv").exists():
                with open(repo / "valid_files.csv", "r") as f:
                    reader = csv.reader(f)
                    for row in reader:
                        valid_files[os.path.join("repos", repo.name, row[0])] = row[1]

    return valid_files
# {file_path: workspace_path}
valid_files = {}

def init_valid_files(repo_path: Path) -> set[Path]:
    for repo in Path(os.path.join(repo_path, "repos")).iterdir():
        if repo.is_dir():
            if (repo / "valid_files.csv").exists():
                with open(repo / "valid_files.csv", "r") as f:
                    reader = csv.reader(f)
                    for row in reader:
                        valid_files[os.path.join("repos", repo.name, row[0])] = row[1]

    return valid_files


# This doc details how to finetune codellama:
# https://github.com/huggingface/trl/blob/main/examples/scripts/sft_trainer.py

# More ideas for arguments here:
# https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments


def get_lora_conf(conf: dict[str, Any]) -> LoraConfig:
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    return peft_config


def get_model(model_name: str) -> PreTrainedModel:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
    )

    # https://huggingface.co/docs/bitsandbytes/main/en/fsdp_qlora
    # model = prepare_model_for_kbit_training(model)
    # https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/inference/quantization/quantization.py
    # https://github.com/microsoft/DeepSpeedExamples/tree/master/inference/huggingface/zero_inference
    return model


def create_filtered_db(source_db_path: Path, target_db_path: Path) -> None:
    """
    Create a new database filtered by valid files using ExampleDB methods.
    
    Args:
        source_db_path: Path to the source database
        target_db_path: Path to save the filtered database
        valid_files: Set of valid file paths to filter by
    """
    if len(valid_files) == 0:
        _logger.warning("No valid files provided for filtering, copying entire database")
        shutil.copy(source_db_path, target_db_path)
        return
    
    if target_db_path.exists():
        raise ValueError(f"Target DB {target_db_path} already exists")
    
    source_db = ExampleDB.load(source_db_path)
    target_db = ExampleDB.create(target_db_path)
    
    total_count = source_db.size()
    total_copied = 0
    batch_size = 10000
    
    for i in range(1, total_count + 1, batch_size):
        batch = []
        for j in range(i, min(i + batch_size, total_count + 1)):
            example_text = source_db.retrieve(j)
            example_data = json.loads(example_text)
            if example_data.get('file_name') in valid_files:
                batch.append((example_text,))
        
        if batch:
            target_db.insert_examples(batch)
            total_copied += len(batch)
    
    source_db.close()
    target_db.close()
    
    print(
        f"Created filtered database with {total_copied} examples out of {total_count} original examples ({(total_copied/total_count)*100:.2f}%)"
    )


def get_datasets(
    conf: dict[str, Any],
) -> tuple[LmDataset | LmProcessedDataset, LmDataset | LmProcessedDataset]:
    if "data_path" in conf:
        example_collator_yaml_conf = get_required_arg("example_collator", conf)
        data_path = Path(get_required_arg("data_path", conf))
        num_eval_examples = get_optional_arg("num_eval_examples", conf, None)
        hard_seq_len = get_required_arg("hard_seq_len", conf)
        orig_train_path, orig_val_path = get_train_val_path(data_path)
        
        filtered_train_path = orig_train_path.parent / orig_train_path.name.replace(".db", "_tmp.db")
        filtered_val_path = orig_val_path.parent / orig_val_path.name.replace(".db", "_tmp.db")
        
        if len(valid_files) > 0:
            if filtered_train_path.exists():
                print(f"Reusing existing filtered training database at {filtered_train_path}")
            else:
                _logger.info(f"Creating filtered training database at {filtered_train_path}")
                create_filtered_db(orig_train_path, filtered_train_path)
                create_filtered_db(orig_train_path, filtered_train_path)
            
            if filtered_val_path.exists():
                print(f"Reusing existing filtered validation database at {filtered_val_path}")
            else:
                _logger.info(f"Creating filtered validation database at {filtered_val_path}")
                create_filtered_db(orig_val_path, filtered_val_path)
                create_filtered_db(orig_val_path, filtered_val_path)
        else:
            _logger.info("No valid files specified, using original databases")
            shutil.copy(orig_train_path, filtered_train_path)
            shutil.copy(orig_val_path, filtered_val_path)
        
        example_collator_conf = example_collator_conf_from_yaml(
            example_collator_yaml_conf
        )
        example_collator = example_collator_from_conf(example_collator_conf)
        _logger.info("EXAMPLE COLLATOR: %s", example_collator)
        tokenizer = get_tokenizer(get_required_arg("model_name", conf))
        train_dataset = LmProcessedDataset(
            filtered_train_path, tokenizer, example_collator, hard_seq_len
        )
        val_dataset = LmProcessedDataset(
            filtered_val_path,
            tokenizer,
            example_collator,
            hard_seq_len,
            num_eval_examples,
        )
        return train_dataset, val_dataset
    else:
        assert "tactic_data" in conf
        dataset_conf = TacticDataConf.from_yaml(conf["tactic_data"])
        train_dataset = LmDataset.from_conf(dataset_conf, Split.TRAIN)
        val_dataset = LmDataset.from_conf(
            dataset_conf, Split.VAL, conf.get("num_eval_examples", None)
        )
        return train_dataset, val_dataset


def get_trainer(
    conf: dict[str, Any], local_rank: Optional[int], checkpoint_name: Optional[str]
) -> Trainer:
    print("\n\nBuilding Training Config...")
    training_args = get_grpo_training_args(conf, local_rank)
    print("\n\nRetrieving Model...")
    model_name = get_required_arg("model_name", conf)
    raw_model = get_model(model_name)
    lora_config = get_lora_conf(conf)
    model = get_peft_model(raw_model, lora_config)

    print("\n\nConstructing Dataset...")
    train_dataset, val_dataset = get_datasets(conf)

    print("\n\nBuilding Trainer...")

    def check_reward(prompts, completions, answer, **kwargs):
        file_name = kwargs["file_name"][0]
        proof_script = kwargs["proof_script"][0]
        temp_file_name = None
        
        try:
            with open(os.path.join(conf["repos_path"], file_name), "r") as f:
                file_contents = f.read()
                theorem_split = file_contents.split(proof_script.split(":")[0])
                if len(theorem_split) == 0:
                    logging.error("Proof script not found in file %s", file_name)
                    exit(-1)
                prefix = theorem_split[0] + "\n" + proof_script
            
            original_file_path = os.path.join(conf["repos_path"], file_name)
            dir_path = os.path.dirname(original_file_path)
            file_basename = os.path.basename(file_name)
            random_id = str(uuid.uuid4())[:8]
            temp_file_name = os.path.join(dir_path, f"temp_{random_id}_{file_basename}")
            uri = f"file://{temp_file_name}"

            line, column = get_last_point(prefix)
            with open(temp_file_name, "w", encoding="utf-8") as temp_file:
                temp_file.write(prefix)
            
            initial_goal = coq_file.coq_lsp_client.proof_goals(
                VersionedTextDocumentIdentifier(uri, coq_file.version),
                Position(line, column)
            )
            print(initial_goal)

            rewards = []
            with CoqFile(
                temp_file_name, 
                workspace=valid_files[file_name]
            ) as coq_file:
                reward_cache = {}

                for completion in completions:
                    if completion.strip() in reward_cache:
                        rewards.append(reward_cache[completion.strip()])
                        continue
                    
                    with open(temp_file_name, "w") as temp_file:
                        temp_file.write(prefix + "\n" + completion)
                    coq_file.version += 1
                    coq_file.coq_lsp_client.didChange(
                        VersionedTextDocumentIdentifier(uri, coq_file.version),
                        [TextDocumentContentChangeEvent(None, None, prefix + "\n" + completion)],
                    )

                    line, column = get_last_point(prefix + "\n" + completion)
                    goal = coq_file.coq_lsp_client.proof_goals(
                        VersionedTextDocumentIdentifier(uri, coq_file.version),
                        Position(line, column)
                    )
                    print(goal)

                    reward = int(len(list(filter(lambda x: x.severity == 1, coq_file.diagnostics))) > 0)
                    reward_cache[completion.strip()] = reward
                    rewards.append(reward)
            return rewards
        finally:
            if temp_file_name:
                os.remove(temp_file_name)
    

    trainer = GRPOTrainer(
        model=model,
        processing_class=train_dataset.tokenizer,
        reward_funcs=[check_reward],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    return trainer


def get_last_point(s: str) -> tuple[int, int]:
    """
    Returns the (line, column) of the last character in the string.
    Lines and columns are 0-based.
    If the string is empty, returns (0, 0).
    """
    if not s:
        return (0, 0)
    lines = s.splitlines(keepends=True)
    if not lines:
        return (0, 0)
    last_line_idx = len(lines) - 1
    last_line = lines[-1]
    # If the string ends with a newline, the last character is at column 0 of the next line
    if last_line.endswith('\n') or last_line.endswith('\r'):
        return (last_line_idx + 1, 0)
    else:
        return (last_line_idx, len(last_line) - 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train code llama by providing a .yaml config file. As an example, see src/tactic_gen/confs/basic_train.yaml"
    )
    print(f"<ARGV>{sys.argv}</ARGV")
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local rank passed from distributed launcher",
    )
    parser.add_argument("yaml_config", help="yaml config file to use for training.")
    args = parser.parse_args(sys.argv[1:])
    set_rango_logger(__file__, logging.DEBUG)
    conf = load_config(args.yaml_config)
    if "repos_path" in conf:
        init_valid_files(conf["repos_path"])
    if "repos_path" in conf:
        init_valid_files(conf["repos_path"])
    train_from_checkpoint = (
        conf["checkpoint_name"] if "checkpoint_name" in conf else None
    )
    trainer = get_trainer(conf, args.local_rank, train_from_checkpoint)
    if train_from_checkpoint:
        checkpoint_name = conf["checkpoint_name"]
        print(f"Training from checkpoint {checkpoint_name}")
        transformers.logging.set_verbosity_info()
        trainer.train(checkpoint_name)
    else:
        make_output_dir(conf)
        copy_configs(args.yaml_config, conf, TrainType.TACTIC)
        print("Training from scratch")
        trainer.train()
