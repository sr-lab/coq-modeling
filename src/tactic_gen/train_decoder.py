from typing import Optional, Any

import unsloth
import os
import csv
import sys
import argparse
from pathlib import Path
import shutil
import json
import numpy.core.multiarray
from coqpyt.coq.base_file import CoqFile
from coqpyt.lsp.structs import (
    VersionedTextDocumentIdentifier,
    TextDocumentContentChangeEvent,
)
from peft import LoraConfig, get_peft_model
import transformers
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    BitsAndBytesConfig
)
import torch

from tactic_gen.reward_utils import (
    reward_goals,
    get_file_info,
    get_last_point,
    get_proof_goals,
    calculate_reasoning_format_reward,
    calculate_reasoning_length_reward,
    calculate_tactic_format_reward,
    calculate_admit_reward
)

from util.train_utils import (
    get_optional_arg,
    get_required_arg,
    get_training_args,
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
import datasets

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


# This doc details how to finetune codellama:
# https://github.com/huggingface/trl/blob/main/examples/scripts/sft_trainer.py

# More ideas for arguments here:
# https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments


def get_lora_conf(conf: dict[str, Any]) -> LoraConfig:
    peft_config = LoraConfig(
        lora_alpha=conf["peft_lora_alpha"],
        lora_dropout=0.1,
        r=conf["peft_lora_r"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    return peft_config


def get_model(model_name: str, conf: dict[str, Any]) -> PreTrainedModel:
    tokenizer = None
    if conf["train_type"] == "grpo":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        )
    elif conf["train_type"] == "sft":
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
            device_map="auto"
        )
    elif conf["train_type"] == "unsloth-sft" or conf["train_type"] == "unsloth-grpo":
        from unsloth import FastModel
        model, tokenizer = FastModel.from_pretrained(
            model_name = model_name,
            max_seq_length = conf["hard_seq_len"],
            load_in_4bit = True,
            load_in_8bit = False,
            full_finetuning = False,
        )

    # https://huggingface.co/docs/bitsandbytes/main/en/fsdp_qlora
    # model = prepare_model_for_kbit_training(model)
    # https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/inference/quantization/quantization.py
    # https://github.com/microsoft/DeepSpeedExamples/tree/master/inference/huggingface/zero_inference
    return model, tokenizer


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
    conf: dict[str, Any], tokenizer: Optional[transformers.PreTrainedTokenizer] = None
) -> tuple[LmDataset | LmProcessedDataset, LmDataset | LmProcessedDataset]:
    if "data_path" in conf:
        example_collator_yaml_conf = get_required_arg("example_collator", conf)
        example_collator_conf = example_collator_conf_from_yaml(
            example_collator_yaml_conf
        )
        example_collator = example_collator_from_conf(example_collator_conf)
        _logger.info("EXAMPLE COLLATOR: %s", example_collator)
        if tokenizer is None:
            tokenizer = get_tokenizer(get_required_arg("model_name", conf))
        elif "tokenizer" in conf:
            tokenizer = get_tokenizer(conf["tokenizer"])

        data_path = Path(get_required_arg("data_path", conf))
        num_eval_examples = get_optional_arg("num_eval_examples", conf, None)
        hard_seq_len = get_required_arg("hard_seq_len", conf)

        if conf["example_collator"]["alias"] == "reasoning":
            cot_train_path = data_path / "cot.db"
            train_dataset = LmProcessedDataset(
                cot_train_path, 
                tokenizer, 
                example_collator, 
                hard_seq_len, 
                train_type=conf["train_type"]
            )
            return train_dataset, None

        orig_train_path, orig_val_path = get_train_val_path(data_path)
        
        filtered_train_path = orig_train_path.parent / orig_train_path.name.replace(".db", "_tmp.db")
        filtered_val_path = orig_val_path.parent / orig_val_path.name.replace(".db", "_tmp.db")
        
        if len(valid_files) > 0:
            if filtered_train_path.exists():
                print(f"Reusing existing filtered training database at {filtered_train_path}")
            else:
                _logger.info(f"Creating filtered training database at {filtered_train_path}")
                create_filtered_db(orig_train_path, filtered_train_path)
            
            if filtered_val_path.exists():
                print(f"Reusing existing filtered validation database at {filtered_val_path}")
            else:
                _logger.info(f"Creating filtered validation database at {filtered_val_path}")
                create_filtered_db(orig_val_path, filtered_val_path)
        else:
            _logger.info("No valid files specified, using original databases")
            shutil.copy(orig_train_path, filtered_train_path)
            shutil.copy(orig_val_path, filtered_val_path)
        
        train_dataset = LmProcessedDataset(
            filtered_train_path, 
            tokenizer, 
            example_collator, 
            hard_seq_len, 
            train_type=conf["train_type"]
        )
        val_dataset = LmProcessedDataset(
            filtered_val_path,
            tokenizer,
            example_collator,
            hard_seq_len,
            num_eval_examples,
            train_type=conf["train_type"]
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


def process_model(model_name: str, conf: dict[str, Any]) -> PreTrainedModel:
    if conf["train_type"] == "grpo" or conf["train_type"] == "sft":
        raw_model, tokenizer = get_model(model_name, conf)
        lora_config = get_lora_conf(conf)
        model = get_peft_model(raw_model, lora_config)
    elif conf["train_type"] == "unsloth-sft" or conf["train_type"] == "unsloth-grpo":        
        from unsloth import FastLanguageModel
        raw_model, tokenizer = get_model(model_name, conf)
        try:
            model = FastLanguageModel.get_peft_model(
                raw_model,
                r = conf["peft_lora_r"],
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj",],
                lora_alpha = conf["peft_lora_alpha"],
                lora_dropout = 0,
                bias = "none",
                use_gradient_checkpointing = "unsloth",
                random_state = 3407,
                max_seq_length = conf["hard_seq_len"],
                use_rslora = False,
                loftq_config = None
            )
        except RuntimeError as e:
            print(f"Warning: getting unsloth model: {e}")
            model = raw_model
    else:
        raise ValueError(f"Invalid train type: {conf['train_type']}")
    
    return model, tokenizer


def get_trainer(
    conf: dict[str, Any], local_rank: Optional[int], checkpoint_name: Optional[str]
) -> "Trainer | GRPOTrainer":
    print("\n\nBuilding Training Config...")
    training_args = get_training_args(conf, local_rank)
    print("\n\nRetrieving Model...")
    model_name = get_required_arg("model_name", conf)
    model, tokenizer = process_model(model_name, conf)
    print("Model", model)
    print("Tokenizer", tokenizer)
    print("\n\nConstructing Dataset...")
    train_dataset, val_dataset = get_datasets(conf, tokenizer)

    print("\n\nBuilding Trainer...")

    def check_reward(prompts, completions, answer, **kwargs):
        file_name = kwargs["file_name"][0]
        proof_script = kwargs["proof_script"][0]
        next_steps = kwargs["next_steps"][0]
        temp_file_name = None
        try:
            temp_file_name, prefix = get_file_info(conf, file_name, proof_script)
            line, column = get_last_point(prefix + "\n" + proof_script)
            with open(temp_file_name, "w", encoding="utf-8") as temp_file:
                temp_file.write(prefix + "\n" + next_steps[0])
            rewards = []
            with CoqFile(
                temp_file_name, 
                workspace=valid_files[file_name],
                timeout=120
            ) as coq_file:
                uri = f"file://{coq_file.path}"
                ground_truth_goals = get_proof_goals(coq_file, line, column, uri)
                # Rewrite the file with only the prefix
                with open(temp_file_name, "w", encoding="utf-8") as temp_file:
                    temp_file.write(prefix)

                reward_cache = {}
                for completion in completions:
                    if completion.strip() in reward_cache:
                        rewards.append(reward_cache[completion.strip()])
                        continue
                    final_reward = calculate_reward(
                        completion, 
                        prefix, 
                        temp_file_name, 
                        uri, 
                        coq_file, 
                        ground_truth_goals
                    )
                    rewards.append(final_reward)
            print("Rewards", rewards, len(rewards))
            return rewards
        except Exception as e:
            print("Exception error", e, temp_file_name, file=sys.stderr)
            return [0] * len(completions)
        finally:
            if temp_file_name:
                os.remove(temp_file_name)
    
    if conf["train_type"] == "grpo" or conf["train_type"] == "unsloth-grpo":
        from trl import GRPOTrainer
        trainer = GRPOTrainer(
            model=model,
            processing_class=train_dataset.tokenizer,
            reward_funcs=[calculate_reasoning_format_reward, calculate_reasoning_length_reward, check_reward],
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )
    elif conf["train_type"] == "sft":
        from transformers import Trainer
        trainer = Trainer(
            model=model,
            tokenizer=train_dataset.tokenizer,
            args=training_args,
            data_collator=train_dataset.collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
    elif conf["train_type"] == "unsloth-sft":
        from trl import SFTTrainer
        from transformers import DataCollatorForSeq2Seq
        from unsloth.chat_templates import get_chat_template

        train_dataset.tokenizer = get_chat_template(
            train_dataset.tokenizer,
            chat_template = "unsloth",
        )

        processed_train_dataset = []
        for i in range(len(train_dataset)):
            processed_train_dataset.append({
                "conversations": [
                    {
                        "role": "user",
                        "content": train_dataset[i]["input"],
                    },
                    {
                        "role": "assistant",
                        "content": train_dataset[i]["output"],
                    }
                ]
            })
        processed_train_dataset = datasets.Dataset.from_list(
            processed_train_dataset
        )
        
        def formatting_prompts_func(examples):
            convos = examples["conversations"]
            texts = [
                train_dataset.tokenizer.apply_chat_template(
                    convo, tokenize = False, add_generation_prompt = False
                ) for convo in convos
            ]
            return { "text" : texts, }
        
        processed_train_dataset = processed_train_dataset.map(
            formatting_prompts_func, batched = True
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=train_dataset.tokenizer,
            args=training_args,
            data_collator=DataCollatorForSeq2Seq(
                tokenizer=train_dataset.tokenizer,
            ),
            train_dataset=processed_train_dataset,
        )
    else:
        raise ValueError(f"Invalid train type: {conf['train_type']}")
    
    return trainer


def calculate_reward(
    completion, 
    prefix, 
    temp_file_name, 
    uri, 
    coq_file, 
    ground_truth_goals
):
    try:
        generated_tactic = completion.split("</think>")[-1].strip()
        tactic_format_reward = calculate_tactic_format_reward(generated_tactic)
        if tactic_format_reward == -1:
            return -1
        admit_reward = calculate_admit_reward(generated_tactic)
        if admit_reward == -1:
            return -1
        
        with open(temp_file_name, "w") as temp_file:
            temp_file.write(prefix + "\n" + generated_tactic)
        coq_file.version += 1

        coq_file.coq_lsp_client.didChange(
            VersionedTextDocumentIdentifier(uri, coq_file.version),
            [TextDocumentContentChangeEvent(None, None, prefix + "\n" + generated_tactic)], 
        )
        has_errors = (len(list(filter(lambda x: x.severity == 1, coq_file.diagnostics))) > 0)
        if not has_errors:
            line, column = get_last_point(prefix + "\n" + generated_tactic)
            goals = get_proof_goals(coq_file, line, column, uri)
            unchanged_reward = reward_goals(ground_truth_goals, goals)
        else:
            unchanged_reward = -1
        
        final_reward = unchanged_reward
    except TimeoutError as e:
        print("Timeout error", e, temp_file_name, file=sys.stderr)
        final_reward = -1
    return final_reward

if __name__ == "__main__":
    #accelerator = Accelerator()
    parser = argparse.ArgumentParser(
        description="Train code llama by providing a .yaml config file. As an example, see src/tactic_gen/confs/basic_train.yaml"
    )
    #print(f"<ARGV>{sys.argv}</ARGV")
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
    trainer.save_model()
    trainer.save_state()
