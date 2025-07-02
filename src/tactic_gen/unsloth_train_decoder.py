import unsloth
import os
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

from typing import Optional, Any

import csv
import sys
import argparse
from pathlib import Path
import shutil
import json
from tqdm import tqdm
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
    PreTrainedTokenizer,
    BitsAndBytesConfig
)
import torch
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util

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
    NEWLINE_RESPONSE_TEMPLATE,
)
from tactic_gen.tactic_data import ReasoningCollator
import datasets

from torch.utils.data import Subset
import logging

from tactic_gen.create_unsloth_dataset import create_unsloth_dataset
from tactic_gen.debug_callbacks import create_debug_callbacks, SimpleCallback

_logger = logging.getLogger(RANGO_LOGGER)
# {file_path: workspace_path}
valid_files = {}

# Unsloth imports
from unsloth import FastModel
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only, get_chat_template

from transformers import DataCollatorForSeq2Seq

from trl import SFTTrainer, GRPOTrainer



def init_valid_files(repo_path: Path) -> set[Path]:
    """
    Initialize the valid files for the repository.
    """
    for repo in Path(os.path.join(repo_path, "repos")).iterdir():
        if repo.is_dir():
            if (repo / "valid_files.csv").exists():
                with open(repo / "valid_files.csv", "r") as f:
                    reader = csv.reader(f)
                    for row in reader:
                        valid_files[os.path.join("repos", repo.name, row[0])] = row[1]
    return valid_files


# ================================= Model Functions =================================

def get_model(model_name: str, conf: dict[str, Any]) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    model, tokenizer = FastModel.from_pretrained(
        model_name = model_name,
        max_seq_length = conf["hard_seq_len"],
        load_in_4bit = True,
        load_in_8bit = False,
        full_finetuning = False,
    )
    return model, tokenizer

def process_model(model_name: str, conf: dict[str, Any]) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    if conf["train_type"] == "unsloth-sft" or conf["train_type"] == "unsloth-grpo":    
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
                loftq_config = None,
                attention_implementation = "flash_attention_2"
            )
        except RuntimeError as e:
            print(f"Warning: getting unsloth model: {e}")
            model = raw_model
    else:
        raise ValueError(f"Invalid train type: {conf['train_type']}")

    return model, tokenizer

# ================================= Data Functions =================================

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
        tokenizer = get_tokenizer(get_required_arg("model_name", conf))

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
            train_type=conf["train_type"],
            skip_instances=conf["skip_instances"]
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


def get_sft_trainer(
        conf: dict[str, Any], 
        local_rank: Optional[int], 
        checkpoint_name: Optional[str]
    ) -> "SFTTrainer":
    
    print("\n\nBuilding Training Config...")
    training_args = get_training_args(conf, local_rank)
    print("\n\nRetrieving Model...")
    model_name = get_required_arg("model_name", conf)
    model, tokenizer = process_model(model_name, conf)
    train_dataset, val_dataset = get_datasets(conf, tokenizer)
    print("\n\nConstructing Dataset...")
    if "dataset_path" not in conf:
        train_dataset_path = Path("unsloth_dataset/train_dataset")
        create_unsloth_dataset(
                conf=conf,
                output_path="unsloth_dataset",
                split="train",
                max_examples=conf.get("max_steps", None) * conf.get("per_device_train_batch_size", 1)
            )
        processed_train_dataset = datasets.load_from_disk(str(train_dataset_path))
    else:
        processed_train_dataset = datasets.load_from_disk(conf["dataset_path"])

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="qwen-2.5",
    )

    EOS_TOKEN = tokenizer.eos_token
    def formatting_prompts_func(examples):
        texts = examples["text"]
        new_texts = []
        for text in texts:
            if "\n[TACTIC]\n" in text:
                user_part, assistant_part = text.split("\n[TACTIC]\n", 1)
            else:
                user_part = text
                assistant_part = ""
            messages = [
                {"role": "system", 
                 "content": "You are a Coq tactic predictor. Given a set of relevant premises, \
                and proofs, the current state of the proof and the current written proof script, \
                generate only the next tactic."},
                {"role": "user", "content": user_part.strip()},
                {"role": "assistant", "content": assistant_part.strip()},
            ]
            formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)
            formatted_text = formatted_text.rsplit("<|im_end|>", 1)[0] + EOS_TOKEN
            new_texts.append(formatted_text)

        return {"text" : new_texts, }
    
    
    processed_train_dataset = processed_train_dataset.map(
        formatting_prompts_func, batched = True,
    )
    print("processed_train_dataset: ", processed_train_dataset["text"][0])
    print("EOS_TOKEN: ", EOS_TOKEN)

    
    print("\n\nBuilding Trainer...")
    if conf["train_type"] == "unsloth-sft":   
        trainer = SFTTrainer(
            model=model,
            tokenizer=train_dataset.tokenizer,
            train_dataset=processed_train_dataset,
            dataset_text_field="text",
            max_seq_length=conf["hard_seq_len"],
            data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
            dataset_num_proc = 16,   
            args=training_args,
            #callbacks=[SimpleCallback("train", tokenizer)],  # Add debug callbacks
        )
        trainer.args.warmup_ratio = 0

        trainer = train_on_responses_only(
            trainer,
            instruction_part = "<|im_start|>user\n",
            response_part = "<|im_start|>assistant\n",
        )

    else:
        raise ValueError(f"Invalid train type: {conf['train_type']}")
    
    return trainer


def get_grpo_trainer(
        conf: dict[str, Any], 
        local_rank: Optional[int], 
        checkpoint_name: Optional[str]
    ) -> "GRPOTrainer":
    print("\n\nBuilding Training Config...")
    training_args = get_training_args(conf, local_rank)
    print("\n\nRetrieving Model...")
    model_name = get_required_arg("model_name", conf)
    model, tokenizer = process_model(model_name, conf)
    train_dataset, val_dataset = get_datasets(conf, tokenizer)
    embedding_model = SentenceTransformer('nomic-ai/CodeRankEmbed', trust_remote_code=True).to('cpu')
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="qwen-2.5",
    )

    def formatting_prompts_func_grpo(examples):
        raw_prompts = examples["prompt"]
        formatted_prompts = []
        for raw_prompt in raw_prompts:
            if "\n[TACTIC]\n" in raw_prompt:
                user_part, assistant_part = raw_prompt.split("\n[TACTIC]\n", 1)
            else:
                user_part = raw_prompt
                assistant_part = ""
            messages = [
                {"role": "system",
                "content": "You are a Coq tactic predictor. Given a set of relevant premises, \
                and proofs, the current state of the proof and the current written proof script, \
                generate only the next tactic."},
                {"role": "user", "content": user_part.strip()},
                #{"role": "assistant", "content": assistant_part.strip()},
            ]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False)
            formatted_prompts.append(formatted)

        # Return *all* original fields, with the modified 'prompt'
        new_examples = {key: examples[key] for key in examples}
        new_examples["prompt"] = formatted_prompts
        return new_examples
    
    processed_train_dataset = train_dataset.map(
        formatting_prompts_func_grpo, batched = True,
    )
    print("processed_train_dataset: ", processed_train_dataset["prompt"][0])


    def check_answer(prompts, completions, answer, **kwargs):
        print("Completions: ", completions)
        print("Answer: ", answer)
        cleaned_completions = [completion.strip(tokenizer.eos_token).strip() for completion in completions]
        cleaned_answers = [a.strip() for a in answer]
        
        rewards = []
        for completion, ans in zip(cleaned_completions, cleaned_answers):
            try:
                embedding1 = embedding_model.encode(completion, convert_to_tensor=True)
                embedding2 = embedding_model.encode(ans, convert_to_tensor=True)
                similarity = util.cos_sim(embedding1, embedding2).item()
                rewards.append(similarity)
            except Exception as e:
                print(f"Error calculating similarity for {completion} and {ans}: {e}")
                rewards.append(0)
        return rewards

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[check_answer],
        args=training_args,
        train_dataset=processed_train_dataset,
        eval_dataset=val_dataset,
        callbacks=[SimpleCallback("train", tokenizer)],
    )
    trainer.args.warmup_ratio = 0

    return trainer
    
def arg_parser():
    parser = argparse.ArgumentParser(
        description="Train code llama by providing a .yaml config file. As an example, see src/tactic_gen/confs/basic_train.yaml"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local rank passed from distributed launcher",
    )
    parser.add_argument("yaml_config", help="yaml config file to use for training.")
    return parser

if __name__ == "__main__":
    print("Parsing args...")
    parser = arg_parser()
    args = parser.parse_args(sys.argv[1:])
    set_rango_logger(__file__, logging.DEBUG)
    conf = load_config(args.yaml_config)


    if "repos_path" in conf:
        init_valid_files(conf["repos_path"])
    train_from_checkpoint = (
        conf["checkpoint_name"] if "checkpoint_name" in conf else None
    )
    trainer = get_sft_trainer(conf, args.local_rank, train_from_checkpoint)  # Fixed function name

    if train_from_checkpoint:
        print("Starting training from checkpoint...")
        checkpoint_name = conf["checkpoint_name"]
        print(f"Training from checkpoint {checkpoint_name}")
        transformers.logging.set_verbosity_info()
        trainer.train(checkpoint_name)
    else:
        print("Starting training from scratch...")
        make_output_dir(conf)
        copy_configs(args.yaml_config, conf, TrainType.TACTIC)
        print("Training from scratch")
        trainer.train()
    trainer.save_model()
    trainer.save_state()
