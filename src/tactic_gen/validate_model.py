#!/usr/bin/env python3
"""
Validation script for tactic generation models.

This script loads a trained model from a checkpoint and runs validation
on the validation dataset without performing any training.
"""

import argparse
import sys
import os
import csv
from pathlib import Path
from typing import Optional, Any
import logging

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import transformers
from transformers import AutoTokenizer

# Monkey-patch torch.load to handle PyTorch 2.6+ weights_only issue
original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    # If weights_only is not explicitly set, set it to False for backward compatibility
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)

torch.load = patched_torch_load

from util.train_utils import (
    get_optional_arg,
    get_required_arg,
    get_training_args,
    load_config,
    get_train_val_path,
)
from util.util import set_rango_logger
from util.constants import RANGO_LOGGER
from tactic_gen.tactic_data import (
    LmDataset,
    LmProcessedDataset,
    TacticDataConf,
    example_collator_conf_from_yaml,
    example_collator_from_conf,
    get_tokenizer,
)
from tactic_gen.tactic_data import ReasoningCollator
from tactic_gen.reward_utils import (
    reward_goals,
    get_file_info,
    get_last_point,
    get_proof_goals,
    calculate_reasoning_format_reward,
    calculate_unchanged_reward,
    embedding_model
)
from data_management.splits import Split
from data_management.jsonl_utils import ExampleDB
from coqpyt.coq.base_file import CoqFile
from coqpyt.lsp.structs import (
    VersionedTextDocumentIdentifier,
    TextDocumentContentChangeEvent,
)
from sentence_transformers import util

_logger = logging.getLogger(RANGO_LOGGER)

# {file_path: workspace_path}
valid_files = {}


def init_valid_files(repo_path: Path) -> set[Path]:
    """Initialize valid files for the repository."""
    for repo in Path(os.path.join(repo_path, "repos")).iterdir():
        if repo.is_dir():
            if (repo / "valid_files.csv").exists():
                with open(repo / "valid_files.csv", "r") as f:
                    reader = csv.reader(f)
                    for row in reader:
                        valid_files[os.path.join("repos", repo.name, row[0])] = row[1]
    return valid_files


def calculate_reward(
    completion, 
    prefix, 
    temp_file_name, 
    uri, 
    coq_file, 
    ground_truth_goals,
    previous_goals
):
    """Calculate reward for a completion using Coq verification."""
    try:
        generated_tactic = ReasoningCollator.extract_tactic(completion)
        if "admit" in generated_tactic:
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
            unchanged_reward = calculate_unchanged_reward(previous_goals, goals)
            progress_reward = -1 if unchanged_reward == -1 else reward_goals(ground_truth_goals, goals)
        else:
            progress_reward = -1
        
        final_reward = progress_reward
    except TimeoutError as e:
        print("Timeout error", e, temp_file_name, file=sys.stderr)
        final_reward = -1
    return final_reward


def get_validation_datasets(
    conf: dict[str, Any], tokenizer: Optional[transformers.PreTrainedTokenizer] = None
) -> LmDataset | LmProcessedDataset:
    """Get validation dataset only."""
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
        
        # Use original validation path directly for validation
        val_dataset = LmProcessedDataset(
            orig_val_path,
            tokenizer,
            example_collator,
            hard_seq_len,
            num_eval_examples,
            train_type=conf["train_type"]
        )
        return val_dataset
    else:
        assert "tactic_data" in conf
        dataset_conf = TacticDataConf.from_yaml(conf["tactic_data"])
        val_dataset = LmDataset.from_conf(
            dataset_conf, Split.VAL, conf.get("num_eval_examples", None)
        )
        return val_dataset


def load_model_from_checkpoint(
    model_name: str, 
    checkpoint_path: str, 
    conf: dict[str, Any]
) -> tuple[Any, Any]:
    """Load model from checkpoint based on training type."""
    if conf["train_type"] == "grpo" or conf["train_type"] == "sft":
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel
        
        # Load base model
        if conf["train_type"] == "sft":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_storage=torch.bfloat16,
                llm_int8_enable_fp32_cpu_offload=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
            )
        
        # Load LoRA weights
        model = PeftModel.from_pretrained(model, checkpoint_path)
        tokenizer = get_tokenizer(model_name)
        
    elif conf["train_type"] == "unsloth-sft" or conf["train_type"] == "unsloth-grpo":
        from unsloth import FastModel, FastLanguageModel
        
        # Load base model
        model, tokenizer = FastModel.from_pretrained(
            model_name=model_name,
            max_seq_length=conf["hard_seq_len"],
            load_in_4bit=True,
            load_in_8bit=False,
            full_finetuning=False,
        )
        
        # Load LoRA weights
        try:
            model = FastLanguageModel.get_peft_model(
                model,
                r=conf["peft_lora_r"],
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj",],
                lora_alpha=conf["peft_lora_alpha"],
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=3407,
                max_seq_length=conf["hard_seq_len"],
                use_rslora=False,
                loftq_config=None
            )
            # Load checkpoint weights
            model.load_state_dict(torch.load(f"{checkpoint_path}/adapter_model.bin"))
        except Exception as e:
            print(f"Warning: loading unsloth checkpoint: {e}")
            # Try to load the entire model
            model = FastModel.from_pretrained(checkpoint_path)
    else:
        raise ValueError(f"Invalid train type: {conf['train_type']}")
    
    return model, tokenizer


def create_validation_trainer(
    conf: dict[str, Any], 
    model: Any, 
    tokenizer: Any, 
    val_dataset: Any
) -> Any:
    """Create trainer for validation only."""
    # Create minimal training args for validation
    training_args = get_training_args(conf, None)
    
    # Override some args for validation-only mode
    training_args.do_train = False
    training_args.do_eval = True
    training_args.evaluation_strategy = "no"  # We'll run evaluation manually
    
    if conf["train_type"] == "grpo" or conf["train_type"] == "unsloth-grpo":
        from trl import GRPOTrainer
        
        # Use the same reward functions as in training
        def check_answer(prompts, completions, answer, **kwargs):
            """Calculate cosine similarity between completions and answers."""
            # Clean the completions and answers
            cleaned_completions = [completion.strip(tokenizer.eos_token).strip() for completion in completions]
            cleaned_answers = [a.strip() for a in answer]
            
            # Calculate cosine similarities
            rewards = []
            for completion, ans in zip(cleaned_completions, cleaned_answers):
                try:
                    # Encode both completion and answer
                    embedding1 = embedding_model.encode(completion, convert_to_tensor=True)
                    embedding2 = embedding_model.encode(ans, convert_to_tensor=True)
                    # Calculate cosine similarity
                    similarity = util.cos_sim(embedding1, embedding2).item()
                    rewards.append(similarity)
                except Exception as e:
                    print(f"Error calculating similarity: {e}")
                    rewards.append(0.0)
            
            print("Rewards Answer (cosine similarity)", rewards, len(rewards))
            return rewards

        def check_reward(prompts, completions, answer, **kwargs):
            """Calculate reward using Coq verification."""
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

                    coq_file.version += 1
                    coq_file.coq_lsp_client.didChange(
                        VersionedTextDocumentIdentifier(uri, coq_file.version),
                        [TextDocumentContentChangeEvent(None, None, prefix)],
                    )
                    line, column = get_last_point(prefix)
                    previous_goals = get_proof_goals(coq_file, line, column, uri)

                    reward_cache = {}
                    print("Completions length", len(completions))
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
                            ground_truth_goals,
                            previous_goals
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
        
        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=[check_answer],  # Use the same reward function as training
            args=training_args,
            train_dataset=None,  # No training dataset
            eval_dataset=val_dataset
        )
        
    elif conf["train_type"] == "sft":
        from transformers import Trainer
        
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            data_collator=val_dataset.collator,
            train_dataset=None,  # No training dataset
            eval_dataset=val_dataset,
        )
        
    elif conf["train_type"] == "unsloth-sft":
        from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
        
        # Note: not adding the ] avoids issues with the tokenizer
        response_template = "[TACTIC"
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            data_collator=DataCollatorForCompletionOnlyLM(
                response_template,
                tokenizer=tokenizer,
            ),
            train_dataset=None,  # No training dataset
        )
    else:
        raise ValueError(f"Invalid train type: {conf['train_type']}")
    
    return trainer


def main():
    parser = argparse.ArgumentParser(
        description="Validate a trained model from checkpoint"
    )
    parser.add_argument(
        "--config", 
        required=True, 
        help="Path to the training configuration YAML file"
    )
    parser.add_argument(
        "--checkpoint", 
        required=True, 
        help="Path to the model checkpoint directory"
    )
    parser.add_argument(
        "--output_dir", 
        default="validation_results",
        help="Directory to save validation results"
    )
    parser.add_argument(
        "--num_eval_examples", 
        type=int, 
        default=None,
        help="Number of evaluation examples to use (default: all)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    set_rango_logger(__file__, logging.INFO)
    
    # Load configuration
    conf = load_config(args.config)
    
    # Initialize valid files if repos_path is provided
    if "repos_path" in conf:
        init_valid_files(conf["repos_path"])
    
    # Override num_eval_examples if provided
    if args.num_eval_examples is not None:
        conf["num_eval_examples"] = args.num_eval_examples
    
    print(f"Loading model from checkpoint: {args.checkpoint}")
    print(f"Configuration: {args.config}")
    print(f"Train type: {conf['train_type']}")
    
    # Load model and tokenizer
    model_name = get_required_arg("model_name", conf)
    model, tokenizer = load_model_from_checkpoint(model_name, args.checkpoint, conf)
    
    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.to("cuda")
        print(f"Model moved to GPU: {next(model.parameters()).device}")
    
    # Load validation dataset
    print("Loading validation dataset...")
    val_dataset = get_validation_datasets(conf, tokenizer)
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create trainer
    print("Creating validation trainer...")
    trainer = create_validation_trainer(conf, model, tokenizer, val_dataset)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Run validation
    print("Running validation...")
    results = trainer.evaluate()
    
    # Save results
    results_file = output_dir / "validation_results.json"
    import json
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Validation completed!")
    print(f"Results saved to: {results_file}")
    print("\nValidation Results:")
    for key, value in results.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main() 