#!/usr/bin/env python3
"""
Script to create a dataset for Unsloth SFT training.
This script processes the raw data and creates a HuggingFace dataset in the format expected by Unsloth's SFTTrainer.
"""

import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import datasets
import json
import logging
from unsloth import FastModel

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from util.train_utils import (
    get_optional_arg,
    get_required_arg,
    load_config,
    get_train_val_path,
)
from util.util import set_rango_logger
from util.constants import RANGO_LOGGER
from tactic_gen.tactic_data import (
    LmProcessedDataset,
    example_collator_conf_from_yaml,
    example_collator_from_conf,
    get_tokenizer,
)

_logger = logging.getLogger(RANGO_LOGGER)


def create_unsloth_dataset(
    conf: dict,
    output_path: str = "unsloth_dataset",
    split: str = "train",
    max_examples: int = None
) -> None:
    """
    Create a dataset for Unsloth SFT training.
    
    Args:
        conf: Configuration dictionary
        output_path: Path to save the dataset
        split: Dataset split ('train' or 'val')
        max_examples: Maximum number of examples to process (None for all)
    """
    print(f"\nCreating Unsloth dataset for {split} split...")
    
    # Get configuration parameters
    example_collator_yaml_conf = get_required_arg("example_collator", conf)
    example_collator_conf = example_collator_conf_from_yaml(example_collator_yaml_conf)
    example_collator = example_collator_from_conf(example_collator_conf)
    
    model_name = get_required_arg("model_name", conf)
    tokenizer = get_tokenizer(model_name)
    
    data_path = Path(get_required_arg("data_path", conf))
    hard_seq_len = get_required_arg("hard_seq_len", conf)
    
    # Get train/val paths
    train_path, val_path = get_train_val_path(data_path)
    data_file_path = train_path if split == "train" else val_path
    
    print(f"Max examples before: {max_examples}")
    # Get number of examples
    if max_examples is None:
        max_examples = get_optional_arg("max_steps", conf, 0) * get_optional_arg("per_device_train_batch_size", conf, 0) * get_optional_arg("gradient_accumulation_steps", conf, 0)
    
    if max_examples == 0:
        max_examples = None
    
    print(f"Using data file: {data_file_path}")
    print(f"Example collator: {example_collator}")
    print(f"Max examples: {max_examples}")
    
    # Create the processed dataset
    dataset = LmProcessedDataset(
        data_file_path,
        tokenizer,
        example_collator,
        hard_seq_len,
        max_examples,
        train_type="unsloth-sft"
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Convert to HuggingFace dataset format
    processed_examples = []
    
    print("Processing examples...")
    empty_examples = 0
    short_examples = 0
    long_examples = 0
    
    for i in tqdm(range(len(dataset))):
        example = dataset[i]
        
        # For unsloth-sft, LmProcessedDataset returns raw text directly
        if isinstance(example, str):
            # Validate the example
            if len(example.strip()) == 0:
                print(f"Warning: Empty example at index {i}")
                empty_examples += 1
                continue
            
            # Check for very short examples (potential issues)
            if len(example) < 50:
                print(f"Warning: Very short example at index {i} (length: {len(example)})")
                short_examples += 1
            
            # Check for very long examples (potential memory issues)
            if len(example) > 10000:
                print(f"Warning: Very long example at index {i} (length: {len(example)})")
                long_examples += 1
            
            # Check for problematic patterns
            if "[TACTIC]" not in example:
                print(f"Warning: Missing [TACTIC] marker at index {i}")

            prompt, completion = example.split("\n[TACTIC]\n")
            processed_examples.append({
                "prompt": prompt,
                "completion": completion
            })
        else:
            print(f"Warning: Unexpected example format at index {i}: {type(example)}")
            continue
    
    print(f"\nDataset validation summary:")
    print(f"Total examples processed: {len(processed_examples)}")
    print(f"Empty examples skipped: {empty_examples}")
    print(f"Short examples (<50 chars): {short_examples}")
    print(f"Long examples (>10000 chars): {long_examples}")
    
    if len(processed_examples) == 0:
        raise ValueError("No valid examples found in dataset!")
    
    # Create HuggingFace dataset
    hf_dataset = datasets.Dataset.from_list(processed_examples)
    
    # Save the dataset
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    dataset_path = output_path / f"{split}_dataset"
    hf_dataset.save_to_disk(str(dataset_path))
    
    print(f"Dataset saved to: {dataset_path}")
    print(f"Dataset info: {hf_dataset.info}")
    
    # Print a sample example
    if len(hf_dataset) > 0:
        print("\nSample example:")
        sample = hf_dataset[0]
        print(f"Text length: {len(sample['text'])} characters")
        print(f"First 200 chars: {sample['text'][:200]}...")
        print(f"Last 100 chars: ...{sample['text'][-100:]}")
        
        # Validate tokenization
        try:
            tokens = tokenizer.encode(sample['text'])
            print(f"Token count: {len(tokens)}")
            if len(tokens) > hard_seq_len:
                print(f"Warning: Sample exceeds hard_seq_len ({len(tokens)} > {hard_seq_len})")
        except Exception as e:
            print(f"Warning: Tokenization error: {e}")
    
    return hf_dataset


def validate_unsloth_dataset(dataset_path: str, tokenizer, max_samples: int = 10000) -> None:
    """
    Validate a created Unsloth dataset for potential issues that could cause training problems.
    
    Args:
        dataset_path: Path to the saved dataset
        tokenizer: Tokenizer to use for validation
        max_samples: Maximum number of samples to check
    """
    print(f"\n=== Validating Unsloth Dataset: {dataset_path} ===")
    
    try:
        dataset = datasets.load_from_disk(dataset_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) == 0:
        print("ERROR: Dataset is empty!")
        return
    
    # Check a sample of examples
    num_to_check = min(max_samples, len(dataset))
    empty_examples = 0
    short_examples = 0
    long_examples = 0
    missing_tactic = 0
    tokenization_errors = 0
    very_long_tokens = 0
    
    print(f"Checking {num_to_check} examples...")
    
    for i in range(num_to_check):
        example = dataset[i]
        text = example.get('text', '')
        
        # Check for empty examples
        if not text or len(text.strip()) == 0:
            empty_examples += 1
            print(f"  Empty example at index {i}")
            continue
        
        # Check for very short examples
        if len(text) < 50:
            short_examples += 1
            print(f"  Very short example at index {i} (length: {len(text)})")
        
        # Check for very long examples
        if len(text) > 10000:
            long_examples += 1
            print(f"  Very long example at index {i} (length: {len(text)})")
        
        # Check for missing [TACTIC] marker
        if "[TACTIC]" not in text:
            missing_tactic += 1
            print(f"  Missing [TACTIC] marker at index {i}")
        
        # Check tokenization
        try:
            tokens = tokenizer.encode(text)
            if len(tokens) > 4096:  # Very long token sequences
                very_long_tokens += 1
                print(f"  Very long token sequence at index {i} ({len(tokens)} tokens)")
        except Exception as e:
            tokenization_errors += 1
            print(f"  Tokenization error at index {i}: {e}")
    
    print(f"\n=== Validation Summary ===")
    print(f"Total examples checked: {num_to_check}")
    print(f"Empty examples: {empty_examples}")
    print(f"Short examples (<50 chars): {short_examples}")
    print(f"Long examples (>10000 chars): {long_examples}")
    print(f"Missing [TACTIC] marker: {missing_tactic}")
    print(f"Tokenization errors: {tokenization_errors}")
    print(f"Very long token sequences (>4096): {very_long_tokens}")
    
    # Overall assessment
    total_issues = empty_examples + short_examples + long_examples + missing_tactic + tokenization_errors + very_long_tokens
    if total_issues == 0:
        print("✅ Dataset appears to be valid")
    elif total_issues < num_to_check * 0.1:  # Less than 10% issues
        print("⚠️  Dataset has some issues but may still be usable")
    else:
        print("❌ Dataset has significant issues that may cause training problems")
    
    # Check for potential causes of NaN loss
    if empty_examples > 0:
        print("  - Empty examples can cause division by zero in loss computation")
    if missing_tactic > 0:
        print("  - Missing [TACTIC] markers can cause incorrect loss computation")
    if tokenization_errors > 0:
        print("  - Tokenization errors can cause training failures")
    if very_long_tokens > 0:
        print("  - Very long sequences can cause memory issues and gradient problems")


def print_unsloth_examples(dataset_path: str, num_examples: int = 5) -> None:
    """
    Load a saved Unsloth dataset from disk and print a few examples.
    Args:
        dataset_path: Path to the saved dataset directory (e.g., 'unsloth_dataset/train_dataset')
        num_examples: Number of examples to print
    """
    dataset = datasets.load_from_disk(dataset_path)
    print(f"Loaded dataset from: {dataset_path}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Printing {min(num_examples, len(dataset))} example(s):\n")
    for i in range(len(dataset)):
        example = dataset[i]
        if len(example['text']) == 0:
            print("Empty example found")
    for i in range(min(num_examples, len(dataset))):
        example = dataset[i]
        print(f"Example {i+1}:")
        print(f"Text length: {len(example['text'])} characters")
        #print(f"First 200 chars: {example['text'][:200]}...")
        #print(f"Last 100 chars: ...{example['text'][-100:]}")
        print(example['text'])
        print("-" * 40)


def append_to_unsloth_dataset(
    conf: dict,
    existing_dataset_path: str,
    output_path: str = None,
    split: str = "train",
    max_examples: int = None
) -> None:
    """
    Append more examples to an existing Unsloth dataset.
    
    Args:
        conf: Configuration dictionary
        existing_dataset_path: Path to the existing dataset
        output_path: Path to save the updated dataset (if None, overwrites existing)
        split: Dataset split ('train' or 'val')
        max_examples: Maximum number of new examples to add (None for all available)
    """
    print(f"\nAppending to existing Unsloth dataset: {existing_dataset_path}")
    
    # Load existing dataset
    existing_dataset = datasets.load_from_disk(existing_dataset_path)
    print(f"Existing dataset size: {len(existing_dataset)}")
    
    # Get configuration parameters for new examples
    example_collator_yaml_conf = get_required_arg("example_collator", conf)
    example_collator_conf = example_collator_conf_from_yaml(example_collator_yaml_conf)
    example_collator = example_collator_from_conf(example_collator_conf)
    
    model_name = get_required_arg("model_name", conf)
    tokenizer = get_tokenizer(model_name)
    
    data_path = Path(get_required_arg("data_path", conf))
    hard_seq_len = get_required_arg("hard_seq_len", conf)
    
    # Get train/val paths
    train_path, val_path = get_train_val_path(data_path)
    data_file_path = train_path if split == "train" else val_path
    
    print(f"Using data file: {data_file_path}")
    print(f"Max new examples: {max_examples}")
    
    # Calculate how many instances to skip to avoid duplicates
    # The existing dataset size tells us how many examples were already processed
    skip_instances = len(existing_dataset)
    print(f"Skipping first {skip_instances} instances to avoid duplicates")
    
    # Create new examples dataset with skip_instances
    new_dataset = LmProcessedDataset(
        data_file_path,
        tokenizer,
        example_collator,
        hard_seq_len,
        max_examples,
        train_type="unsloth-sft",
        skip_instances=skip_instances
    )
    
    print(f"New examples to add: {len(new_dataset)}")
    
    # Convert new examples to HuggingFace format
    new_examples = []
    print("Processing new examples...")
    for i in tqdm(range(len(new_dataset))):
        example = new_dataset[i]
        
        if isinstance(example, str):
            new_examples.append({
                "text": example
            })
        else:
            print(f"Warning: Unexpected example format at index {i}: {type(example)}")
            continue
    
    # Create HuggingFace dataset from new examples
    new_hf_dataset = datasets.Dataset.from_list(new_examples)
    
    # Concatenate existing and new datasets
    combined_dataset = datasets.concatenate_datasets([existing_dataset, new_hf_dataset])
    
    print(f"Combined dataset size: {len(combined_dataset)}")
    
    # Save the combined dataset
    if output_path is None:
        output_path = existing_dataset_path
    else:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
    
    combined_dataset.save_to_disk(str(output_path))
    
    print(f"Combined dataset saved to: {output_path}")
    print(f"Dataset info: {combined_dataset.info}")


def main():
    parser = argparse.ArgumentParser(
        description="Create a dataset for Unsloth SFT training"
    )
    parser.add_argument(
        "yaml_config",
        help="YAML config file to use for dataset creation"
    )
    parser.add_argument(
        "--output-path",
        default="unsloth_dataset",
        help="Path to save the dataset (default: unsloth_dataset)"
    )
    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default="train",
        help="Dataset split to create (default: train)"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of examples to process (default: all)"
    )
    parser.add_argument(
        "--append-to",
        help="Path to existing dataset to append to (instead of creating new dataset)"
    )
    parser.add_argument(
        "--output-path-append",
        help="Path to save the updated dataset when appending (default: overwrites existing)"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the created dataset for potential issues"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    set_rango_logger(__file__, logging.DEBUG)
    
    # Load configuration
    conf = load_config(args.yaml_config)
    
    # Get tokenizer for validation
    model_name = get_required_arg("model_name", conf)
    tokenizer = get_tokenizer(model_name)
    
    # Create or append dataset
    if args.append_to:
        append_to_unsloth_dataset(
            conf=conf,
            existing_dataset_path=args.append_to,
            output_path=args.output_path_append,
            split=args.split,
            max_examples=args.max_examples
        )
        if args.validate:
            validate_unsloth_dataset(args.append_to, tokenizer)
    else:
        dataset = create_unsloth_dataset(
            conf=conf,
            output_path=args.output_path,
            split=args.split,
            max_examples=args.max_examples
        )
        
        if args.validate:
            dataset_path = Path(args.output_path) / f"{args.split}_dataset"
            validate_unsloth_dataset(str(dataset_path), tokenizer)


if __name__ == "__main__":
    main() 
    # model_name = "unsloth/codellama-7b-bnb-4bit"
    # tokenizer = get_tokenizer(model_name)
    # _, tokenizer = FastModel.from_pretrained(
    #     model_name = model_name,
    #     max_seq_length = 4096,
    #     load_in_4bit = True,
    #     load_in_8bit = False,
    #     full_finetuning = False,
    # )
    
    # validate_unsloth_dataset("train_dataset", tokenizer)
    #print_unsloth_examples("train_dataset", num_examples=1)