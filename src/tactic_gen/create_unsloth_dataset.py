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
    for i in tqdm(range(len(dataset))):
        example = dataset[i]
        
        # For unsloth-sft, LmProcessedDataset returns raw text directly
        if isinstance(example, str):
            processed_examples.append({
                "text": example
            })
        else:
            print(f"Warning: Unexpected example format at index {i}: {type(example)}")
            continue
    
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
    return hf_dataset


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
    
    args = parser.parse_args()
    
    # Setup logging
    set_rango_logger(__file__, logging.DEBUG)
    
    # Load configuration
    conf = load_config(args.yaml_config)
    
    # Create or append dataset
    if args.append_to:
        append_to_unsloth_dataset(
            conf=conf,
            existing_dataset_path=args.append_to,
            output_path=args.output_path_append,
            split=args.split,
            max_examples=args.max_examples
        )
    else:
        create_unsloth_dataset(
            conf=conf,
            output_path=args.output_path,
            split=args.split,
            max_examples=args.max_examples
        )


if __name__ == "__main__":
    #main() 
    print_unsloth_examples("train_dataset", num_examples=1)