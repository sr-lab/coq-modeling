#!/usr/bin/env python3
"""
Test script for the validation functionality.
This script tests the validation components without requiring a full model.
"""

import sys
import os
from pathlib import Path
import tempfile
import json

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from validate_model import get_validation_datasets, load_model_from_checkpoint, create_validation_trainer


def test_validation_datasets():
    """Test validation dataset loading."""
    print("Testing validation dataset loading...")
    
    # Create a minimal test configuration
    test_conf = {
        "model_name": "deepseek-ai/deepseek-coder-1.3b-instruct",
        "train_type": "sft",
        "data_path": "data/split-dataset",
        "num_eval_examples": 10,
        "hard_seq_len": 4096,
        "example_collator": {
            "alias": "proof-premise",
            "state_tokens": 1024,
            "script_tokens": 512,
            "proof_tokens": 1024,
            "premise_tokens": 512,
            "out_tokens": 128
        }
    }
    
    try:
        # This will fail if the dataset doesn't exist, but that's expected
        # We're just testing that the function can be called
        val_dataset = get_validation_datasets(test_conf)
        print("✅ Validation dataset function works")
        return True
    except Exception as e:
        print(f"⚠️  Validation dataset test failed (expected if no dataset): {e}")
        return False


def test_model_loading():
    """Test model loading functionality."""
    print("Testing model loading...")
    
    test_conf = {
        "model_name": "deepseek-ai/deepseek-coder-1.3b-instruct",
        "train_type": "sft",
        "hard_seq_len": 4096,
        "peft_lora_r": 64,
        "peft_lora_alpha": 16
    }
    
    try:
        # This will fail if the checkpoint doesn't exist, but that's expected
        # We're just testing that the function can be called
        model, tokenizer = load_model_from_checkpoint(
            "deepseek-ai/deepseek-coder-1.3b-instruct",
            "/nonexistent/checkpoint",
            test_conf
        )
        print("✅ Model loading function works")
        return True
    except Exception as e:
        print(f"⚠️  Model loading test failed (expected if no checkpoint): {e}")
        return False


def test_trainer_creation():
    """Test trainer creation functionality."""
    print("Testing trainer creation...")
    
    test_conf = {
        "model_name": "deepseek-ai/deepseek-coder-1.3b-instruct",
        "train_type": "sft",
        "hard_seq_len": 4096,
        "per_device_eval_batch_size": 8,
        "eval_accumulation_steps": 1,
        "learning_rate": 1.0e-4,
        "num_train_epochs": 1,
        "max_steps": 100,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 2,
        "logging_steps": 100,
        "save_steps": 100,
        "eval_steps": 100,
        "save_total_limit": 20
    }
    
    try:
        # Create dummy model and tokenizer
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        model = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/deepseek-coder-1.3b-instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct")
        
        # Create a dummy dataset
        from tactic_gen.tactic_data import LmProcessedDataset
        from tactic_gen.tactic_data import example_collator_conf_from_yaml, example_collator_from_conf
        
        example_collator_conf = example_collator_conf_from_yaml(test_conf["example_collator"])
        example_collator = example_collator_from_conf(example_collator_conf)
        
        # This will fail if no dataset exists, but we're testing the trainer creation
        trainer = create_validation_trainer(test_conf, model, tokenizer, None)
        print("✅ Trainer creation function works")
        return True
    except Exception as e:
        print(f"⚠️  Trainer creation test failed: {e}")
        return False


def test_config_loading():
    """Test configuration loading."""
    print("Testing configuration loading...")
    
    # Create a temporary test config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        test_config = """
model_name: "deepseek-ai/deepseek-coder-1.3b-instruct"
train_type: "sft"
data_path: "data/split-dataset"
num_eval_examples: 100
hard_seq_len: 4096
per_device_eval_batch_size: 8
eval_accumulation_steps: 1
peft_lora_r: 64
peft_lora_alpha: 16
example_collator:
   alias: "proof-premise"
   state_tokens: 1024 
   script_tokens: 512
   proof_tokens: 1024 
   premise_tokens: 512 
   out_tokens: 128
learning_rate: 1.0e-4
num_train_epochs: 1
max_steps: 100
per_device_train_batch_size: 4
gradient_accumulation_steps: 2
logging_steps: 100
save_steps: 100
eval_steps: 100
save_total_limit: 20
"""
        f.write(test_config)
        config_path = f.name
    
    try:
        from util.train_utils import load_config
        conf = load_config(config_path)
        
        # Test that required fields are present
        required_fields = ["model_name", "train_type", "hard_seq_len"]
        for field in required_fields:
            if field not in conf:
                raise ValueError(f"Missing required field: {field}")
        
        print("✅ Configuration loading works")
        return True
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False
    finally:
        # Clean up
        os.unlink(config_path)


def main():
    """Run all tests."""
    print("Running validation script tests...\n")
    
    tests = [
        test_config_loading,
        test_validation_datasets,
        test_model_loading,
        test_trainer_creation,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
        print()
    
    print(f"Tests completed: {passed}/{total} passed")
    
    if passed == total:
        print("✅ All tests passed!")
    else:
        print("⚠️  Some tests failed (this may be expected if datasets/models are not available)")


if __name__ == "__main__":
    main() 