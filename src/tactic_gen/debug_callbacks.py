import torch
import numpy as np
import logging
from typing import Any, Dict, List, Optional, Union
from transformers import TrainerCallback, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import os
from pathlib import Path
import json
from datetime import datetime

_logger = logging.getLogger(__name__)

# Note: In Transformers 4.34.0, the on_step_end callback method signature is:
# def on_step_end(self, args, state, control, model, **kwargs)
# 
# The inputs and outputs parameters are not available in this version.
# To access inputs and outputs, consider using:
# 1. on_prediction_step for evaluation outputs
# 2. Custom training loop with explicit callback points
# 3. Subclassing the Trainer class
# 4. Using on_log callback to access logged metrics


class SimpleCallback(TrainerCallback):
    def __init__(self, callback_type: str, tokenizer):
        self.callback_type = callback_type
        self.tokenizer = tokenizer
    
    def on_step_begin(self, args, state, control, **kwargs):
        print(f"on_step_begin: {self.callback_type}")
        delimiters = ["[TACTIC", "[TACTIC]", " \n[TACTIC]\n ", "[TACTIC]\n"]
        for delimiter in delimiters:
            print(self.tokenizer.encode(delimiter, add_special_tokens=False))
        print("==============================")
        #print(kwargs)
        if 'train_dataloader' in kwargs:
            train_dataloader = kwargs['train_dataloader']
            for batch in train_dataloader:
                inputs = batch['input_ids']
                labels = batch['labels']
                print("inputs shape: ", inputs.shape)
                print("labels shape: ", labels.shape)
                tokenized_inputs = self.tokenizer.decode(inputs[0], skip_special_tokens=False)
                #tokenized_labels = self.tokenizer.decode(labels[0], skip_special_tokens=False)
                print("==============================")
                print("tokenized_inputs: ", tokenized_inputs)
                #print("tokenized_labels: ", tokenized_labels)
                print("==============================")
                print("inputs: ", inputs)
                print("labels: ", labels)
                print("==============================")
                break
        else:
            print("No train_dataloader found")

class TokenizerInspectionCallback(TrainerCallback):
    def __init__(self, tokenizer, check_frequency=1, max_samples_to_check=3):
        """
        Callback to inspect tokenizer behavior during training
        
        Args:
            tokenizer: The tokenizer being used
            check_frequency: Check every N steps (default: 1 for immediate debugging)
            max_samples_to_check: Number of samples to inspect per batch
        """
        self.tokenizer = tokenizer
        self.check_frequency = check_frequency
        self.max_samples_to_check = max_samples_to_check
        self.step_count = 0
    
    def on_step_begin(self, args, state, control, **kwargs):
        print("on_step_begin")
        print(kwargs)
        self.step_count += 1
        
        # Only check at specified frequency
        if self.step_count % self.check_frequency != 0:
            return
            
        # Get the current batch from kwargs
        if 'inputs' in kwargs:
            inputs = kwargs['inputs']
            self._inspect_batch(inputs, state.global_step)
    
    def _inspect_batch(self, inputs, step):
        print(f"\n=== TOKENIZER INSPECTION - Step {step} ===")
        
        # Check input_ids
        if 'input_ids' in inputs:
            input_ids = inputs['input_ids']
            print(f"Input IDs shape: {input_ids.shape}")
            print(f"Input IDs dtype: {input_ids.dtype}")
            print(f"Input IDs device: {input_ids.device}")
            
            # Check for problematic values
            unique_ids = torch.unique(input_ids)
            print(f"Unique token count: {len(unique_ids)}")
            print(f"Min token ID: {input_ids.min().item()}")
            print(f"Max token ID: {input_ids.max().item()}")
            
            # Check for out-of-vocabulary tokens
            vocab_size = self.tokenizer.vocab_size
            oov_mask = input_ids >= vocab_size
            if oov_mask.any():
                print(f"⚠️  WARNING: Found {oov_mask.sum().item()} out-of-vocabulary tokens!")
                print(f"   Vocab size: {vocab_size}, Max token ID: {input_ids.max().item()}")
            
            # Check for special tokens
            pad_token_id = getattr(self.tokenizer, 'pad_token_id', None)
            eos_token_id = getattr(self.tokenizer, 'eos_token_id', None)
            bos_token_id = getattr(self.tokenizer, 'bos_token_id', None)
            unk_token_id = getattr(self.tokenizer, 'unk_token_id', None)
            
            print(f"Special tokens - PAD: {pad_token_id}, EOS: {eos_token_id}, BOS: {bos_token_id}, UNK: {unk_token_id}")
            
            # Inspect first few samples
            batch_size = min(self.max_samples_to_check, input_ids.shape[0])
            for i in range(batch_size):
                sample_ids = input_ids[i]
                
                # Decode the sample
                try:
                    decoded_text = self.tokenizer.decode(sample_ids, skip_special_tokens=False)
                    print(f"\nSample {i}:")
                    print(f"  Length: {len(sample_ids)}")
                    print(f"  Decoded: {repr(decoded_text[:100])}{'...' if len(decoded_text) > 100 else ''}")
                    
                    # Check for None or empty decoded text
                    if not decoded_text or decoded_text.strip() == "":
                        print(f"  ⚠️  WARNING: Sample {i} decoded to empty text!")
                        
                except Exception as e:
                    print(f"  ❌ ERROR decoding sample {i}: {e}")
        
        # Check labels if present
        if 'labels' in inputs:
            labels = inputs['labels']
            print(f"\nLabels shape: {labels.shape}")
            print(f"Labels dtype: {labels.dtype}")
            
            # Check for -100 (ignore index)
            ignore_mask = labels == -100
            print(f"Ignored tokens (labels == -100): {ignore_mask.sum().item()}/{labels.numel()}")
            
            # Check for valid label range
            valid_labels = labels[labels != -100]
            if len(valid_labels) > 0:
                print(f"Valid labels range: {valid_labels.min().item()} to {valid_labels.max().item()}")
                
                # Check if labels are in valid range
                if valid_labels.max().item() >= vocab_size:
                    print(f"⚠️  WARNING: Labels contain out-of-vocabulary indices!")
            else:
                print(f"⚠️  WARNING: All labels are -100 (ignored)!")
        
        # Check attention mask
        if 'attention_mask' in inputs:
            attention_mask = inputs['attention_mask']
            print(f"\nAttention mask shape: {attention_mask.shape}")
            active_tokens = attention_mask.sum(dim=1)
            print(f"Active tokens per sample: min={active_tokens.min().item()}, max={active_tokens.max().item()}, mean={active_tokens.float().mean().item():.1f}")
        
        print("=" * 50)


class NaNLossCallback(TrainerCallback):
    """
    Callback to detect and handle NaN loss during training.
    """
    
    def __init__(self, nan_threshold: float = 1e6, save_debug_info: bool = True):
        self.nan_threshold = nan_threshold
        self.save_debug_info = save_debug_info
        self.nan_detected = False
        self.debug_info = []
        
    def on_step_end(self, args, state: TrainerState, control: TrainerControl, 
                   model, **kwargs):
        """Check for NaN loss after each step."""
        # Access loss from the trainer's log history
        if hasattr(state, 'log_history') and state.log_history:
            latest_log = state.log_history[-1]
            loss = latest_log.get("loss", None)
            
            if loss is not None:
                if torch.isnan(torch.tensor(loss)) or torch.isinf(torch.tensor(loss)) or loss > self.nan_threshold:
                    self.nan_detected = True
                    _logger.error(f"NaN/Inf loss detected at step {state.global_step}: {loss}")
                    
                    if self.save_debug_info:
                        debug_info = self._collect_debug_info(model, state, loss)
                        self.debug_info.append(debug_info)
                        self._save_debug_info(debug_info, state.global_step)
                    
                    # Optionally stop training
                    control.should_training_stop = True
    
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """Alternative method to access loss through logs."""
        if logs is not None and "loss" in logs:
            loss = logs["loss"]
            if torch.isnan(torch.tensor(loss)) or torch.isinf(torch.tensor(loss)) or loss > self.nan_threshold:
                self.nan_detected = True
                _logger.error(f"NaN/Inf loss detected at step {state.global_step}: {loss}")
                
                if self.save_debug_info:
                    debug_info = self._collect_debug_info(None, state, loss)
                    self.debug_info.append(debug_info)
                    self._save_debug_info(debug_info, state.global_step)
                
                # Optionally stop training
                control.should_training_stop = True
                
    def _collect_debug_info(self, model, state, loss) -> Dict[str, Any]:
        """Collect debugging information when NaN loss is detected."""
        debug_info = {
            "step": state.global_step,
            "epoch": state.epoch,
            "loss": loss,
            "timestamp": datetime.now().isoformat(),
            "model_gradients": {},
            "model_stats": {}
        }
        
        # Check model gradients
        if model is not None and hasattr(model, 'named_parameters'):
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        debug_info["model_gradients"][name] = {
                            "grad_norm": grad_norm,
                            "param_norm": param.norm().item(),
                            "has_nan_grad": torch.isnan(param.grad).any().item(),
                            "has_inf_grad": torch.isinf(param.grad).any().item()
                        }
        
        # Check model parameter statistics
        if model is not None and hasattr(model, 'named_parameters'):
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param_norm = param.norm().item()
                    if torch.isnan(param_norm) or torch.isinf(param_norm):
                        debug_info["model_stats"][name] = {
                            "param_norm": param_norm,
                            "has_nan": torch.isnan(param).any().item(),
                            "has_inf": torch.isinf(param).any().item()
                        }
        
        return debug_info
    
    def _save_debug_info(self, debug_info: Dict[str, Any], step: int):
        """Save debugging information to file."""
        debug_dir = Path("debug_info")
        debug_dir.mkdir(exist_ok=True)
        
        filename = debug_dir / f"nan_debug_step_{step}.json"
        with open(filename, 'w') as f:
            json.dump(debug_info, f, indent=2, default=str)
        
        _logger.info(f"Debug info saved to {filename}")


class GradientMonitoringCallback(TrainerCallback):
    """
    Callback to monitor gradient statistics during training.
    """
    
    def __init__(self, log_every_n_steps: int = 100, save_gradients: bool = False):
        self.log_every_n_steps = log_every_n_steps
        self.save_gradients = save_gradients
        self.gradient_history = []
        
    def on_step_end(self, args, state: TrainerState, control: TrainerControl, 
                   model, **kwargs):
        """Monitor gradients after each step."""
        if state.global_step % self.log_every_n_steps == 0:
            grad_stats = self._compute_gradient_stats(model)
            self.gradient_history.append({
                "step": state.global_step,
                "grad_stats": grad_stats
            })
            
            # Log gradient statistics
            _logger.info(f"Step {state.global_step} - Gradient stats: {grad_stats}")
            
            # Check for gradient explosion/vanishing
            if grad_stats["max_grad_norm"] > 10.0:
                _logger.warning(f"Large gradient norm detected: {grad_stats['max_grad_norm']}")
            
            if grad_stats["max_grad_norm"] < 1e-6:
                _logger.warning(f"Very small gradient norm detected: {grad_stats['max_grad_norm']}")
    
    def _compute_gradient_stats(self, model) -> Dict[str, float]:
        """Compute gradient statistics for the model."""
        grad_norms = []
        param_norms = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                param_norm = param.norm().item()
                grad_norms.append(grad_norm)
                param_norms.append(param_norm)
        
        if not grad_norms:
            return {"max_grad_norm": 0.0, "mean_grad_norm": 0.0, "max_param_norm": 0.0}
        
        return {
            "max_grad_norm": max(grad_norms),
            "mean_grad_norm": np.mean(grad_norms),
            "std_grad_norm": np.std(grad_norms),
            "max_param_norm": max(param_norms),
            "mean_param_norm": np.mean(param_norms)
        }


class LossMonitoringCallback(TrainerCallback):
    """
    Callback to monitor loss statistics and detect anomalies.
    """
    
    def __init__(self, window_size: int = 100, anomaly_threshold: float = 3.0):
        self.window_size = window_size
        self.anomaly_threshold = anomaly_threshold
        self.loss_history = []
        
    def on_step_end(self, args, state: TrainerState, control: TrainerControl, 
                   model, **kwargs):
        """Monitor loss after each step."""
        # Access loss from the trainer's log history
        if hasattr(state, 'log_history') and state.log_history:
            latest_log = state.log_history[-1]
            loss = latest_log.get("loss", None)
            
            if loss is not None:
                if isinstance(loss, torch.Tensor):
                    loss = loss.item()
                
                self.loss_history.append(loss)
                
                # Keep only the last window_size losses
                if len(self.loss_history) > self.window_size:
                    self.loss_history = self.loss_history[-self.window_size:]
                
                # Check for anomalies if we have enough history
                if len(self.loss_history) >= 10:
                    recent_losses = self.loss_history[-10:]
                    mean_loss = np.mean(recent_losses)
                    std_loss = np.std(recent_losses)
                    
                    if std_loss > 0:
                        z_score = abs(loss - mean_loss) / std_loss
                        if z_score > self.anomaly_threshold:
                            _logger.warning(f"Anomalous loss detected at step {state.global_step}: "
                                          f"loss={loss:.6f}, z_score={z_score:.2f}")
                
                # Log loss statistics periodically
                if state.global_step % 100 == 0:
                    if len(self.loss_history) > 0:
                        _logger.info(f"Step {state.global_step} - Loss: {loss:.6f}, "
                                   f"Recent mean: {np.mean(self.loss_history[-10:]):.6f}")

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """Alternative method to access loss through logs."""
        if logs is not None and "loss" in logs:
            loss = logs["loss"]
            if isinstance(loss, torch.Tensor):
                loss = loss.item()
            
            self.loss_history.append(loss)
            
            # Keep only the last window_size losses
            if len(self.loss_history) > self.window_size:
                self.loss_history = self.loss_history[-self.window_size:]
            
            # Check for anomalies if we have enough history
            if len(self.loss_history) >= 10:
                recent_losses = self.loss_history[-10:]
                mean_loss = np.mean(recent_losses)
                std_loss = np.std(recent_losses)
                
                if std_loss > 0:
                    z_score = abs(loss - mean_loss) / std_loss
                    if z_score > self.anomaly_threshold:
                        _logger.warning(f"Anomalous loss detected at step {state.global_step}: "
                                      f"loss={loss:.6f}, z_score={z_score:.2f}")
            
            # Log loss statistics periodically
            if state.global_step % 100 == 0:
                if len(self.loss_history) > 0:
                    _logger.info(f"Step {state.global_step} - Loss: {loss:.6f}, "
                               f"Recent mean: {np.mean(self.loss_history[-10:]):.6f}")


class DataValidationCallback(TrainerCallback):
    """
    Callback to validate input data and detect potential issues.
    """
    
    def __init__(self, validate_every_n_steps: int = 1000):
        self.validate_every_n_steps = validate_every_n_steps
        
    def on_step_end(self, args, state: TrainerState, control: TrainerControl, 
                   model, **kwargs):
        """Validate input data periodically."""
        if state.global_step % self.validate_every_n_steps == 0:
            # Note: We can't access inputs directly in on_step_end
            # This callback would need to be modified to work with the available data
            pass
    
    def _validate_inputs(self, inputs, step: int):
        """Validate input tensors for potential issues."""
        if inputs is None:
            return
            
        issues = []
        
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                # Check for NaN/Inf values
                if torch.isnan(value).any():
                    issues.append(f"NaN values found in {key}")
                
                if torch.isinf(value).any():
                    issues.append(f"Inf values found in {key}")
                
                # Check for extreme values
                if value.numel() > 0:
                    min_val = value.min().item()
                    max_val = value.max().item()
                    
                    if abs(min_val) > 1e6 or abs(max_val) > 1e6:
                        issues.append(f"Extreme values in {key}: min={min_val}, max={max_val}")
                
                # Check for empty tensors
                if value.numel() == 0:
                    issues.append(f"Empty tensor found in {key}")
        
        if issues:
            _logger.warning(f"Data validation issues at step {step}: {issues}")


class ModelStateCallback(TrainerCallback):
    """
    Callback to monitor model state and parameters.
    """
    
    def __init__(self, log_every_n_steps: int = 500):
        self.log_every_n_steps = log_every_n_steps
        
    def on_step_end(self, args, state: TrainerState, control: TrainerControl, 
                   model, **kwargs):
        """Monitor model state periodically."""
        if state.global_step % self.log_every_n_steps == 0:
            self._log_model_stats(model, state.global_step)
    
    def _log_model_stats(self, model, step: int):
        """Log model statistics."""
        total_params = 0
        trainable_params = 0
        param_norms = []
        
        for name, param in model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                param_norms.append(param.norm().item())
        
        if param_norms:
            _logger.info(f"Step {step} - Model stats: "
                        f"Total params: {total_params:,}, "
                        f"Trainable: {trainable_params:,}, "
                        f"Max param norm: {max(param_norms):.6f}, "
                        f"Mean param norm: {np.mean(param_norms):.6f}")


class LearningRateCallback(TrainerCallback):
    """
    Callback to monitor learning rate changes.
    """
    
    def __init__(self, log_every_n_steps: int = 100):
        self.log_every_n_steps = log_every_n_steps
        self.last_lr = None
        
    def on_step_end(self, args, state: TrainerState, control: TrainerControl, 
                   model, **kwargs):
        """Monitor learning rate changes."""
        if state.global_step % self.log_every_n_steps == 0:
            current_lr = self._get_current_lr(model)
            if current_lr != self.last_lr:
                _logger.info(f"Step {state.global_step} - Learning rate changed: {current_lr}")
                self.last_lr = current_lr
    
    def _get_current_lr(self, model) -> float:
        """Get current learning rate from optimizer."""
        if hasattr(model, 'optimizer') and model.optimizer is not None:
            return model.optimizer.param_groups[0]['lr']
        return 0.0


class MemoryMonitoringCallback(TrainerCallback):
    """
    Callback to monitor GPU memory usage.
    """
    
    def __init__(self, log_every_n_steps: int = 100):
        self.log_every_n_steps = log_every_n_steps
        
    def on_step_end(self, args, state: TrainerState, control: TrainerControl, 
                   model, **kwargs):
        """Monitor memory usage periodically."""
        if state.global_step % self.log_every_n_steps == 0:
            self._log_memory_stats()
    
    def _log_memory_stats(self):
        """Log GPU memory statistics."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
            
            _logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, "
                        f"Reserved: {reserved:.2f}GB, "
                        f"Max allocated: {max_allocated:.2f}GB")


def create_debug_callbacks(config: Dict[str, Any], tokenizer) -> List[TrainerCallback]:
    """
    Create a list of debugging callbacks based on configuration.
    
    Args:
        config: Configuration dictionary with callback settings
        
    Returns:
        List of TrainerCallback instances
    """
    callbacks = []
    
    # Always include NaN loss detection
    # callbacks.append(NaNLossCallback(
    #     nan_threshold=config.get("nan_threshold", 1e6),
    #     save_debug_info=config.get("save_debug_info", True)
    # ))
    
    # Add gradient monitoring
    if config.get("monitor_gradients", True):
        callbacks.append(GradientMonitoringCallback(
            log_every_n_steps=config.get("gradient_log_steps", 100),
            save_gradients=config.get("save_gradients", False)
        ))
    
    # Add loss monitoring
    if config.get("monitor_loss", True):
        callbacks.append(LossMonitoringCallback(
            window_size=config.get("loss_window_size", 100),
            anomaly_threshold=config.get("loss_anomaly_threshold", 3.0)
        ))
    
    # Add data validation
    if config.get("validate_data", True):
        callbacks.append(DataValidationCallback(
            validate_every_n_steps=config.get("data_validation_steps", 1000)
        ))
    
    # Add model state monitoring
    if config.get("monitor_model_state", True):
        callbacks.append(ModelStateCallback(
            log_every_n_steps=config.get("model_state_log_steps", 500)
        ))
    
    # Add learning rate monitoring
    if config.get("monitor_lr", True):
        callbacks.append(LearningRateCallback(
            log_every_n_steps=config.get("lr_log_steps", 100)
        ))
    
    # Add memory monitoring
    if config.get("monitor_memory", True):
        callbacks.append(MemoryMonitoringCallback(
            log_every_n_steps=config.get("memory_log_steps", 100)
        ))
    
    # Add tokenization validation
    if config.get("validate_tokenization", True):
        callbacks.append(TokenizerInspectionCallback(
            tokenizer=tokenizer,
            check_frequency=config.get("tokenization_check_frequency", 1),
            max_samples_to_check=config.get("max_samples_to_check", 3)
        ))
    
    return callbacks 