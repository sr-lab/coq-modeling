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

class TokenizationValidationCallback(TrainerCallback):
    """
    Callback to validate tokenization and detect potential issues.
    """
    
    def __init__(self, validate_every_n_steps: int = 1000, max_seq_length: Optional[int] = None):
        self.validate_every_n_steps = validate_every_n_steps
        self.max_seq_length = max_seq_length
        self.tokenization_issues = []
        
    def on_step_end(self, args, state: TrainerState, control: TrainerControl, 
                   model, **kwargs):
        """Validate tokenization periodically."""
        if state.global_step % self.validate_every_n_steps == 0:
            # Note: We can't access inputs directly in on_step_end
            # This callback would need to be modified to work with the available data
            pass
    
    def _validate_tokenization(self, inputs, model, step: int):
        """Validate tokenization for potential issues."""
        if inputs is None:
            return
            
        issues = []
        
        # Check for input_ids
        if "input_ids" in inputs:
            input_ids = inputs["input_ids"]
            if isinstance(input_ids, torch.Tensor):
                issues.extend(self._check_input_ids(input_ids, step))
        
        # Check for attention_mask
        if "attention_mask" in inputs:
            attention_mask = inputs["attention_mask"]
            if isinstance(attention_mask, torch.Tensor):
                issues.extend(self._check_attention_mask(attention_mask, step))
        
        # Check for labels if present
        if "labels" in inputs:
            labels = inputs["labels"]
            if isinstance(labels, torch.Tensor):
                issues.extend(self._check_labels(labels, step))
        
        # Check for tokenization consistency
        if "input_ids" in inputs and "attention_mask" in inputs:
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            if isinstance(input_ids, torch.Tensor) and isinstance(attention_mask, torch.Tensor):
                issues.extend(self._check_tokenization_consistency(input_ids, attention_mask, step))
        
        # Check for special tokens
        if "input_ids" in inputs and hasattr(model, 'config'):
            input_ids = inputs["input_ids"]
            if isinstance(input_ids, torch.Tensor):
                issues.extend(self._check_special_tokens(input_ids, model.config, step))
        
        if issues:
            self.tokenization_issues.extend(issues)
            _logger.warning(f"Tokenization issues at step {step}: {issues}")
    
    def _check_input_ids(self, input_ids: torch.Tensor, step: int) -> List[str]:
        """Check input_ids for potential issues."""
        issues = []
        
        # Check for empty sequences
        if input_ids.numel() == 0:
            issues.append("Empty input_ids tensor")
            return issues
        
        # Check for sequences that are too long
        if self.max_seq_length and input_ids.size(-1) > self.max_seq_length:
            issues.append(f"Sequence length {input_ids.size(-1)} exceeds max_seq_length {self.max_seq_length}")
        
        # Check for all-zero sequences (might indicate padding issues)
        if torch.all(input_ids == 0):
            issues.append("All-zero input_ids (possible padding issue)")
        
        # Check for sequences that are all the same token
        if input_ids.size(-1) > 1:
            unique_tokens = torch.unique(input_ids)
            if len(unique_tokens) == 1:
                issues.append(f"All tokens are the same: {unique_tokens[0].item()}")
        
        # Check for extreme token IDs
        min_token = input_ids.min().item()
        max_token = input_ids.max().item()
        if min_token < 0:
            issues.append(f"Negative token ID found: {min_token}")
        if max_token > 100000:  # Arbitrary large number
            issues.append(f"Unusually large token ID found: {max_token}")
        
        # Check for NaN/Inf in token IDs
        if torch.isnan(input_ids).any():
            issues.append("NaN values found in input_ids")
        if torch.isinf(input_ids).any():
            issues.append("Inf values found in input_ids")
        
        return issues
    
    def _check_attention_mask(self, attention_mask: torch.Tensor, step: int) -> List[str]:
        """Check attention_mask for potential issues."""
        issues = []
        
        # Check for empty tensor
        if attention_mask.numel() == 0:
            issues.append("Empty attention_mask tensor")
            return issues
        
        # Check for invalid values (should only be 0 or 1)
        unique_values = torch.unique(attention_mask)
        if not torch.all(torch.isin(unique_values, torch.tensor([0, 1]))):
            issues.append(f"Invalid attention_mask values: {unique_values.tolist()}")
        
        # Check for all-zero attention masks
        if torch.all(attention_mask == 0):
            issues.append("All-zero attention_mask (no tokens attended to)")
        
        # Check for all-one attention masks (might be suspicious for long sequences)
        if torch.all(attention_mask == 1) and attention_mask.size(-1) > 1000:
            issues.append("All-one attention_mask for long sequence (possible issue)")
        
        # Check for NaN/Inf
        if torch.isnan(attention_mask).any():
            issues.append("NaN values found in attention_mask")
        if torch.isinf(attention_mask).any():
            issues.append("Inf values found in attention_mask")
        
        return issues
    
    def _check_labels(self, labels: torch.Tensor, step: int) -> List[str]:
        """Check labels for potential issues."""
        issues = []
        
        # Check for empty tensor
        if labels.numel() == 0:
            issues.append("Empty labels tensor")
            return issues
        
        # Check for all -100 labels (no loss computed)
        if torch.all(labels == -100):
            issues.append("All labels are -100 (no loss will be computed)")
        
        # Check for extreme label values
        min_label = labels.min().item()
        max_label = labels.max().item()
        if min_label < -100:
            issues.append(f"Label value below -100: {min_label}")
        if max_label > 100000:  # Arbitrary large number
            issues.append(f"Unusually large label value: {max_label}")
        
        # Check for NaN/Inf
        if torch.isnan(labels).any():
            issues.append("NaN values found in labels")
        if torch.isinf(labels).any():
            issues.append("Inf values found in labels")
        
        return issues
    
    def _check_tokenization_consistency(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, step: int) -> List[str]:
        """Check consistency between input_ids and attention_mask."""
        issues = []
        
        # Check if shapes match
        if input_ids.shape != attention_mask.shape:
            issues.append(f"Shape mismatch: input_ids {input_ids.shape} vs attention_mask {attention_mask.shape}")
            return issues
        
        # Check if attention_mask is 1 where input_ids is not padding (assuming 0 is padding)
        # This is a common pattern, but might vary by tokenizer
        if input_ids.size(-1) > 0:
            # Check if there are any non-zero input_ids with zero attention_mask
            non_zero_inputs = input_ids != 0
            zero_attention = attention_mask == 0
            inconsistent = torch.logical_and(non_zero_inputs, zero_attention)
            
            if torch.any(inconsistent):
                num_inconsistent = inconsistent.sum().item()
                issues.append(f"{num_inconsistent} tokens have non-zero input_ids but zero attention_mask")
        
        return issues
    
    def _check_special_tokens(self, input_ids: torch.Tensor, config, step: int) -> List[str]:
        """Check for special token usage."""
        issues = []
        
        # Check for BOS/EOS tokens if config has them
        if hasattr(config, 'bos_token_id') and config.bos_token_id is not None:
            bos_count = (input_ids == config.bos_token_id).sum().item()
            if bos_count == 0:
                issues.append("No BOS token found in sequences")
            elif bos_count > input_ids.size(0):
                issues.append(f"Multiple BOS tokens found: {bos_count} for {input_ids.size(0)} sequences")
        
        if hasattr(config, 'eos_token_id') and config.eos_token_id is not None:
            eos_count = (input_ids == config.eos_token_id).sum().item()
            if eos_count == 0:
                issues.append("No EOS token found in sequences")
            elif eos_count > input_ids.size(0):
                issues.append(f"Multiple EOS tokens found: {eos_count} for {input_ids.size(0)} sequences")
        
        # Check for pad token if config has it
        if hasattr(config, 'pad_token_id') and config.pad_token_id is not None:
            pad_count = (input_ids == config.pad_token_id).sum().item()
            if pad_count == 0 and input_ids.size(-1) > 1:
                issues.append("No PAD tokens found despite variable sequence lengths")
        
        return issues
    
    def get_tokenization_summary(self) -> Dict[str, Any]:
        """Get a summary of all tokenization issues found."""
        return {
            "total_issues": len(self.tokenization_issues),
            "issues": self.tokenization_issues,
            "unique_issue_types": list(set(self.tokenization_issues))
        }


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


def create_debug_callbacks(config: Dict[str, Any]) -> List[TrainerCallback]:
    """
    Create a list of debugging callbacks based on configuration.
    
    Args:
        config: Configuration dictionary with callback settings
        
    Returns:
        List of TrainerCallback instances
    """
    callbacks = []
    
    # Always include NaN loss detection
    callbacks.append(NaNLossCallback(
        nan_threshold=config.get("nan_threshold", 1e6),
        save_debug_info=config.get("save_debug_info", True)
    ))
    
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
        callbacks.append(TokenizationValidationCallback(
            validate_every_n_steps=config.get("tokenization_validation_steps", 1000),
            max_seq_length=config.get("max_seq_length", None)
        ))
    
    return callbacks 