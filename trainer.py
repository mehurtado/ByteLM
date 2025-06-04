# trainer.py

import torch
import torch.optim as optim
import torch.profiler
from torch.amp import GradScaler, autocast
from pathlib import Path
import traceback # For detailed error logging in checkpoint loading
import numpy as np
from torch.utils.data import DataLoader, Subset, Dataset

# Assuming other modules are in the same directory or accessible in Python path
from model import ByteLLM_GrugV3
from predictor import Predictor # For interim testing
from utils import ensure_dir
from config import CONFIG_V3 # For default train_config values and interim test params

class Trainer:
    """
    Manages the training and evaluation process for the ByteLLM_GrugV3 model.
    """
    def __init__(self, model: ByteLLM_GrugV3,
                 train_dataloader: torch.utils.data.DataLoader,
                 val_dataloader: torch.utils.data.DataLoader | None,
                 optimizer: torch.optim.Optimizer,
                 criterion: torch.nn.Module,
                 device: torch.device,
                 checkpoint_dir: str or Path,
                 model_name: str,
                 scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
                 train_config: dict = None):
        """
        Initializes the Trainer.
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_name = model_name
        self.scheduler = scheduler
        
        self.train_config = train_config if train_config is not None else CONFIG_V3.copy()

        ensure_dir(self.checkpoint_dir)
        if self.train_config.get("enable_profiler", False):
            profiler_log_dir = Path(self.train_config.get("profiler_log_dir", "./profiler_logs_grug_v3"))
            ensure_dir(profiler_log_dir)
            ensure_dir(profiler_log_dir / "train") 
            ensure_dir(profiler_log_dir / "eval")  

        self.current_global_step = 0 
        self.best_val_loss = float('inf') # Initialize best_val_loss here
        
        self.use_amp = self.train_config.get("use_amp", False) and self.device.type == 'cuda'
        self.scaler = GradScaler(enabled=self.use_amp)
        
        if self.use_amp:
            print("Automatic Mixed Precision (AMP) is ENABLED for training.")
        else:
            print("Automatic Mixed Precision (AMP) is DISABLED for training (either use_amp=False or device is not CUDA).")

    def _perform_lr_warmup(self):
        """Handles learning rate warmup at the beginning of training."""
        if not self.train_config.get("use_lr_warmup", False) or \
           self.current_global_step >= self.train_config.get("lr_warmup_steps", 0):
            return

        warmup_steps = self.train_config["lr_warmup_steps"]
        target_lr = self.train_config["learning_rate"] 
        init_factor = self.train_config.get("lr_warmup_init_factor", 0.01)

        if warmup_steps == 0: 
            lr_scale = 1.0
        elif self.current_global_step == 0: 
            lr_scale = init_factor
        else:
            lr_scale = init_factor + (1.0 - init_factor) * (self.current_global_step / warmup_steps)
        
        lr_scale = min(lr_scale, 1.0) 

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = target_lr * lr_scale
        
        if self.current_global_step == 0 or \
           (self.current_global_step + 1) % max(1, warmup_steps // 10) == 0 or \
           self.current_global_step == warmup_steps - 1:
            print(f"Warmup Step {self.current_global_step + 1}/{warmup_steps}, Current LR: {self.optimizer.param_groups[0]['lr']:.2e}")
        
        if self.current_global_step == warmup_steps -1: 
             print(f"Warmup finished. LR will be managed by scheduler or remain at target: {target_lr:.2e}")


    def _run_training_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        """Performs a single training step (forward, backward, optimize)."""
        inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
        
        self.optimizer.zero_grad(set_to_none=True) 
        
        with autocast(device_type=self.device.type, enabled=self.use_amp):
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
        
        self.scaler.scale(loss).backward()
        
        clip_val = self.train_config.get("clip_grad_norm_value")
        if clip_val is not None and clip_val > 0:
            self.scaler.unscale_(self.optimizer) 
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_val)
            
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()

    def run_interim_test(self, epoch_num: int, batch_idx: int):
        """Runs a quick generation test during training."""
        print(f"\n--- Interim Test @ Epoch {epoch_num+1}, Batch {batch_idx+1} (Global Step: {self.current_global_step}) ---")
        self.model.eval() 

        interim_gen_config = {
            "generation_temperature": self.train_config.get("interim_test_temperature", 1.0),
            "generation_top_k": self.train_config.get("interim_test_top_k", 0)
        }
        model_cfg_for_pred = {
            "sequence_length": self.model.config.get("sequence_length", self.train_config.get("sequence_length")),
            "max_positional_encoding_len": self.model.config.get("max_positional_encoding_len", self.train_config.get("max_positional_encoding_len")),
        }
        
        interim_predictor = Predictor(self.model, self.device, interim_gen_config, model_cfg_for_pred)
        
        seed_text = "The meaning of life is " 
        seed_bytes = seed_text.encode('utf-8')
        print(f"Seed: '{seed_text}'")
        
        generated_bytes = interim_predictor.generate_sequence(seed_bytes, length=128) 
        
        try:
            generated_text = generated_bytes.decode('utf-8', errors='replace')
            print(f"Generated (128 bytes): \"{generated_text}\"")
            with open("grug_outputs.txt", "a") as f:
                f.write(f"\n--- Sample @ Step {self.current_global_step} ---\n")
                f.write(f"Seed: '{seed_text}'\n")
                f.write(f"Output: {generated_text.strip()}\n")
                f.write("--- End Sample ---\n")
        except Exception as e:
            print(f"Error decoding generated bytes for interim test: {e}")
            print(f"Raw generated bytes: {generated_bytes}")
            
        self.model.train() 
        print(f"--- End Interim Test ---\n")

    def train_epoch(self, epoch_num: int, profiler_context: torch.profiler.profile | None = None):
        """Trains the model for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(self.train_dataloader)

        if num_batches == 0:
            print(f"Epoch {epoch_num+1}: Training dataloader is empty. Skipping training for this epoch.")
            return float('inf') 

        for batch_idx, (inputs, targets) in enumerate(self.train_dataloader):
            self._perform_lr_warmup() 

            # Step the scheduler per batch if it's not ReduceLROnPlateau and warmup is done
            is_after_warmup = not self.train_config.get("use_lr_warmup", False) or \
                              self.current_global_step >= self.train_config.get("lr_warmup_steps", 0)

            if self.scheduler and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) and is_after_warmup:
                self.scheduler.step()
            
            loss_item = self._run_training_step(inputs, targets)
            epoch_loss += loss_item
            
            if profiler_context:
                profiler_context.step() 

            self.current_global_step += 1 
            
            validate_batch_freq = self.train_config.get("validate_and_checkpoint_best_every_batches", 0)
            if validate_batch_freq > 0 and \
               (self.current_global_step % validate_batch_freq == 0) and \
               self.val_dataloader and len(self.val_dataloader) > 0:
                print(f"\n--- Interim Validation @ Global Step: {self.current_global_step} (Epoch: {epoch_num+1}, Batch: {batch_idx+1}) ---")
                interim_val_loss = self.evaluate_epoch(epoch_num) 
                print(f"--- Interim Validation Loss: {interim_val_loss:.4f} (Current Best: {self.best_val_loss:.4f}, Global Step: {self.current_global_step}) ---")
                
                if interim_val_loss < self.best_val_loss:
                    self.best_val_loss = interim_val_loss
                    print(f"New best GrugV3 validation loss during interim validation: {self.best_val_loss:.4f} (Global Step: {self.current_global_step}). Saving best model...")
                    self.save_checkpoint(epoch=epoch_num, val_loss=self.best_val_loss, is_best=True)
                
                self.model.train() 

            if (batch_idx + 1) % self.train_config.get("print_every", 100) == 0 or (batch_idx + 1) == num_batches:
                current_lr = self.optimizer.param_groups[0]['lr'] if self.optimizer.param_groups else 0.0
                print(f"Epoch {epoch_num+1}/{self.train_config['num_epochs']}, "
                      f"Batch {batch_idx+1}/{num_batches}, "
                      f"Train Loss: {loss_item:.4f}, LR: {current_lr:.2e}, "
                      f"Global Step: {self.current_global_step}")

            test_interval_batches = self.train_config.get("test_every_batches", 0)
            if test_interval_batches > 0 and (self.current_global_step % test_interval_batches == 0) and self.current_global_step > 0:
                self.run_interim_test(epoch_num, batch_idx)

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            
        return epoch_loss / num_batches if num_batches > 0 else float('inf')

    def evaluate_epoch(self, epoch_num: int, profiler_context: torch.profiler.profile | None = None):
        """Evaluates the model on the validation set for one epoch."""
        self.model.eval()
        val_loss = 0.0

        if not self.val_dataloader:
            # print(f"Epoch {epoch_num+1}: Validation dataloader is not available. Skipping validation.") # Logging handled by caller
            return float('inf')

        # Sampling logic starts here
        eval_dataloader = self.val_dataloader
        try:
            # Ensure self.val_dataloader.dataset exists and is a Dataset instance itself or wraps one
            # Common case: val_dataloader.dataset is already the Dataset
            # Less common: val_dataloader.dataset is a Subset, then .dataset is the original Dataset
            # Safest check:
            current_dataset_obj = self.val_dataloader.dataset
            original_val_dataset = None
            if isinstance(current_dataset_obj, Dataset): # Covers direct Dataset and Subset (Subset is a Dataset)
                 # If it's a Subset, current_dataset_obj.dataset gives the underlying Dataset
                 # If it's already the main Dataset, this might be an issue if it doesn't have a .dataset attr
                 # Let's assume if it's a Subset, we want the original one it was created from.
                 if hasattr(current_dataset_obj, 'dataset') and isinstance(current_dataset_obj.dataset, Dataset):
                      original_val_dataset = current_dataset_obj.dataset # This gets the original dataset from a Subset
                 else: # It's likely the direct Dataset instance
                      original_val_dataset = current_dataset_obj

            if original_val_dataset is None or not isinstance(original_val_dataset, Dataset):
                print(f"Warning: Could not access a valid original Dataset for sampling in evaluate_epoch. Using full val_dataloader.")
            else:
                num_total_val_samples = len(original_val_dataset)
                num_samples_to_use = self.train_config.get("val_samples_to_use", 100000) # Get from config or default

                if num_total_val_samples > num_samples_to_use:
                    selected_indices = np.random.choice(num_total_val_samples, num_samples_to_use, replace=False)
                else:
                    selected_indices = np.arange(num_total_val_samples)

                sampled_val_subset = Subset(original_val_dataset, selected_indices.tolist())

                eval_dataloader = DataLoader(
                    sampled_val_subset,
                    batch_size=self.val_dataloader.batch_size,
                    num_workers=self.val_dataloader.num_workers,
                    pin_memory=self.val_dataloader.pin_memory,
                    shuffle=False # Usually False for validation
                )
                print(f"Using a sampled validation set of {len(sampled_val_subset)} samples out of {num_total_val_samples} for evaluation.")
        except AttributeError as e:
            print(f"Warning: Error during validation set sampling (AttributeError: {e}). Defaulting to full validation set. Check Dataloader and Dataset structure.")
        except Exception as e: # Catch any other unexpected error during sampling
            print(f"Warning: An unexpected error occurred during validation set sampling: {e}. Defaulting to full validation set.")


        num_val_batches = len(eval_dataloader)
        if num_val_batches == 0:
            # print(f"Epoch {epoch_num+1}: Validation dataloader is empty. Skipping validation.") # Logging handled by caller
            return float('inf')

        active_profiler_steps = 0
        if profiler_context:
            active_profiler_steps = min(self.train_config.get("profiler_schedule_active", 5), num_val_batches)

        with torch.no_grad():
            for batch_idx_eval, (inputs, targets) in enumerate(eval_dataloader): # Use eval_dataloader
                inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                with autocast(device_type=self.device.type, enabled=self.use_amp): 
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                val_loss += loss.item()
                
                if profiler_context and batch_idx_eval < active_profiler_steps:
                     profiler_context.step() 

        avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else float('inf')
        
        is_after_warmup = not self.train_config.get("use_lr_warmup",False) or \
                          self.current_global_step >= self.train_config.get("lr_warmup_steps",0)
        
        if self.scheduler and isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau) and is_after_warmup:
            self.scheduler.step(avg_val_loss)
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        return avg_val_loss

    def train(self, num_epochs: int):
        """Main training loop."""
        print(f"Starting GrugV3 training with model {self.model_name} on device {self.device}...")
        self.model.to(self.device) 
        
        start_epoch = 0
        # self.best_val_loss is already initialized in __init__

        resume_path_str = self.train_config.get("resume_from_checkpoint")
        if resume_path_str:
            loaded_info = self.load_checkpoint(resume_path_str)
            if loaded_info: 
                start_epoch = loaded_info.get('epoch', -1) + 1
                self.current_global_step = loaded_info.get('current_global_step', 0)
                print(f"Resuming GrugV3 training from epoch {start_epoch}. Global step set to {self.current_global_step}.")
                
                if self.train_config.get("reset_best_val_loss_on_resume", False):
                    self.best_val_loss = float('inf') # Reset self.best_val_loss
                    print("Best validation loss reset on resume.")
                elif loaded_info.get('loss') is not None: # Fallback for older checkpoints or if best_val_loss key is missing
                    # Prefer 'best_val_loss' if present in checkpoint, else 'loss'
                    self.best_val_loss = loaded_info.get('best_val_loss', loaded_info.get('loss', float('inf')))
                
                if self.use_amp and 'scaler_state_dict' in loaded_info and loaded_info['scaler_state_dict']:
                    try:
                        self.scaler.load_state_dict(loaded_info['scaler_state_dict'])
                        print("GradScaler state loaded successfully.")
                    except Exception as e:
                        print(f"Warning: Could not load GradScaler state from checkpoint: {e}")
            else:
                print(f"Could not load checkpoint from {resume_path_str}. Starting fresh.")
        else:
            print("No checkpoint specified to resume from, or path was invalid. Starting fresh.")
            self.current_global_step = 0 
            self.best_val_loss = float('inf') # Ensure it's reset if not resuming or if resume fails before this point

        for epoch in range(start_epoch, num_epochs):
            print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
            
            train_prof = None
            eval_prof = None
            profiler_active_this_epoch = (self.train_config.get("enable_profiler", False) and
                                          epoch == self.train_config.get("profile_epoch_target", 0))

            if profiler_active_this_epoch:
                prof_log_dir = Path(self.train_config.get("profiler_log_dir", "./profiler_logs_grug_v3"))
                p_wait = self.train_config.get("profiler_schedule_wait", 1)
                p_warmup = self.train_config.get("profiler_schedule_warmup", 1)
                p_active = self.train_config.get("profiler_schedule_active", 3) 
                p_repeat = self.train_config.get("profiler_schedule_repeat", 1)
                
                train_schedule = torch.profiler.schedule(wait=p_wait, warmup=p_warmup, active=p_active, repeat=p_repeat)
                train_prof = torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA] if self.device.type == 'cuda' else [torch.profiler.ProfilerActivity.CPU],
                    schedule=train_schedule,
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(str(prof_log_dir / "train")),
                    record_shapes=True, profile_memory=True, with_stack=True
                )
                
                eval_active_steps = min(p_active, len(self.val_dataloader) if self.val_dataloader else 1)
                eval_schedule = torch.profiler.schedule(wait=0, warmup=0, active=eval_active_steps, repeat=1) # Corrected eval schedule
                eval_prof = torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA] if self.device.type == 'cuda' else [torch.profiler.ProfilerActivity.CPU],
                    schedule=eval_schedule,
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(str(prof_log_dir / "eval")),
                    record_shapes=True, profile_memory=True, with_stack=True
                )
                print(f"--- Profiler activated for Epoch {epoch+1} ---")
                if train_prof: train_prof.start()

            avg_train_loss = self.train_epoch(epoch, profiler_context=train_prof)
            if train_prof: 
                train_prof.stop()
                print(f"Training Profiler traces for Epoch {epoch+1} saved to: {prof_log_dir / 'train'}")

            print(f"\n--- End of Epoch {epoch+1} Evaluation ---")
            if eval_prof: eval_prof.start()
            current_val_loss = self.evaluate_epoch(epoch, profiler_context=eval_prof) 
            if eval_prof: 
                eval_prof.stop()
                print(f"Evaluation Profiler traces for Epoch {epoch+1} saved to: {prof_log_dir / 'eval'}")
            
            print(f"Epoch {epoch+1}/{num_epochs}, Avg Train Loss: {avg_train_loss:.4f}, End-of-Epoch Validation Loss: {current_val_loss:.4f} (Current Best: {self.best_val_loss:.4f})")

            is_after_warmup = not self.train_config.get("use_lr_warmup",False) or \
                              self.current_global_step >= self.train_config.get("lr_warmup_steps",0)
            if self.scheduler and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau) and is_after_warmup:
                self.scheduler.step()

            self.model.train() 

            if current_val_loss < self.best_val_loss: 
                self.best_val_loss = current_val_loss 
                print(f"New best GrugV3 validation loss at end of epoch: {self.best_val_loss:.4f}. Saving best model...")
                self.save_checkpoint(epoch, self.best_val_loss, is_best=True)
            
            epoch_checkpoint_filename = f"{self.model_name}_epoch_{epoch+1}.pth"
            self.save_checkpoint(epoch, current_val_loss, is_best=False, custom_filename=epoch_checkpoint_filename)

        print("\nGrugV3 Training finished.")

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False, custom_filename: str = None):
        """Saves a model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': val_loss, 
            'config': self.model.config, 
            'train_config': self.train_config, 
            'current_global_step': self.current_global_step,
            'best_val_loss': self.best_val_loss 
        }
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        if self.use_amp: 
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        if is_best:
            filename = f"{self.model_name}_best.pth"
        elif custom_filename:
            filename = custom_filename
        else: 
            filename = f"{self.model_name}_epoch_{epoch+1}_generic.pth"
            
        filepath = self.checkpoint_dir / filename
        try:
            torch.save(checkpoint, filepath)
            print(f"GrugV3 Checkpoint saved to {filepath} (Epoch {epoch+1}, Val Loss: {val_loss:.4f})")
        except Exception as e:
            print(f"Error saving checkpoint {filepath}: {e}")


    def load_checkpoint(self, checkpoint_path_str: str) -> dict | None:
        """Loads a model checkpoint."""
        load_path = Path(checkpoint_path_str)
        if not load_path.is_file():
            print(f"Warning: GrugV3 checkpoint path '{load_path}' not found or is not a file.")
            return None
        
        try:
            print(f"Loading GrugV3 checkpoint from: {load_path}")
            checkpoint = torch.load(load_path, map_location=self.device)

            loaded_model_cfg = checkpoint.get('config')
            if not loaded_model_cfg:
                print("Warning: Checkpoint does not contain model 'config'. Model architecture might be incompatible.")
            
            loaded_train_cfg = checkpoint.get('train_config')
            if loaded_train_cfg and self.train_config.get("use_amp") != loaded_train_cfg.get("use_amp"):
                print("Warning: AMP setting 'use_amp' mismatch between current training config and checkpoint's training config.")

            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            if 'optimizer_state_dict' in checkpoint and self.optimizer:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print("Optimizer state loaded.")
                except ValueError as e: 
                    print(f"Warning: Could not load optimizer state, possibly due to model structure changes: {e}")
            
            if 'scheduler_state_dict' in checkpoint and self.scheduler:
                try:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    print("Scheduler state loaded.")
                except Exception as e: 
                    print(f"Warning: Could not load scheduler state: {e}")
            
            if not self.train_config.get("reset_best_val_loss_on_resume", False):
                # Prefer 'best_val_loss' from checkpoint, then 'loss', then float('inf')
                self.best_val_loss = checkpoint.get('best_val_loss', checkpoint.get('loss', float('inf')))
                print(f"Best validation loss loaded/set from checkpoint: {self.best_val_loss:.4f}")


            print(f"GrugV3 Checkpoint loaded successfully from {load_path}.")
            return {
                'epoch': checkpoint.get('epoch', -1), 
                'loss': checkpoint.get('loss', float('inf')), 
                'config': loaded_model_cfg,
                'train_config': loaded_train_cfg,
                'current_global_step': checkpoint.get('current_global_step', 0),
                'scaler_state_dict': checkpoint.get('scaler_state_dict'),
                'best_val_loss': checkpoint.get('best_val_loss', float('inf'))
            }
            
        except Exception as e:
            print(f"Error loading GrugV3 checkpoint {load_path}: {e}")
            traceback.print_exc()
            return None

if __name__ == '__main__':
    print("--- Testing trainer.py (Illustrative: Full test requires main script context) ---")
    
    class MockDataLoader: 
        def __init__(self, num_batches=10): self.num_batches = num_batches
        def __len__(self): return self.num_batches
        def __iter__(self): 
            for i in range(self.num_batches):
                # Yielding dummy targets that match expected output of model for CrossEntropyLoss
                yield torch.randint(0, 256, (4, 16), dtype=torch.long), torch.randint(0, 256, (4,), dtype=torch.long)


    dummy_trainer_config_main = CONFIG_V3.copy() # To ensure it's defined for finally block

    try:
        print("Setting up minimal components for Trainer instantiation test...")
        device = torch.device("cpu") 
        
        dummy_model_config = CONFIG_V3.copy()
        dummy_model_config["embedding_dim"] = 16 
        dummy_model_config["attention_d_model"] = 16
        dummy_model_config["num_attention_layers"] = 1
        dummy_model_config["attention_num_heads"] = 1
        dummy_model_config["use_cnn_frontend"] = False
        dummy_model_config["sequence_length"] = 16 
        dummy_model_config["vocab_size"] = 256 # Ensure vocab_size is set


        dummy_model_instance = ByteLLM_GrugV3(dummy_model_config).to(device)
        dummy_optimizer = optim.Adam(dummy_model_instance.parameters(), lr=1e-3)
        dummy_criterion = torch.nn.CrossEntropyLoss()
        
        mock_train_dl = MockDataLoader(num_batches=20)
        mock_val_dl = MockDataLoader(num_batches=5) 

        dummy_trainer_config_main = CONFIG_V3.copy() 
        dummy_trainer_config_main["num_epochs"] = 1 
        dummy_trainer_config_main["print_every"] = 5
        dummy_trainer_config_main["use_amp"] = False 
        dummy_trainer_config_main["checkpoint_dir"] = "./temp_test_checkpoints_trainer"
        dummy_trainer_config_main["model_name"] = "test_grug_trainer_combined"
        dummy_trainer_config_main["use_lr_warmup"] = False 
        dummy_trainer_config_main["test_every_batches"] = 0 
        dummy_trainer_config_main["validate_every_batches"] = 3 
        # "checkpoint_every_batches" is no longer used for non-best batch saves.

        ensure_dir(dummy_trainer_config_main["checkpoint_dir"])

        trainer_instance = Trainer(
            model=dummy_model_instance,
            train_dataloader=mock_train_dl,
            val_dataloader=mock_val_dl, 
            optimizer=dummy_optimizer,
            criterion=dummy_criterion,
            device=device,
            checkpoint_dir=dummy_trainer_config_main["checkpoint_dir"],
            model_name=dummy_trainer_config_main["model_name"],
            scheduler=None,
            train_config=dummy_trainer_config_main
        )
        print(f"Trainer instance created. Initial best_val_loss: {trainer_instance.best_val_loss}")

        print("Illustrative: calling train_epoch to test combined validation and best loss update...")
        original_evaluate_epoch = trainer_instance.evaluate_epoch
        
        mock_losses = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3] 
        def mock_evaluate_epoch_fn(epoch_num, profiler_context=None):
            loss_to_return = mock_losses.pop(0) if mock_losses else 0.9 
            print(f"[Mocked evaluate_epoch] Returning loss: {loss_to_return}")
            # Call the original method's structure but return a mocked value
            trainer_instance.model.eval() # Ensure model is in eval mode for this mock
            # inputs, targets = next(iter(trainer_instance.val_dataloader)) # Simulate getting data
            # inputs, targets = inputs.to(device), targets.to(device)
            # with torch.no_grad():
            #     with autocast(device_type=device.type, enabled=trainer_instance.use_amp):
            #         outputs = trainer_instance.model(inputs)
            #         # loss = trainer_instance.criterion(outputs, targets) # Don't actually compute
            return loss_to_return 
        
        trainer_instance.evaluate_epoch = mock_evaluate_epoch_fn
        
        trainer_instance.train_epoch(epoch_num=0) 

        trainer_instance.evaluate_epoch = original_evaluate_epoch

        print(f"After train_epoch, best_val_loss: {trainer_instance.best_val_loss}")
        best_model_path = Path(dummy_trainer_config_main["checkpoint_dir"]) / f"{dummy_trainer_config_main['model_name']}_best.pth"
        if best_model_path.exists():
            print(f"Best model saved at: {best_model_path}")
            # Verify that no other batch-specific (non-best) checkpoints were made by the combined logic
            other_batch_checkpoints = list(Path(dummy_trainer_config_main["checkpoint_dir"]).glob(f"{dummy_trainer_config_main['model_name']}_epoch_*_batch_*.pth"))
            if not other_batch_checkpoints:
                print("Correctly, no non-best batch-specific checkpoints were found from the combined logic.")
            else:
                print(f"ERROR: Found unexpected non-best batch checkpoints: {other_batch_checkpoints}")

        else:
            print(f"Best model was NOT saved at: {best_model_path} (This might be okay if losses didn't improve as expected by mock).")


        print("\n--- Basic Trainer instantiation and illustrative train_epoch call finished ---")

    except Exception as e:
        print(f"Error during illustrative Trainer test: {e}")
        traceback.print_exc()
    finally:
        temp_ckpt_dir_main = Path(dummy_trainer_config_main.get("checkpoint_dir", "./temp_test_checkpoints_trainer"))
        if temp_ckpt_dir_main.exists():
            try:
                for item in temp_ckpt_dir_main.glob('*'): 
                    if item.is_file(): item.unlink()
                temp_ckpt_dir_main.rmdir()
                print(f"Cleaned up {temp_ckpt_dir_main}")
            except Exception as e_clean: print(f"Error cleaning up {temp_ckpt_dir_main}: {e_clean}")