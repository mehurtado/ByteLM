# trainer.py

import torch
import torch.optim as optim
import torch.profiler
from torch.amp import GradScaler, autocast
from pathlib import Path
import traceback # For detailed error logging in checkpoint loading

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

        Args:
            model (ByteLLM_GrugV3): The model to be trained.
            train_dataloader (DataLoader): DataLoader for the training set.
            val_dataloader (DataLoader | None): DataLoader for the validation set. Can be None.
            optimizer (Optimizer): The optimizer for training.
            criterion (Module): The loss function.
            device (device): The device to train on ('cuda' or 'cpu').
            checkpoint_dir (str or Path): Directory to save checkpoints.
            model_name (str): Name of the model, used for checkpoint filenames.
            scheduler (LRScheduler | None): Learning rate scheduler. Defaults to None.
            train_config (dict | None): Configuration dictionary for training parameters.
                                        Uses global CONFIG_V3 if None or for missing keys.
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
        
        # Use provided train_config, fall back to global CONFIG_V3 for defaults
        self.train_config = train_config if train_config is not None else CONFIG_V3.copy()

        ensure_dir(self.checkpoint_dir)
        if self.train_config.get("enable_profiler", False):
            profiler_log_dir = Path(self.train_config.get("profiler_log_dir", "./profiler_logs_grug_v3"))
            ensure_dir(profiler_log_dir)
            ensure_dir(profiler_log_dir / "train") # Subdirectory for training traces
            ensure_dir(profiler_log_dir / "eval")   # Subdirectory for evaluation traces


        self.current_global_step = 0 # Tracks total number of optimization steps
        
        # Automatic Mixed Precision (AMP) setup
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
        target_lr = self.train_config["learning_rate"] # Base LR after warmup
        init_factor = self.train_config.get("lr_warmup_init_factor", 0.01)

        if warmup_steps == 0: # Should not happen if use_lr_warmup is true and steps > 0
            lr_scale = 1.0
        elif self.current_global_step == 0: # First step
            lr_scale = init_factor
        else:
            # Linear warmup from init_factor * target_lr to target_lr
            lr_scale = init_factor + (1.0 - init_factor) * (self.current_global_step / warmup_steps)
        
        lr_scale = min(lr_scale, 1.0) # Cap at 1.0

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = target_lr * lr_scale
        
        # Log warmup LR changes periodically
        if self.current_global_step == 0 or \
           (self.current_global_step + 1) % max(1, warmup_steps // 10) == 0 or \
           self.current_global_step == warmup_steps - 1:
            print(f"Warmup Step {self.current_global_step + 1}/{warmup_steps}, Current LR: {self.optimizer.param_groups[0]['lr']:.2e}")
        
        if self.current_global_step == warmup_steps -1: # Last warmup step
             print(f"Warmup finished. LR will be managed by scheduler or remain at target: {target_lr:.2e}")


    def _run_training_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        """Performs a single training step (forward, backward, optimize)."""
        inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
        
        self.optimizer.zero_grad(set_to_none=True) # More memory efficient
        
        with autocast(device_type=self.device.type, enabled=self.use_amp):
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
        
        self.scaler.scale(loss).backward()
        
        # Gradient Clipping (unscale first if using AMP)
        clip_val = self.train_config.get("clip_grad_norm_value")
        if clip_val is not None and clip_val > 0:
            self.scaler.unscale_(self.optimizer) # Unscales gradients held by optimizer.params
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_val)
            
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()

    def run_interim_test(self, epoch_num: int, batch_idx: int):
        """Runs a quick generation test during training."""
        print(f"\n--- Interim Test @ Epoch {epoch_num+1}, Batch {batch_idx+1} (Global Step: {self.current_global_step}) ---")
        self.model.eval() # Switch model to evaluation mode

        # Use generation parameters from the main training config
        interim_gen_config = {
            "generation_temperature": self.train_config.get("interim_test_temperature", 1.0),
            "generation_top_k": self.train_config.get("interim_test_top_k", 0)
        }
        # Model internal config for predictor should reflect the currently trained model's settings
        model_cfg_for_pred = {
            "sequence_length": self.model.config.get("sequence_length", self.train_config.get("sequence_length")),
            "max_positional_encoding_len": self.model.config.get("max_positional_encoding_len", self.train_config.get("max_positional_encoding_len")),
        }
        
        interim_predictor = Predictor(self.model, self.device, interim_gen_config, model_cfg_for_pred)
        
        seed_text = "The meaning of life is " # A common seed phrase
        seed_bytes = seed_text.encode('utf-8')
        print(f"Seed: '{seed_text}'")
        
        generated_bytes = interim_predictor.generate_sequence(seed_bytes, length=128) # Generate 128 bytes
        
        try:
            generated_text = generated_bytes.decode('utf-8', errors='replace')
            print(f"Generated (128 bytes): \"{generated_text}\"")
        except Exception as e:
            print(f"Error decoding generated bytes for interim test: {e}")
            print(f"Raw generated bytes: {generated_bytes}")
            
        self.model.train() # Switch model back to training mode
        print(f"--- End Interim Test ---\n")

    def train_epoch(self, epoch_num: int, profiler_context: torch.profiler.profile | None = None):
        """Trains the model for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(self.train_dataloader)

        if num_batches == 0:
            print(f"Epoch {epoch_num+1}: Training dataloader is empty. Skipping training for this epoch.")
            return float('inf') # Return inf loss if no training happened

        for batch_idx, (inputs, targets) in enumerate(self.train_dataloader):
            self._perform_lr_warmup() # Apply LR warmup if active
            
            loss_item = self._run_training_step(inputs, targets)
            epoch_loss += loss_item
            
            if profiler_context:
                profiler_context.step() # Step profiler after optimizer step

            self.current_global_step += 1 # Increment global step after each optimization

            if (batch_idx + 1) % self.train_config.get("print_every", 100) == 0 or (batch_idx + 1) == num_batches:
                current_lr = self.optimizer.param_groups[0]['lr'] if self.optimizer.param_groups else 0.0
                print(f"Epoch {epoch_num+1}/{self.train_config['num_epochs']}, "
                      f"Batch {batch_idx+1}/{num_batches}, "
                      f"Train Loss: {loss_item:.4f}, LR: {current_lr:.2e}, "
                      f"Global Step: {self.current_global_step}")

            # Interim testing based on global steps
            test_interval_batches = self.train_config.get("test_every_batches", 0)
            if test_interval_batches > 0 and (self.current_global_step % test_interval_batches == 0) and self.current_global_step > 0:
                self.run_interim_test(epoch_num, batch_idx)
                self.model.train() # Ensure model is back in train mode after interim test

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            
        return epoch_loss / num_batches if num_batches > 0 else float('inf')

    def evaluate_epoch(self, epoch_num: int, profiler_context: torch.profiler.profile | None = None):
        """Evaluates the model on the validation set for one epoch."""
        self.model.eval()
        val_loss = 0.0

        if not self.val_dataloader:
            print(f"Epoch {epoch_num+1}: Validation dataloader is not available. Skipping validation.")
            # Step scheduler if it's not ReduceLROnPlateau and not in warmup
            is_after_warmup = not self.train_config.get("use_lr_warmup", False) or \
                              self.current_global_step >= self.train_config.get("lr_warmup_steps", 0)
            if self.scheduler and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau) and is_after_warmup:
                self.scheduler.step()
            return float('inf') # Return inf loss if no validation

        num_val_batches = len(self.val_dataloader)
        if num_val_batches == 0:
            print(f"Epoch {epoch_num+1}: Validation dataloader is empty. Skipping validation.")
            is_after_warmup = not self.train_config.get("use_lr_warmup", False) or \
                              self.current_global_step >= self.train_config.get("lr_warmup_steps", 0)
            if self.scheduler and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau) and is_after_warmup:
                self.scheduler.step()
            return float('inf')

        active_profiler_steps = 0
        if profiler_context:
            # For eval, profile a few batches if num_val_batches is large, or all if small
            active_profiler_steps = min(self.train_config.get("profiler_schedule_active", 5), num_val_batches)


        with torch.no_grad():
            for batch_idx_eval, (inputs, targets) in enumerate(self.val_dataloader):
                inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                with autocast(device_type=self.device.type, enabled=self.use_amp): # AMP for evaluation too
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                val_loss += loss.item()
                
                if profiler_context and batch_idx_eval < active_profiler_steps:
                     profiler_context.step() # Step profiler for eval batches

        avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else float('inf')
        print(f"Epoch {epoch_num+1}/{self.train_config['num_epochs']}, Validation Loss: {avg_val_loss:.4f}")

        # Learning rate scheduler step (after warmup phase)
        is_after_warmup = not self.train_config.get("use_lr_warmup",False) or \
                          self.current_global_step >= self.train_config.get("lr_warmup_steps",0)
        
        if self.scheduler and is_after_warmup:
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(avg_val_loss)
            else:
                self.scheduler.step()
        
        self.model.train() # Set model back to training mode
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        return avg_val_loss

    def train(self, num_epochs: int):
        """Main training loop."""
        print(f"Starting GrugV3 training with model {self.model_name} on device {self.device}...")
        self.model.to(self.device) # Ensure model is on the correct device
        
        start_epoch = 0
        best_val_loss = float('inf')

        # Resume from checkpoint if specified
        resume_path_str = self.train_config.get("resume_from_checkpoint")
        if resume_path_str:
            loaded_info = self.load_checkpoint(resume_path_str)
            if loaded_info:
                start_epoch = loaded_info.get('epoch', -1) + 1
                self.current_global_step = loaded_info.get('current_global_step', 0)
                print(f"Resuming GrugV3 training from epoch {start_epoch}. Global step set to {self.current_global_step}.")
                
                if self.train_config.get("reset_best_val_loss_on_resume", False):
                    best_val_loss = float('inf')
                    print("Best validation loss reset on resume.")
                elif loaded_info.get('loss') is not None:
                    best_val_loss = loaded_info['loss']
                
                # Load GradScaler state if AMP was used and state exists
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
            self.current_global_step = 0 # Ensure global step is 0 if not resuming

        for epoch in range(start_epoch, num_epochs):
            print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
            
            # Profiler setup for this epoch if enabled and matches target epoch
            train_prof = None
            eval_prof = None
            profiler_active_this_epoch = (self.train_config.get("enable_profiler", False) and
                                          epoch == self.train_config.get("profile_epoch_target", 0))

            if profiler_active_this_epoch:
                prof_log_dir = Path(self.train_config.get("profiler_log_dir", "./profiler_logs_grug_v3"))
                p_wait = self.train_config.get("profiler_schedule_wait", 1)
                p_warmup = self.train_config.get("profiler_schedule_warmup", 1)
                p_active = self.train_config.get("profiler_schedule_active", 3) # Profile a few steps
                p_repeat = self.train_config.get("profiler_schedule_repeat", 1)
                
                train_schedule = torch.profiler.schedule(wait=p_wait, warmup=p_warmup, active=p_active, repeat=p_repeat)
                train_prof = torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA] if self.device.type == 'cuda' else [torch.profiler.ProfilerActivity.CPU],
                    schedule=train_schedule,
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(str(prof_log_dir / "train")),
                    record_shapes=True, profile_memory=True, with_stack=True
                )
                
                # For eval, profile a few steps as well, simpler schedule
                eval_active_steps = min(p_active, len(self.val_dataloader) if self.val_dataloader else 1)
                eval_schedule = torch.profiler.schedule(wait=0, warmup=0, active=eval_active_steps, repeat=1)
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

            if eval_prof: eval_prof.start()
            current_val_loss = self.evaluate_epoch(epoch, profiler_context=eval_prof)
            if eval_prof: 
                eval_prof.stop()
                print(f"Evaluation Profiler traces for Epoch {epoch+1} saved to: {prof_log_dir / 'eval'}")

            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                print(f"New best GrugV3 validation loss: {best_val_loss:.4f}. Saving best model...")
                self.save_checkpoint(epoch, best_val_loss, is_best=True)
            
            # Optionally save checkpoint after every epoch (or N epochs)
            # For now, just saving the best one and the latest one (implicitly by custom_filename)
            epoch_checkpoint_filename = f"{self.model_name}_epoch_{epoch+1}.pth"
            self.save_checkpoint(epoch, current_val_loss, is_best=False, custom_filename=epoch_checkpoint_filename)

        print("\nGrugV3 Training finished.")

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False, custom_filename: str = None):
        """Saves a model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': val_loss, # This is typically the validation loss
            'config': self.model.config, # Save the model's internal config
            'train_config': self.train_config, # Save the training config used
            'current_global_step': self.current_global_step
        }
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        if self.use_amp: # Save GradScaler state if AMP is used
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        if is_best:
            filename = f"{self.model_name}_best.pth"
        elif custom_filename:
            filename = custom_filename
        else: # Fallback generic name if no custom name and not best
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
            # Load checkpoint to the same device type the model is on, to avoid issues
            # If model is on CPU, load to CPU. If on CUDA, load to CUDA.
            checkpoint = torch.load(load_path, map_location=self.device)

            # Integrity checks for loaded config (optional but good practice)
            loaded_model_cfg = checkpoint.get('config')
            if not loaded_model_cfg:
                print("Warning: Checkpoint does not contain model 'config'. Model architecture might be incompatible.")
            # Compare key parameters if needed, e.g., vocab_size, embedding_dim
            # For example:
            # if loaded_model_cfg and loaded_model_cfg.get('vocab_size') != self.model.config.get('vocab_size'):
            #     print("CRITICAL WARNING: Vocab size mismatch between current model and checkpoint!")
            
            loaded_train_cfg = checkpoint.get('train_config')
            if loaded_train_cfg and self.train_config.get("use_amp") != loaded_train_cfg.get("use_amp"):
                print("Warning: AMP setting 'use_amp' mismatch between current training config and checkpoint's training config.")

            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            if 'optimizer_state_dict' in checkpoint and self.optimizer:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print("Optimizer state loaded.")
                except ValueError as e: # Can happen if model parameters changed
                    print(f"Warning: Could not load optimizer state, possibly due to model structure changes: {e}")
            
            if 'scheduler_state_dict' in checkpoint and self.scheduler:
                try:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    print("Scheduler state loaded.")
                except Exception as e: # Schedulers can also have issues if optimizer changed
                    print(f"Warning: Could not load scheduler state: {e}")
            
            print(f"GrugV3 Checkpoint loaded successfully from {load_path}.")
            return {
                'epoch': checkpoint.get('epoch', -1), # Default to -1 if not found
                'loss': checkpoint.get('loss', float('inf')),
                'config': loaded_model_cfg,
                'train_config': loaded_train_cfg,
                'current_global_step': checkpoint.get('current_global_step', 0),
                'scaler_state_dict': checkpoint.get('scaler_state_dict') # Pass along for AMP
            }
            
        except Exception as e:
            print(f"Error loading GrugV3 checkpoint {load_path}: {e}")
            traceback.print_exc()
            return None

if __name__ == '__main__':
    # This block is for illustrative purposes. Training is complex to set up for a simple module test.
    # A full test would require dummy data, dataloaders, model, optimizer, etc.
    print("--- Testing trainer.py (Illustrative: Full test requires main script context) ---")

    # Basic check: Can we instantiate Trainer?
    # Needs a dummy model, optimizer, criterion, device, dataloaders (can be mock/None for simple init test)
    
    class MockDataLoader: # Minimal mock
        def __init__(self, num_batches=10): self.num_batches = num_batches
        def __len__(self): return self.num_batches
        def __iter__(self): 
            for i in range(self.num_batches):
                # Yield dummy data matching expected structure (input, target)
                # Shapes depend on config (batch_size, sequence_length)
                # For ByteLLM, input is (B,S), target is (B)
                # Using placeholder shapes for now.
                yield torch.randint(0, 256, (4, 16)), torch.randint(0, 256, (4,)) 

    try:
        print("Setting up minimal components for Trainer instantiation test...")
        device = torch.device("cpu") # Use CPU for simple test
        
        # Use a minimal config for the dummy model
        dummy_model_config = CONFIG_V3.copy()
        dummy_model_config["embedding_dim"] = 16
        dummy_model_config["attention_d_model"] = 16
        dummy_model_config["num_attention_layers"] = 1
        dummy_model_config["attention_num_heads"] = 1
        dummy_model_config["use_cnn_frontend"] = False
        dummy_model_config["sequence_length"] = 16 # For MockDataLoader consistency


        dummy_model_instance = ByteLLM_GrugV3(dummy_model_config).to(device)
        dummy_optimizer = optim.Adam(dummy_model_instance.parameters(), lr=1e-3)
        dummy_criterion = torch.nn.CrossEntropyLoss()
        
        # Mock dataloaders
        mock_train_dl = MockDataLoader(num_batches=20)
        mock_val_dl = MockDataLoader(num_batches=5)

        # Minimal training config for the trainer itself
        dummy_trainer_config = CONFIG_V3.copy() # Start with global defaults
        dummy_trainer_config["num_epochs"] = 1 # Just for init
        dummy_trainer_config["print_every"] = 5
        dummy_trainer_config["use_amp"] = False # Simpler for CPU test
        dummy_trainer_config["checkpoint_dir"] = "./temp_test_checkpoints_trainer"
        dummy_trainer_config["model_name"] = "test_grug_trainer"
        dummy_trainer_config["use_lr_warmup"] = False # Disable for simple test
        dummy_trainer_config["test_every_batches"] = 0 # Disable interim for simple test

        ensure_dir(dummy_trainer_config["checkpoint_dir"])

        trainer_instance = Trainer(
            model=dummy_model_instance,
            train_dataloader=mock_train_dl,
            val_dataloader=mock_val_dl,
            optimizer=dummy_optimizer,
            criterion=dummy_criterion,
            device=device,
            checkpoint_dir=dummy_trainer_config["checkpoint_dir"],
            model_name=dummy_trainer_config["model_name"],
            scheduler=None,
            train_config=dummy_trainer_config
        )
        print("Trainer instance created successfully.")

        # Illustrative: call a method like _perform_lr_warmup (if enabled) or save_checkpoint
        print("Illustrative: calling save_checkpoint...")
        trainer_instance.save_checkpoint(epoch=0, val_loss=0.5, is_best=True)
        
        # Illustrative: try loading it back
        print("Illustrative: calling load_checkpoint...")
        best_ckpt_path = Path(dummy_trainer_config["checkpoint_dir"]) / f"{dummy_trainer_config['model_name']}_best.pth"
        if best_ckpt_path.exists():
            loaded = trainer_instance.load_checkpoint(str(best_ckpt_path))
            if loaded:
                print(f"Checkpoint loaded, epoch: {loaded.get('epoch')}, loss: {loaded.get('loss')}")
            else:
                print("Failed to load checkpoint for test.")
        else:
            print(f"Checkpoint {best_ckpt_path} not found for loading test.")


        print("\n--- Basic Trainer instantiation and checkpoint save/load test finished ---")

    except Exception as e:
        print(f"Error during illustrative Trainer test: {e}")
        traceback.print_exc()
    finally:
        # Clean up dummy checkpoint dir
        temp_ckpt_dir = Path(dummy_trainer_config.get("checkpoint_dir", "./temp_test_checkpoints_trainer"))
        if temp_ckpt_dir.exists():
            try:
                for item in temp_ckpt_dir.glob('*'): item.unlink()
                temp_ckpt_dir.rmdir()
                print(f"Cleaned up {temp_ckpt_dir}")
            except Exception as e_clean: print(f"Error cleaning up {temp_ckpt_dir}: {e_clean}")
