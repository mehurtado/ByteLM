import torch
import torch.nn as nn
import torch.nn.functional as F



import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F # Not strictly for data, but often included
import torch.optim as optim # Not strictly for data
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from pathlib import Path
import numpy as np
import traceback # For error handling

# Mamba import is not strictly needed for GrugV3 but keeping for minimum diff if other parts of grug_v2 are copied later
# It's guarded by try-except, so it won't break if mamba_ssm is not installed in this environment.
try:
    from mamba_ssm import Mamba # Not used by GrugV3 model itself
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    class Mamba: 
        def __init__(self, *args, **kwargs): pass
        def forward(self, *args, **kwargs): pass



# --- Utility Functions ---
from pathlib import Path # ensure_dir needs this
import numpy as np # generate_dummy_data might use if extended

def ensure_dir(directory_path):
    Path(directory_path).mkdir(parents=True, exist_ok=True)

def generate_dummy_data(data_dir, config_dict, num_files=5, lines_per_file=10000): # Added config_dict
    ensure_dir(data_dir)
    if not config_dict.get("generate_dummy_data_if_empty", False):
        # print(f"Dummy data generation is disabled in config for {data_dir}.")
        return
        
    # Check if .txt files already exist
    if not list(Path(data_dir).glob('*.txt')):
        print(f"Generating dummy data in {data_dir}...")
        for i in range(num_files):
            with open(Path(data_dir) / f"dummy_data_{i}.txt", "w", encoding="utf-8") as f:
                for j in range(lines_per_file):
                    # Corrected escaped newline
                    f.write(f"This is line {j+1} of GrugV3 dummy file {i+1}. The quick brown fox jumps over the lazy dog. 0123456789. áéíóúñü. " * 5 + "\n")
        print("Dummy data generated.")
    else:
        print(f"Directory {data_dir} already contains .txt files. Skipping dummy data generation.")



# --- Custom Dataset ---
from torch.utils.data import Dataset # Already in imports, but good for clarity
import torch # For torch.tensor

class ByteSequenceDataset(Dataset):
    def __init__(self, all_bytes_mmap_array, sequence_length):
        self.all_bytes = all_bytes_mmap_array
        self.sequence_length = sequence_length
        
        if len(self.all_bytes) <= self.sequence_length:
            raise ValueError(
                f"Total data length ({len(self.all_bytes)} bytes) is less than or equal to sequence_length "
                f"({self.sequence_length}). Not enough data to create sequences."
            )
        self.num_sequences = len(self.all_bytes) - self.sequence_length

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        if not (0 <= idx < self.num_sequences):
            raise IndexError(f"Index {idx} out of range for dataset of length {self.num_sequences}")
        
        input_sequence_np = self.all_bytes[idx : idx + self.sequence_length].copy()
        target_np = self.all_bytes[idx + self.sequence_length].copy()
        
        input_tensor = torch.tensor(input_sequence_np, dtype=torch.long)
        target_tensor = torch.tensor(target_np, dtype=torch.long)
            
        return input_tensor, target_tensor



# --- Data Processor ---
from pathlib import Path # Already in imports
import numpy as np # Already in imports
import glob # For glob.glob
from torch.utils.data import DataLoader, random_split, Subset # Already in imports

class DataProcessor:
    def __init__(self, data_dir, processed_data_dir, sequence_length, force_reprocess=False, config_for_data_gen=None): # Added config_for_data_gen
        self.data_dir = Path(data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.sequence_length = sequence_length
        self.force_reprocess = force_reprocess
        self.config_for_data_gen = config_for_data_gen # Store the config
        ensure_dir(self.processed_data_dir)
        # Modified for GrugV3
        self.all_bytes_path = Path(self.processed_data_dir) / "all_bytes_grug_v3.npy"


    def load_or_create_all_bytes(self):
        if not self.force_reprocess and self.all_bytes_path.exists():
            print(f"Loading cached {self.all_bytes_path.name} using memory-mapping...")
            try:
                all_bytes_mmap = np.load(self.all_bytes_path, mmap_mode='r')
                print(f"Successfully memory-mapped (Length: {len(all_bytes_mmap):,} bytes).")
                return all_bytes_mmap
            except Exception as e:
                print(f"Error memory-mapping {self.all_bytes_path}: {e}. Reprocessing...")

        print(f"Processing text files from {self.data_dir} to create {self.all_bytes_path.name}...")
        text_files = glob.glob(str(self.data_dir / "*.txt")) 
        if not text_files:
            print(f"No .txt files found in {self.data_dir}.")
            # Use self.config_for_data_gen here
            if self.config_for_data_gen and self.config_for_data_gen.get("generate_dummy_data_if_empty", False):
                generate_dummy_data(str(self.data_dir), self.config_for_data_gen) 
                text_files = glob.glob(str(self.data_dir / "*.txt"))
                if not text_files:
                    raise FileNotFoundError(f"Still no .txt files found in {self.data_dir} after dummy data generation attempt.")
            else:
                raise FileNotFoundError(f"No .txt files found in {self.data_dir} and dummy data generation is off or config not provided.")

        full_text_content = []
        print(f"Found {len(text_files)} .txt files. Reading content...")
        for file_path in text_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    full_text_content.append(f.read())
            except Exception as e:
                print(f"Warning: Could not read file {file_path} as text: {e}")
                continue
        
        if not full_text_content:
            raise ValueError("No text data could be read from the files.")

        final_text_string = "".join(full_text_content)
        print(f"Total characters read: {len(final_text_string):,}")
        encoded_bytes = final_text_string.encode('utf-8', errors='replace')
        all_bytes_np_array = np.array(list(encoded_bytes), dtype=np.uint8)

        if len(all_bytes_np_array) == 0:
            raise ValueError("Processed data resulted in an empty byte array after encoding.")

        np.save(self.all_bytes_path, all_bytes_np_array)
        print(f"Saved {self.all_bytes_path.name} (Length: {len(all_bytes_np_array):,} bytes). Now loading with memory-mapping...")
        all_bytes_mmap = np.load(self.all_bytes_path, mmap_mode='r')
        return all_bytes_mmap

    def get_dataloaders(self, batch_size, val_split_ratio, num_workers, current_sequence_length):
        all_bytes_mmap = self.load_or_create_all_bytes()
        full_dataset = ByteSequenceDataset(all_bytes_mmap, current_sequence_length)
        
        num_total_sequences = len(full_dataset)
        if num_total_sequences == 0:
            raise ValueError("The full dataset resulted in 0 sequences. Cannot create dataloaders.")

        if not (0 < val_split_ratio < 1):
            print(f"Warning: val_split_ratio ({val_split_ratio}) is not between 0 and 1. Using all data for training.")
            train_indices = np.arange(num_total_sequences)
            val_indices = np.array([])
        else:
            print(f"Generating and shuffling indices for {num_total_sequences} sequences...")
            indices = np.arange(num_total_sequences)
            np.random.shuffle(indices)
            print("Indices shuffled.")

            num_val_sequences = int(val_split_ratio * num_total_sequences)
            num_train_sequences = num_total_sequences - num_val_sequences

            if num_val_sequences == 0 or num_train_sequences == 0:
                print(f"Warning: Dataset size ({num_total_sequences}) too small for val_split_ratio ({val_split_ratio}). Adjusting split.")
                if num_train_sequences < batch_size : # Check if train set is too small for one batch
                     print(f"Warning: num_train_sequences ({num_train_sequences}) is smaller than batch_size ({batch_size}). This might lead to issues.")
                train_indices = indices 
                val_indices = np.array([]) # No validation if split is problematic
            else:
                train_indices = indices[:num_train_sequences]
                val_indices = indices[num_train_sequences:]
        
        train_dataset = Subset(full_dataset, train_indices.tolist())
        val_dataset = Subset(full_dataset, val_indices.tolist())
        print(f"Training set size: {len(train_dataset)} sequences")
        print(f"Validation set size: {len(val_dataset)} sequences")

        pin_memory_flag = (num_workers > 0 and torch.cuda.is_available())
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                      num_workers=num_workers, pin_memory=pin_memory_flag, drop_last=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                                    num_workers=num_workers, pin_memory=pin_memory_flag, drop_last=False) if len(val_dataset) > 0 else None
        
        if len(train_dataloader) == 0 and len(train_dataset) > 0 :
            print(f"Warning: Training DataLoader is empty. Batch size ({batch_size}) might be too large for training set ({len(train_dataset)}).")
        if val_dataloader is None and len(val_dataset) > 0: # Corrected logic for print
             print(f"Info: Validation DataLoader not created as validation set is empty or too small.")
        elif val_dataloader and len(val_dataloader) == 0 and len(val_dataset) > 0: # Corrected logic for print
            print(f"Warning: Validation DataLoader is empty. Batch size ({batch_size}) might be too large for validation set ({len(val_dataset)}).")
        return train_dataloader, val_dataloader

    def get_vocab_size(self):
        return 256 # For byte-level models


# --- Configuration for GrugV3 ---
CONFIG_V3 = {
    # Data and General (Copied from GrugV2, modify as needed for GrugV3)
    "data_dir": "./dataset/USE_V3", # Potentially new data directory
    "processed_data_dir": "./dataset/USE_V3_processed",
    "checkpoint_dir": "./checkpoints_grug_v3",
    "model_name": "grug_v3_cnn_attention",
    "resume_from_checkpoint": None, # Example: "./checkpoints_grug_v3/grug_v3_cnn_attention_best.pth"
    "sequence_length": 16,
    "batch_size": 32,
    "vocab_size": 256, # Should be consistent with data
    "val_split_ratio": 0.1,
    "num_workers": 8,
    "generate_dummy_data_if_empty": True,
    "force_reprocess_data": False,

    # Embedding (Copied from GrugV2)
    "embedding_dim": 512,

    # CNN Frontend (Copied from GrugV2)
    "use_cnn_frontend": True,
    "cnn_out_channels_list": [512, 512], # Output of last CNN layer
    "cnn_kernel_sizes": [9, 3],
    "cnn_stride": 1,
    "cnn_padding_mode": "zeros",
    "cnn_activation": "GELU",
    "cnn_dropout": 0.2,
    "cnn_use_layernorm": True,

    # Learnable Positional Encoding (Copied from GrugV2)
    "max_positional_encoding_len": 512,
    "pe_dropout": 0.3,

    # Multihead Attention Parameters
    # 'attention_d_model' will be dynamically set in the model __init__ based on CNN/Embedding output.
    # It's included here for completeness but will be overridden.
    # If cnn_out_channels_list is [512, 512], then attention_d_model will be 512.
    # If no CNN, it will be embedding_dim (512).
    "attention_d_model": 512, # Placeholder, will be determined by preceding layer's output dim.
    "attention_num_heads": 8,       # e.g., 8. Must be a divisor of attention_d_model.
    "attention_dropout": 0.1,     # Dropout for attention layers
    "num_attention_layers": 4,      # Number of stacked attention layers (like num_mamba_layers)

    # Output Layer (Copied from GrugV2)
    "output_dropout": 0.2,

    # Training (Copied from GrugV2, adjust as needed)
    "num_epochs": 50,
    "learning_rate": 3e-5,
    "optimizer_type": "AdamW",
    "adam_beta1": 0.9,
    "adam_beta2": 0.98,
    "adam_eps": 1e-9,
    "weight_decay": 0.01,
    "scheduler_type": "CosineAnnealingLR",
    "lr_scheduler_T_max": 50 * 1000, # Placeholder, adjust based on actual batches
    "lr_scheduler_eta_min": 1e-6,
    "lr_scheduler_patience": 10, # For ReduceLROnPlateau
    "lr_scheduler_factor": 0.1,  # For ReduceLROnPlateau
    "clip_grad_norm_value": 1.0,
    "print_every": 100,
    "test_every_batches": 500,
    "reset_best_val_loss_on_resume": True,

    # LR Warmup (Copied from GrugV2)
    "use_lr_warmup": True,
    "lr_warmup_steps": 2000,
    "lr_warmup_init_factor": 0.01,

    # Automatic Mixed Precision (AMP) (Copied from GrugV2)
    "use_amp": False, # Set to True to enable AMP

    # Generation / Prediction (Copied from GrugV2)
    "generation_temperature": 1.0,
    "generation_top_k": 50,
    "interim_test_temperature": 0.6,
    "interim_test_top_k": 20,

    # Profiling (Copied from GrugV2)
    "enable_profiler": False,
    "profiler_log_dir": "./profiler_logs_grug_v3",
    "profile_epoch_target": 0,
    "profiler_schedule_wait": 5,
    "profiler_schedule_warmup": 5,
    "profiler_schedule_active": 10,
    "profiler_schedule_repeat": 1,

    # Main script flow control (Copied from GrugV2)
    "DO_TRAINING": True,
    "DO_PREDICTION": True,

    # CuDNN Benchmarking (Copied from GrugV2)
    "cudnn_benchmark": True
}

# --- Imports needed for Trainer/Predictor ---
import torch
import torch.optim as optim # For optim.lr_scheduler
from torch.cuda.amp import GradScaler, autocast # For AMP in Trainer
from pathlib import Path # For checkpoint paths
import traceback # For error printing
import time # For profiler
import torch.profiler # For profiler
# from mamba_ssm.utils.generation import InferenceParams # Not needed for GrugV3 Predictor

# --- Predictor (Adapted for GrugV3 - standard autoregressive generation) ---
class Predictor:
    def __init__(self, model, device, generation_config, model_internal_config):
        self.model = model.to(device).eval()
        self.device = device
        self.temperature = generation_config.get("generation_temperature", 1.0)
        self.top_k = generation_config.get("generation_top_k", 0)
        self.model_internal_config = model_internal_config
        if self.temperature <= 0: raise ValueError("Temperature must be positive.")
        print(f"Predictor initialized for GrugV3: Temp={self.temperature}, TopK={self.top_k}")
        print(f"Predictor using model's max_pos_len: {self.model_internal_config.get('max_positional_encoding_len', 'N/A')}")
        print(f"Predictor using model's training sequence_length for context: {self.model_internal_config.get('sequence_length', 'N/A')}")

    @torch.no_grad()
    def generate_sequence(self, seed_bytes, length=100):
        self.model.eval()
        if isinstance(seed_bytes, bytes):
            current_sequence_values = list(seed_bytes)
        elif isinstance(seed_bytes, list) and all(isinstance(x, int) for x in seed_bytes):
            current_sequence_values = list(seed_bytes)
        else:
            raise ValueError("seed_bytes must be bytes or list of ints.")
        generated_values = list(current_sequence_values)
        max_len_for_model_input = self.model_internal_config.get('max_positional_encoding_len', 512)
        context_len = self.model_internal_config.get('sequence_length', 16)
        for _ in range(length):
            start_idx = max(0, len(generated_values) - context_len)
            input_sequence = generated_values[start_idx:]
            if len(input_sequence) > max_len_for_model_input:
                input_sequence = input_sequence[-max_len_for_model_input:]
            if not input_sequence:
                if not generated_values:
                    input_tensor = torch.tensor([[0]], dtype=torch.long).to(self.device)
                else:
                    break
            else:
                input_tensor = torch.tensor([input_sequence], dtype=torch.long).to(self.device)
            logits = self.model(input_tensor) 
            logits_scaled = logits / self.temperature
            if self.top_k > 0:
                k = min(max(1, self.top_k), logits_scaled.size(-1))
                top_k_vals, top_k_indices = torch.topk(logits_scaled, k, dim=-1)
                filtered_logits = torch.full_like(logits_scaled, -float('Inf'))
                filtered_logits.scatter_(-1, top_k_indices, top_k_vals)
            else:
                filtered_logits = logits_scaled
            probabilities = torch.softmax(filtered_logits, dim=-1)
            if torch.isnan(probabilities).any() or probabilities.sum() < 1e-6:
                print("Warning: Invalid probabilities. Using argmax.")
                next_byte_val = torch.argmax(logits_scaled, dim=-1).item()
            else:
                next_byte_val = torch.multinomial(probabilities, 1).item()
            generated_values.append(next_byte_val)
        return bytes(generated_values)

# --- Trainer (Adapted for GrugV3) ---
class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, criterion, device, 
                 checkpoint_dir, model_name, scheduler=None, train_config=None):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_name = model_name
        self.scheduler = scheduler
        self.train_config = train_config if train_config else CONFIG_V3
        ensure_dir(self.checkpoint_dir)
        if self.train_config.get("enable_profiler"):
            ensure_dir(self.train_config.get("profiler_log_dir", "./profiler_logs_grug_v3"))
        self.current_config_for_checkpoint = self.train_config
        self.current_global_step = 0
        self.use_amp = self.train_config.get("use_amp", False) and self.device.type == 'cuda'
        self.scaler = GradScaler(enabled=self.use_amp)
        if self.use_amp: print("Automatic Mixed Precision (AMP) is ENABLED for training.")
        else: print("Automatic Mixed Precision (AMP) is DISABLED for training.")

    def _run_profiler_step(self, profiler_context, epoch_num, batch_idx, inputs, targets):
        inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=self.use_amp):
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
        self.scaler.scale(loss).backward()
        clip_val = self.train_config.get("clip_grad_norm_value")
        if clip_val is not None and clip_val > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_val)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if profiler_context: profiler_context.step()
        return loss.item()

    def _perform_lr_warmup(self):
        if self.train_config.get("use_lr_warmup", False) and \
           self.current_global_step < self.train_config.get("lr_warmup_steps", 0):
            warmup_steps = self.train_config["lr_warmup_steps"]
            target_lr = self.train_config["learning_rate"]
            init_factor = self.train_config.get("lr_warmup_init_factor", 0.01)
            if warmup_steps == 0: lr_scale = 1.0
            elif self.current_global_step == 0: lr_scale = init_factor
            else: lr_scale = init_factor + (1.0 - init_factor) * (self.current_global_step / warmup_steps)
            lr_scale = min(lr_scale, 1.0) 
            for param_group in self.optimizer.param_groups: param_group['lr'] = target_lr * lr_scale
            if self.current_global_step == 0 or (self.current_global_step + 1) % (warmup_steps // 10 if warmup_steps >=10 else 1) == 0 or self.current_global_step == warmup_steps -1 :
                 print(f"Warmup Step {self.current_global_step+1}/{warmup_steps}, Current LR: {self.optimizer.param_groups[0]['lr']:.2e}")
        elif self.train_config.get("use_lr_warmup", False) and \
             self.current_global_step == self.train_config.get("lr_warmup_steps", 0):
            target_lr = self.train_config["learning_rate"]
            for param_group in self.optimizer.param_groups: param_group['lr'] = target_lr
            print(f"Warmup finished. LR set to target: {target_lr:.2e}")

    def run_interim_test(self, epoch_num, batch_idx):
        print(f"\n--- Interim Test @ Epoch {epoch_num+1}, Batch {batch_idx+1} ---")
        self.model.eval()
        interim_gen_config = {
            "generation_temperature": self.train_config.get("interim_test_temperature", 1.0),
            "generation_top_k": self.train_config.get("interim_test_top_k", 0)
        }
        model_cfg_for_pred = {
            "max_positional_encoding_len": self.train_config["max_positional_encoding_len"],
            "sequence_length": self.train_config["sequence_length"],
        }
        interim_predictor = Predictor(self.model, self.device, interim_gen_config, model_cfg_for_pred)
        seed_text = "The meaning of life is "
        seed_bytes = seed_text.encode('utf-8')
        print(f"Seed: '{seed_text}'")
        generated_bytes = interim_predictor.generate_sequence(seed_bytes, length=128)
        try:
            generated_text = generated_bytes.decode('utf-8', errors='replace')
            print(f"Generated (128 bytes): \"{generated_text}\"")
        except Exception as e:
            print(f"Error decoding generated bytes for interim test: {e}")
            print(f"Raw generated bytes: {generated_bytes}")
        self.model.train()
        print(f"--- End Interim Test ---\n")

    def train_epoch(self, epoch_num):
        self.model.train()
        epoch_loss = 0
        num_batches = len(self.train_dataloader)
        if num_batches == 0:
            print(f"Epoch {epoch_num+1}: Training dataloader is empty. Skipping.")
            return float('inf')
        profiler_active_this_epoch = (self.train_config.get("enable_profiler", False) and 
                                      epoch_num == self.train_config.get("profile_epoch_target", 0))
        prof_context = None
        if profiler_active_this_epoch:
            print(f"--- Profiler activated for Training, Epoch {epoch_num+1} ---")
            p_wait = self.train_config.get("profiler_schedule_wait", 5)
            p_warmup = self.train_config.get("profiler_schedule_warmup", 5)
            p_active = self.train_config.get("profiler_schedule_active", 10)
            p_repeat = self.train_config.get("profiler_schedule_repeat", 1)
            prof_schedule = torch.profiler.schedule(wait=p_wait, warmup=p_warmup, active=p_active, repeat=p_repeat)
            prof_log_dir = self.train_config.get("profiler_log_dir", "./profiler_logs_grug_v3")
            ensure_dir(Path(prof_log_dir) / "train")
            prof_context = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=prof_schedule,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(Path(prof_log_dir) / "train"),
                record_shapes=True, profile_memory=True, with_stack=True )
            prof_context.start()
        for batch_idx, (inputs, targets) in enumerate(self.train_dataloader):
            self._perform_lr_warmup()
            current_loss = self._run_profiler_step(profiler_context, epoch_num, batch_idx, inputs, targets)
            epoch_loss += current_loss
            self.current_global_step += 1
            if (batch_idx + 1) % self.train_config["print_every"] == 0 or (batch_idx + 1) == num_batches:
                print(f"Epoch {epoch_num+1}/{self.train_config['num_epochs']}, Batch {batch_idx+1}/{num_batches}, Train Loss: {current_loss:.4f}, Current LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            test_interval = self.train_config.get("test_every_batches", 0)
            if test_interval > 0 and (self.current_global_step % test_interval == 0) and self.current_global_step > 0:
                self.run_interim_test(epoch_num, batch_idx)
        if prof_context:
            prof_context.stop()
            print(f"--- Profiler stopped for Training, Epoch {epoch_num+1} ---")
            print(f"Training Profiler traces saved to: {Path(prof_log_dir) / 'train'}")
        if self.device.type == 'cuda': torch.cuda.empty_cache()
        return epoch_loss / num_batches if num_batches > 0 else float('inf')

    def evaluate_epoch(self, epoch_num):
        self.model.eval()
        val_loss = 0
        if not self.val_dataloader:
            print(f"Epoch {epoch_num+1}: Validation dataloader is not available. Skipping validation.")
            if self.scheduler and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                is_after_warmup = not self.train_config.get("use_lr_warmup",False) or self.current_global_step >= self.train_config.get("lr_warmup_steps",0)
                if is_after_warmup: self.scheduler.step()
            return float('inf')
        num_val_batches = len(self.val_dataloader)
        if num_val_batches == 0:
            print(f"Epoch {epoch_num+1}: Validation dataloader is empty. Skipping validation.")
            if self.scheduler and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                is_after_warmup = not self.train_config.get("use_lr_warmup",False) or self.current_global_step >= self.train_config.get("lr_warmup_steps",0)
                if is_after_warmup: self.scheduler.step()
            return float('inf')
        profiler_active_this_epoch = (self.train_config.get("enable_profiler", False) and 
                                      epoch_num == self.train_config.get("profile_epoch_target", 0))
        prof_context_eval = None
        if profiler_active_this_epoch:
            print(f"--- Profiler activated for Validation, Epoch {epoch_num+1} ---")
            p_active_eval = min(5, num_val_batches)
            prof_log_dir = self.train_config.get("profiler_log_dir", "./profiler_logs_grug_v3")
            ensure_dir(Path(prof_log_dir) / "eval")
            prof_context_eval = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=0, warmup=0, active=p_active_eval, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(Path(prof_log_dir) / "eval"),
                record_shapes=True, profile_memory=True, with_stack=True )
            prof_context_eval.start()
        with torch.no_grad():
            for batch_idx_eval, (inputs, targets) in enumerate(self.val_dataloader):
                inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                with autocast(enabled=self.use_amp):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                val_loss += loss.item()
                if prof_context_eval and batch_idx_eval < p_active_eval: prof_context_eval.step()
        if prof_context_eval:
            prof_context_eval.stop()
            print(f"--- Profiler stopped for Validation, Epoch {epoch_num+1} ---")
            print(f"Validation Profiler traces saved to: {Path(prof_log_dir) / 'eval'}")
        avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else float('inf')
        print(f"Epoch {epoch_num+1}/{self.train_config['num_epochs']}, Validation Loss: {avg_val_loss:.4f}")
        if self.scheduler:
            is_after_warmup = not self.train_config.get("use_lr_warmup",False) or self.current_global_step >= self.train_config.get("lr_warmup_steps",0)
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau): self.scheduler.step(avg_val_loss)
            elif is_after_warmup: self.scheduler.step()
        self.model.train()
        if self.device.type == 'cuda': torch.cuda.empty_cache()
        return avg_val_loss

    def train(self, num_epochs):
        print(f"Starting GrugV3 training with model {self.model_name}..."); self.model.to(self.device)
        start_epoch = 0; best_val_loss = float('inf')
        loaded_info = self.load_checkpoint(self.train_config.get("resume_from_checkpoint"))
        if loaded_info:
            loaded_epoch = loaded_info.get('epoch', -1)
            self.current_global_step = loaded_info.get('current_global_step', 0)
            print(f"Resuming GrugV3 training from epoch {loaded_epoch + 1}. Global step set to {self.current_global_step}")
            start_epoch = loaded_epoch + 1
            if self.train_config.get("reset_best_val_loss_on_resume", False): best_val_loss = float('inf')
            elif loaded_info.get('loss') is not None: best_val_loss = loaded_info['loss']
            if self.use_amp and 'scaler_state_dict' in loaded_info and loaded_info['scaler_state_dict']:
                try: self.scaler.load_state_dict(loaded_info['scaler_state_dict']); print("GradScaler state loaded.")
                except: print("Warning: Could not load GradScaler state.")
        else: self.current_global_step = 0; print("No checkpoint for GrugV3, starting fresh.")
        for epoch in range(start_epoch, num_epochs):
            avg_train_loss = self.train_epoch(epoch)
            current_val_loss = self.evaluate_epoch(epoch)
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                print(f"New best GrugV3 validation loss: {best_val_loss:.4f}. Saving best model...")
                self.save_checkpoint(epoch, best_val_loss, is_best=True)
            epoch_checkpoint_filename = f"{self.model_name}_epoch_{epoch+1}.pth"
            self.save_checkpoint(epoch, current_val_loss, is_best=False, custom_filename=epoch_checkpoint_filename)
        print("GrugV3 Training finished.")

    def save_checkpoint(self, epoch, val_loss, is_best=False, custom_filename=None):
        checkpoint = { 'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                       'optimizer_state_dict': self.optimizer.state_dict(), 'loss': val_loss,
                       'config': self.current_config_for_checkpoint,
                       'current_global_step': self.current_global_step }
        if self.scheduler: checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        if self.use_amp: checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        filename = f"{self.model_name}_best.pth" if is_best else custom_filename if custom_filename else f"{self.model_name}_epoch_{epoch+1}_generic.pth"
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        print(f"GrugV3 Checkpoint saved to {filepath} (Epoch {epoch+1}, Val Loss: {val_loss:.4f})")

    def load_checkpoint(self, specific_checkpoint_path_str=None):
        load_path = None
        if specific_checkpoint_path_str:
            p = Path(specific_checkpoint_path_str)
            if p.is_file(): load_path = p
            else: print(f"Warning: GrugV3 resume_from_checkpoint path '{p}' not found.")
        if not load_path:
            if specific_checkpoint_path_str: print(f"GrugV3 Checkpoint '{specific_checkpoint_path_str}' not found.")
            return None
        try:
            print(f"Loading GrugV3 checkpoint from: {load_path}")
            checkpoint = torch.load(load_path, map_location=self.device)
            chkpt_config = checkpoint.get('config', {})
            if not chkpt_config: print("Warning: GrugV3 Checkpoint no config.")
            elif self.train_config.get("use_amp") != chkpt_config.get("use_amp"): print("Warning: AMP setting mismatch in GrugV3 checkpoint.")
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint and self.optimizer:
                try: self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']); print("Optimizer state loaded.")
                except ValueError as e: print(f"Warning: Optim state load fail: {e}")
            if 'scheduler_state_dict' in checkpoint and self.scheduler:
                try: self.scheduler.load_state_dict(checkpoint['scheduler_state_dict']); print("Scheduler state loaded.")
                except Exception as e: print(f"Warning: Scheduler state load fail: {e}")
            return { 'epoch': checkpoint.get('epoch', -1), 'loss': checkpoint.get('loss', float('inf')), 
                     'config': chkpt_config, 'current_global_step': checkpoint.get('current_global_step',0),
                     'scaler_state_dict': checkpoint.get('scaler_state_dict') }
        except Exception as e:
            print(f"Error loading GrugV3 checkpoint {load_path}: {e}"); traceback.print_exc()
        return None
# --- Model Architecture Components (Partially from GrugV2) ---
# --- Imports needed for Trainer/Predictor ---
import torch
import torch.optim as optim # For optim.lr_scheduler
from torch.cuda.amp import GradScaler, autocast # For AMP in Trainer
from pathlib import Path # For checkpoint paths
import traceback # For error printing
import time # For profiler
import torch.profiler # For profiler
# from mamba_ssm.utils.generation import InferenceParams # Not needed for GrugV3 Predictor

# --- Predictor (Adapted for GrugV3 - standard autoregressive generation) ---
class Predictor:
    def __init__(self, model, device, generation_config, model_internal_config):
        self.model = model.to(device).eval()
        self.device = device
        self.temperature = generation_config.get("generation_temperature", 1.0)
        self.top_k = generation_config.get("generation_top_k", 0)
        self.model_internal_config = model_internal_config # For max_len, sequence_length (context)
        if self.temperature <= 0: raise ValueError("Temperature must be positive.")
        print(f"Predictor initialized for GrugV3: Temp={self.temperature}, TopK={self.top_k}")
        print(f"Predictor using model's max_pos_len: {self.model_internal_config.get('max_positional_encoding_len', 'N/A')}")
        print(f"Predictor using model's training sequence_length for context: {self.model_internal_config.get('sequence_length', 'N/A')}")

    @torch.no_grad()
    def generate_sequence(self, seed_bytes, length=100):
        self.model.eval()
        
        if isinstance(seed_bytes, bytes):
            current_sequence_values = list(seed_bytes)
        elif isinstance(seed_bytes, list) and all(isinstance(x, int) for x in seed_bytes):
            current_sequence_values = list(seed_bytes)
        else:
            raise ValueError("seed_bytes must be bytes or list of ints.")

        generated_values = list(current_sequence_values) # Start with seed
        
        max_len_for_model_input = self.model_internal_config.get('max_positional_encoding_len', 512)
        context_len = self.model_internal_config.get('sequence_length', 16)

        for _ in range(length):
            start_idx = max(0, len(generated_values) - context_len)
            input_sequence = generated_values[start_idx:]
            
            if len(input_sequence) > max_len_for_model_input:
                input_sequence = input_sequence[-max_len_for_model_input:]

            if not input_sequence:
                if not generated_values: # if generated_values itself is empty (e.g. empty seed)
                    input_tensor = torch.tensor([[0]], dtype=torch.long).to(self.device)
                else: # This path should ideally not be taken if logic is correct
                    break # Stop if sequence becomes empty somehow mid-generation
            else:
                input_tensor = torch.tensor([input_sequence], dtype=torch.long).to(self.device)

            logits = self.model(input_tensor)
            logits_scaled = logits / self.temperature

            if self.top_k > 0:
                k = min(max(1, self.top_k), logits_scaled.size(-1))
                top_k_vals, top_k_indices = torch.topk(logits_scaled, k, dim=-1)
                filtered_logits = torch.full_like(logits_scaled, -float('Inf'))
                filtered_logits.scatter_(-1, top_k_indices, top_k_vals)
            else:
                filtered_logits = logits_scaled

            probabilities = torch.softmax(filtered_logits, dim=-1)
            
            if torch.isnan(probabilities).any() or probabilities.sum() < 1e-6:
                print("Warning: Invalid probabilities. Using argmax.")
                next_byte_val = torch.argmax(logits_scaled, dim=-1).item()
            else:
                next_byte_val = torch.multinomial(probabilities, 1).item()

            generated_values.append(next_byte_val)
        return bytes(generated_values)

# --- Trainer (Adapted for GrugV3) ---
class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, criterion, device, 
                 checkpoint_dir, model_name, scheduler=None, train_config=None):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_name = model_name
        self.scheduler = scheduler
        self.train_config = train_config if train_config else CONFIG_V3
        ensure_dir(self.checkpoint_dir)
        if self.train_config.get("enable_profiler"):
            ensure_dir(self.train_config.get("profiler_log_dir", "./profiler_logs_grug_v3"))

        self.current_config_for_checkpoint = self.train_config
        self.current_global_step = 0
        self.use_amp = self.train_config.get("use_amp", False) and self.device.type == 'cuda'
        self.scaler = GradScaler(enabled=self.use_amp)
        if self.use_amp: print("Automatic Mixed Precision (AMP) is ENABLED for training.")
        else: print("Automatic Mixed Precision (AMP) is DISABLED for training.")

    def _run_profiler_step(self, profiler_context, epoch_num, batch_idx, inputs, targets):
        inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=self.use_amp):
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
        self.scaler.scale(loss).backward()
        clip_val = self.train_config.get("clip_grad_norm_value")
        if clip_val is not None and clip_val > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_val)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if profiler_context: profiler_context.step()
        return loss.item()

    def _perform_lr_warmup(self):
        if self.train_config.get("use_lr_warmup", False) and \
           self.current_global_step < self.train_config.get("lr_warmup_steps", 0):
            warmup_steps = self.train_config["lr_warmup_steps"]
            target_lr = self.train_config["learning_rate"]
            init_factor = self.train_config.get("lr_warmup_init_factor", 0.01)
            if warmup_steps == 0: lr_scale = 1.0
            elif self.current_global_step == 0: lr_scale = init_factor
            else: lr_scale = init_factor + (1.0 - init_factor) * (self.current_global_step / warmup_steps)
            lr_scale = min(lr_scale, 1.0)
            for param_group in self.optimizer.param_groups: param_group['lr'] = target_lr * lr_scale
            if self.current_global_step == 0 or (self.current_global_step + 1) % (warmup_steps // 10 if warmup_steps >=10 else 1) == 0 or self.current_global_step == warmup_steps -1 :
                 print(f"Warmup Step {self.current_global_step+1}/{warmup_steps}, Current LR: {self.optimizer.param_groups[0]['lr']:.2e}")
        elif self.train_config.get("use_lr_warmup", False) and \
             self.current_global_step == self.train_config.get("lr_warmup_steps", 0):
            target_lr = self.train_config["learning_rate"]
            for param_group in self.optimizer.param_groups: param_group['lr'] = target_lr
            print(f"Warmup finished. LR set to target: {target_lr:.2e}")

    def run_interim_test(self, epoch_num, batch_idx):
        print(f"\n--- Interim Test @ Epoch {epoch_num+1}, Batch {batch_idx+1} ---")
        self.model.eval()
        interim_gen_config = {
            "generation_temperature": self.train_config.get("interim_test_temperature", 1.0),
            "generation_top_k": self.train_config.get("interim_test_top_k", 0)
        }
        model_cfg_for_pred = {
            "max_positional_encoding_len": self.train_config["max_positional_encoding_len"],
            "sequence_length": self.train_config["sequence_length"],
        }
        interim_predictor = Predictor(self.model, self.device, interim_gen_config, model_cfg_for_pred)
        seed_text = "The meaning of life is "
        seed_bytes = seed_text.encode('utf-8')
        print(f"Seed: '{seed_text}'")
        generated_bytes = interim_predictor.generate_sequence(seed_bytes, length=128)
        try:
            generated_text = generated_bytes.decode('utf-8', errors='replace')
            print(f"Generated (128 bytes): \"{generated_text}\"")
        except Exception as e:
            print(f"Error decoding generated bytes for interim test: {e}")
            print(f"Raw generated bytes: {generated_bytes}")
        self.model.train()
        print(f"--- End Interim Test ---\n")

    def train_epoch(self, epoch_num):
        self.model.train()
        epoch_loss = 0
        num_batches = len(self.train_dataloader)
        if num_batches == 0:
            print(f"Epoch {epoch_num+1}: Training dataloader is empty. Skipping.")
            return float('inf')
        profiler_active_this_epoch = (self.train_config.get("enable_profiler", False) and 
                                      epoch_num == self.train_config.get("profile_epoch_target", 0))
        prof_context = None
        if profiler_active_this_epoch:
            print(f"--- Profiler activated for Training, Epoch {epoch_num+1} ---")
            p_wait = self.train_config.get("profiler_schedule_wait", 5)
            p_warmup = self.train_config.get("profiler_schedule_warmup", 5)
            p_active = self.train_config.get("profiler_schedule_active", 10)
            p_repeat = self.train_config.get("profiler_schedule_repeat", 1)
            prof_schedule = torch.profiler.schedule(wait=p_wait, warmup=p_warmup, active=p_active, repeat=p_repeat)
            prof_log_dir = self.train_config.get("profiler_log_dir", "./profiler_logs_grug_v3")
            ensure_dir(Path(prof_log_dir) / "train")
            prof_context = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=prof_schedule,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(Path(prof_log_dir) / "train"),
                record_shapes=True, profile_memory=True, with_stack=True )
            prof_context.start()
        for batch_idx, (inputs, targets) in enumerate(self.train_dataloader):
            self._perform_lr_warmup()
            current_loss = self._run_profiler_step(profiler_context, epoch_num, batch_idx, inputs, targets)
            epoch_loss += current_loss
            self.current_global_step += 1
            if (batch_idx + 1) % self.train_config["print_every"] == 0 or (batch_idx + 1) == num_batches:
                print(f"Epoch {epoch_num+1}/{self.train_config['num_epochs']}, Batch {batch_idx+1}/{num_batches}, Train Loss: {current_loss:.4f}, Current LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            test_interval = self.train_config.get("test_every_batches", 0)
            if test_interval > 0 and (self.current_global_step % test_interval == 0) and self.current_global_step > 0:
                self.run_interim_test(epoch_num, batch_idx)
        if prof_context:
            prof_context.stop()
            print(f"--- Profiler stopped for Training, Epoch {epoch_num+1} ---")
            print(f"Training Profiler traces saved to: {Path(prof_log_dir) / 'train'}")
        if self.device.type == 'cuda': torch.cuda.empty_cache()
        return epoch_loss / num_batches if num_batches > 0 else float('inf')

    def evaluate_epoch(self, epoch_num):
        self.model.eval()
        val_loss = 0
        if not self.val_dataloader:
            print(f"Epoch {epoch_num+1}: Validation dataloader is not available. Skipping validation.")
            if self.scheduler and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                is_after_warmup = not self.train_config.get("use_lr_warmup",False) or self.current_global_step >= self.train_config.get("lr_warmup_steps",0)
                if is_after_warmup: self.scheduler.step()
            return float('inf')
        num_val_batches = len(self.val_dataloader)
        if num_val_batches == 0:
            print(f"Epoch {epoch_num+1}: Validation dataloader is empty. Skipping validation.")
            if self.scheduler and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                is_after_warmup = not self.train_config.get("use_lr_warmup",False) or self.current_global_step >= self.train_config.get("lr_warmup_steps",0)
                if is_after_warmup: self.scheduler.step()
            return float('inf')
        profiler_active_this_epoch = (self.train_config.get("enable_profiler", False) and 
                                      epoch_num == self.train_config.get("profile_epoch_target", 0))
        prof_context_eval = None
        if profiler_active_this_epoch:
            print(f"--- Profiler activated for Validation, Epoch {epoch_num+1} ---")
            p_active_eval = min(5, num_val_batches)
            prof_log_dir = self.train_config.get("profiler_log_dir", "./profiler_logs_grug_v3")
            ensure_dir(Path(prof_log_dir) / "eval")
            prof_context_eval = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=0, warmup=0, active=p_active_eval, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(Path(prof_log_dir) / "eval"),
                record_shapes=True, profile_memory=True, with_stack=True )
            prof_context_eval.start()
        with torch.no_grad():
            for batch_idx_eval, (inputs, targets) in enumerate(self.val_dataloader):
                inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                with autocast(enabled=self.use_amp):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                val_loss += loss.item()
                if prof_context_eval and batch_idx_eval < p_active_eval: prof_context_eval.step()
        if prof_context_eval:
            prof_context_eval.stop()
            print(f"--- Profiler stopped for Validation, Epoch {epoch_num+1} ---")
            print(f"Validation Profiler traces saved to: {Path(prof_log_dir) / 'eval'}")
        avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else float('inf')
        print(f"Epoch {epoch_num+1}/{self.train_config['num_epochs']}, Validation Loss: {avg_val_loss:.4f}")
        if self.scheduler:
            is_after_warmup = not self.train_config.get("use_lr_warmup",False) or self.current_global_step >= self.train_config.get("lr_warmup_steps",0)
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau): self.scheduler.step(avg_val_loss)
            elif is_after_warmup: self.scheduler.step()
        self.model.train()
        if self.device.type == 'cuda': torch.cuda.empty_cache()
        return avg_val_loss

    def train(self, num_epochs):
        print(f"Starting GrugV3 training with model {self.model_name}..."); self.model.to(self.device)
        start_epoch = 0; best_val_loss = float('inf')
        loaded_info = self.load_checkpoint(self.train_config.get("resume_from_checkpoint"))
        if loaded_info:
            loaded_epoch = loaded_info.get('epoch', -1)
            self.current_global_step = loaded_info.get('current_global_step', 0)
            print(f"Resuming GrugV3 training from epoch {loaded_epoch + 1}. Global step set to {self.current_global_step}")
            start_epoch = loaded_epoch + 1
            if self.train_config.get("reset_best_val_loss_on_resume", False): best_val_loss = float('inf')
            elif loaded_info.get('loss') is not None: best_val_loss = loaded_info['loss']
            if self.use_amp and 'scaler_state_dict' in loaded_info and loaded_info['scaler_state_dict']:
                try: self.scaler.load_state_dict(loaded_info['scaler_state_dict']); print("GradScaler state loaded.")
                except: print("Warning: Could not load GradScaler state.")
        else: self.current_global_step = 0; print("No checkpoint for GrugV3, starting fresh.")
        for epoch in range(start_epoch, num_epochs):
            avg_train_loss = self.train_epoch(epoch)
            current_val_loss = self.evaluate_epoch(epoch)
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                print(f"New best GrugV3 validation loss: {best_val_loss:.4f}. Saving best model...")
                self.save_checkpoint(epoch, best_val_loss, is_best=True)
            epoch_checkpoint_filename = f"{self.model_name}_epoch_{epoch+1}.pth"
            self.save_checkpoint(epoch, current_val_loss, is_best=False, custom_filename=epoch_checkpoint_filename)
        print("GrugV3 Training finished.")

    def save_checkpoint(self, epoch, val_loss, is_best=False, custom_filename=None):
        checkpoint = { 'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                       'optimizer_state_dict': self.optimizer.state_dict(), 'loss': val_loss,
                       'config': self.current_config_for_checkpoint,
                       'current_global_step': self.current_global_step }
        if self.scheduler: checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        if self.use_amp: checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        filename = f"{self.model_name}_best.pth" if is_best else custom_filename if custom_filename else f"{self.model_name}_epoch_{epoch+1}_generic.pth"
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        print(f"GrugV3 Checkpoint saved to {filepath} (Epoch {epoch+1}, Val Loss: {val_loss:.4f})")

    def load_checkpoint(self, specific_checkpoint_path_str=None):
        load_path = None
        if specific_checkpoint_path_str:
            p = Path(specific_checkpoint_path_str)
            if p.is_file(): load_path = p
            else: print(f"Warning: GrugV3 resume_from_checkpoint path '{p}' not found.")
        if not load_path:
            if specific_checkpoint_path_str: print(f"GrugV3 Checkpoint '{specific_checkpoint_path_str}' not found.")
            return None
        try:
            print(f"Loading GrugV3 checkpoint from: {load_path}")
            checkpoint = torch.load(load_path, map_location=self.device)
            chkpt_config = checkpoint.get('config', {})
            if not chkpt_config: print("Warning: GrugV3 Checkpoint no config.")
            elif self.train_config.get("use_amp") != chkpt_config.get("use_amp"): print("Warning: AMP setting mismatch in GrugV3 checkpoint.")
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint and self.optimizer:
                try: self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']); print("Optimizer state loaded.")
                except ValueError as e: print(f"Warning: Optim state load fail: {e}")
            if 'scheduler_state_dict' in checkpoint and self.scheduler:
                try: self.scheduler.load_state_dict(checkpoint['scheduler_state_dict']); print("Scheduler state loaded.")
                except Exception as e: print(f"Warning: Scheduler state load fail: {e}")
            return { 'epoch': checkpoint.get('epoch', -1), 'loss': checkpoint.get('loss', float('inf')), 
                     'config': chkpt_config, 'current_global_step': checkpoint.get('current_global_step',0),
                     'scaler_state_dict': checkpoint.get('scaler_state_dict') }
        except Exception as e:
            print(f"Error loading GrugV3 checkpoint {load_path}: {e}"); traceback.print_exc()
        return None
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        self.dropout = nn.Dropout(p=dropout)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.register_buffer("position_ids", torch.arange(max_len).expand((1, -1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        if seq_len > self.pos_embedding.num_embeddings: # num_embeddings is max_len
            raise ValueError(
                f"Input sequence length ({seq_len}) exceeds max_len "
                f"({self.pos_embedding.num_embeddings}) for positional embeddings."
            )
        pos_enc = self.pos_embedding(self.position_ids[:, :seq_len]) # Shape: (1, seq_len, d_model)
        x = x + pos_enc
        return self.dropout(x)

class CNNFrontend(nn.Module):
    def __init__(self, in_channels: int, out_channels_list: list[int], kernel_sizes: list[int],
                 stride: int = 1, cnn_dropout: float = 0.1, activation: str = "GELU",
                 use_layernorm: bool = True, padding_mode: str = "zeros"):
        super().__init__()
        if not (len(out_channels_list) == len(kernel_sizes)):
            raise ValueError("out_channels_list and kernel_sizes must have the same length.")

        self.conv_layers = nn.ModuleList()
        current_in_channels = in_channels
        for i, (k_size, o_channels) in enumerate(zip(kernel_sizes, out_channels_list)):
            padding = (k_size - 1) // 2
            conv_layer = nn.Conv1d(
                in_channels=current_in_channels,
                out_channels=o_channels,
                kernel_size=k_size,
                stride=stride,
                padding=padding,
                padding_mode=padding_mode,
                bias=True
            )
            self.conv_layers.append(conv_layer)
            current_in_channels = o_channels

        if activation.upper() == "RELU": self.activation_fn = nn.ReLU()
        elif activation.upper() == "GELU": self.activation_fn = nn.GELU()
        else: raise ValueError(f"Unsupported activation: {activation}")

        self.dropout_fn = nn.Dropout(cnn_dropout)
        self.use_layernorm = use_layernorm
        if self.use_layernorm:
            self.layer_norms = nn.ModuleList([nn.LayerNorm(ch) for ch in out_channels_list])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x shape: (batch_size, seq_len, embedding_dim)
        x = x.permute(0, 2, 1) # (batch_size, embedding_dim, seq_len) for Conv1D
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            x = self.activation_fn(x)
            if self.use_layernorm:
                x_permuted = x.permute(0, 2, 1)
                x_normed = self.layer_norms[i](x_permuted)
                x = x_normed.permute(0, 2, 1)
            x = self.dropout_fn(x)
        x = x.permute(0, 2, 1) # (batch_size, seq_len, final_cnn_channels)
        return x

# --- Grug V3 Model Architecture ---
class ByteLLM_GrugV3(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.config = model_config
        vocab_size = model_config["vocab_size"]
        embedding_dim = model_config["embedding_dim"]

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        current_dim_after_embedding = embedding_dim
        if model_config.get("use_cnn_frontend", False):
            self.cnn_frontend = CNNFrontend(
                in_channels=embedding_dim,
                out_channels_list=model_config["cnn_out_channels_list"],
                kernel_sizes=model_config["cnn_kernel_sizes"],
                stride=model_config.get("cnn_stride", 1),
                cnn_dropout=model_config.get("cnn_dropout", 0.1),
                activation=model_config.get("cnn_activation", "GELU"),
                use_layernorm=model_config.get("cnn_use_layernorm", True),
                padding_mode=model_config.get("cnn_padding_mode", "zeros")
            )
            current_dim_after_cnn = model_config["cnn_out_channels_list"][-1]
        else:
            self.cnn_frontend = None
            current_dim_after_cnn = embedding_dim

        dim_for_pe_and_attention = current_dim_after_cnn
        
        # Override attention_d_model in config if it's different from dim_for_pe_and_attention
        # This ensures the PE and Attention are compatible with the output of CNN/Embedding
        if model_config["attention_d_model"] != dim_for_pe_and_attention:
            print(f"Warning: Overriding config's attention_d_model ({model_config['attention_d_model']}) to {dim_for_pe_and_attention} to match CNN/Embedding output.")
            model_config["attention_d_model"] = dim_for_pe_and_attention

        self.positional_encoder = LearnablePositionalEncoding(
            d_model=model_config["attention_d_model"], # PE d_model should match attention d_model
            dropout=model_config.get("pe_dropout", 0.1),
            max_len=model_config["max_positional_encoding_len"]
        )

        # Standard Multihead Attention Layer(s)
        self.attention_layers = nn.ModuleList()
        for _ in range(model_config.get("num_attention_layers", 1)):
            attention_layer = nn.MultiheadAttention(
                embed_dim=model_config["attention_d_model"],
                num_heads=model_config["attention_num_heads"],
                dropout=model_config.get("attention_dropout", 0.1),
                batch_first=True # Crucial: input format (batch, seq, feature)
            )
            self.attention_layers.append(attention_layer)
            # Optional: Add LayerNorm after attention, and a FeedForward network
            # For simplicity, starting with just attention.
            
        self.output_dropout = nn.Dropout(model_config.get("output_dropout", 0.1))
        self.fc_out = nn.Linear(model_config["attention_d_model"], vocab_size)

        print(f"ByteLLM_GrugV3 Initialized. Embedding Dim: {embedding_dim}, CNN Out (if used): {current_dim_after_cnn if self.cnn_frontend else 'N/A'}, PE/Attention Dim: {model_config['attention_d_model']}, Vocab Size: {vocab_size}")

    def forward(self, x: torch.Tensor, inference_params=None): # Added inference_params
        # x shape: (batch_size, seq_len)
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)

        if self.cnn_frontend:
            x = self.cnn_frontend(x)  # (batch_size, seq_len, cnn_out_channels_list[-1])

        x = self.positional_encoder(x)  # (batch_size, seq_len, attention_d_model)

        # Multihead Attention
        # Input to MHA: (batch, seq_len, embed_dim) if batch_first=True
        # Output from MHA: (batch, seq_len, embed_dim) if batch_first=True
        attention_output = x
        for attn_layer in self.attention_layers:
            # Q, K, V are all the same for self-attention
            attention_output, _ = attn_layer(attention_output, attention_output, attention_output, need_weights=False)
            # Optional: Add residual connection, layer norm here if building a full Transformer block

        # For next-token prediction, use the representation of the *last* token
        output_representation = attention_output[:, -1, :]  # (batch_size, attention_d_model)
        output_representation = self.output_dropout(output_representation)
        logits = self.fc_out(output_representation)  # (batch_size, vocab_size)

        return logits

# --- Main execution (placeholder for basic testing) ---
if __name__ == '__main__':
    print("GrugV3 Model Definition File")
    
    # Example: Instantiate the model with the defined CONFIG_V3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create a dummy config, ensuring attention_d_model is consistent with CNN output
    test_config = CONFIG_V3.copy()
    if test_config["use_cnn_frontend"]:
        test_config["attention_d_model"] = test_config["cnn_out_channels_list"][-1]
    else:
        test_config["attention_d_model"] = test_config["embedding_dim"]

    try:
        model_v3 = ByteLLM_GrugV3(test_config).to(device)
        print("ByteLLM_GrugV3 model instantiated successfully.")

        # Create a dummy input tensor
        batch_size = 4
        seq_len = test_config["sequence_length"] # Use configured sequence length
        dummy_input = torch.randint(0, test_config["vocab_size"], (batch_size, seq_len), device=device)
        print(f"Dummy input shape: {dummy_input.shape}")

        # Perform a forward pass
        with torch.no_grad():
            logits = model_v3(dummy_input)
        print(f"Output logits shape: {logits.shape}")
        assert logits.shape == (batch_size, test_config["vocab_size"])
        print("Forward pass successful.")

    except Exception as e:
        print(f"Error during GrugV3 model test: {e}")
        import traceback
        traceback.print_exc()
