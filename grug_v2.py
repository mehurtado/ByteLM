import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F # For activations in CNN
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from pathlib import Path
import numpy as np
import traceback # For more detailed error printing
from mamba_ssm.utils.generation import InferenceParams
import torch.profiler # Added for profiling
import time # Added for simple timing if needed, and for profiler example
from torch.cuda.amp import GradScaler, autocast # For Automatic Mixed Precision

# --- ANOMALY DETECTION ---
# For debugging NaN/Inf issues. Disable for performance profiling as it adds overhead.
# Consider disabling this if you are confident NaNs are not an issue and want max speed.
torch.autograd.set_detect_anomaly(True) 
# print("WARNING: torch.autograd.set_detect_anomaly(True) is active. This will slow down training/profiling.")

# Attempt to import Mamba
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    print("Warning: mamba-ssm library not found. GrugV2 model will not be usable.")
    print("Please install it, e.g., 'pip install mamba-ssm causal-conv1d'.")
    MAMBA_AVAILABLE = False
    class Mamba: # Dummy class if not available
        def __init__(self, *args, **kwargs):
            raise ImportError("Mamba class not loaded. Install mamba-ssm.")
        def forward(self, *args, **kwargs):
            raise ImportError("Mamba class not loaded. Install mamba-ssm.")

# --- Configuration ---
CONFIG = {
    # Data and General
    "data_dir": "./dataset/USE",
    "processed_data_dir": "./dataset/USE", 
    "checkpoint_dir": "./checkpoints_grug_v2",
    "model_name": "grug_v2_cnn_mamba_amp", # Added _amp to model name
    "resume_from_checkpoint": "./checkpoints_grug_v2/grug_v2_cnn_mamba_amp_best.pth", # Set to path if resuming, e.g., "./checkpoints_grug_v2/grug_v2_cnn_mamba_amp_best.pth"
    "sequence_length": 16, 
    "batch_size": 32,  # INCREASED BATCH SIZE (Adjust based on your GPU memory)
    "vocab_size": 256,
    "val_split_ratio": 0.1,
    "num_workers": 8, 
    "generate_dummy_data_if_empty": True,
    "force_reprocess_data":  True, # Set to True to re-process data from scratch

    # Embedding
    "embedding_dim": 512,

    # CNN Frontend (Optional)
    "use_cnn_frontend": True,
    "cnn_out_channels_list": [512, 512],
    "cnn_kernel_sizes": [9, 3], 
    "cnn_stride": 1,
    "cnn_padding_mode": "zeros",
    "cnn_activation": "GELU",
    "cnn_dropout": 0.2,
    "cnn_use_layernorm": True,

    # Learnable Positional Encoding
    "max_positional_encoding_len": 512,
    "pe_dropout": 0.3,

    # Mamba Architecture
    "mamba_d_model": 512,
    "mamba_d_state": 16,
    "mamba_d_conv": 4,
    "mamba_expand": 2,
    "num_mamba_layers": 4,
    "mamba_bias": False,
    "mamba_conv_bias": True,

    # Output Layer
    "output_dropout": 0.2,

    # Training
    "num_epochs": 50,
    "learning_rate": 3e-5, # Initial learning rate
    "optimizer_type": "AdamW",
    "adam_beta1": 0.9,
    "adam_beta2": 0.98,
    "adam_eps": 1e-9,
    "weight_decay": 0.01,
    "scheduler_type": "CosineAnnealingLR", # Options: "ReduceLROnPlateau", "CosineAnnealingLR", None
    "lr_scheduler_T_max": 50 * 1000, # Example, adjust based on actual batches (num_epochs * batches_per_epoch)
    "lr_scheduler_eta_min": 1e-6,
    "lr_scheduler_patience": 10, # For ReduceLROnPlateau
    "lr_scheduler_factor": 0.1,  # For ReduceLROnPlateau
    "clip_grad_norm_value": 1.0,
    "print_every": 100, # Batches
    "test_every_batches": 500, # Batches, 0 to disable
    "reset_best_val_loss_on_resume": True,
    
    # LR Warmup
    "use_lr_warmup": True, 
    "lr_warmup_steps": 2000, 
    "lr_warmup_init_factor": 0.01, 

    # Automatic Mixed Precision (AMP) - NEW
    "use_amp": False, # Set to True to enable AMP for training

    # Generation / Prediction
    "generation_temperature": 1.0,
    "generation_top_k": 50,
    "interim_test_temperature": 0.6,
    "interim_test_top_k": 20,

    # Profiling
    "enable_profiler": False,  
    "profiler_log_dir": "./profiler_logs_grug_v2",
    "profile_epoch_target": 0, 
    "profiler_schedule_wait": 5, 
    "profiler_schedule_warmup": 5,  
    "profiler_schedule_active": 10, 
    "profiler_schedule_repeat": 1,  

    # Main script flow control
    "DO_TRAINING": True,
    "DO_PREDICTION": True,

    # CuDNN Benchmarking - NEW
    "cudnn_benchmark": True # Set to True if input sizes are consistent for potential speedup
}

# --- Utility Functions ---
def ensure_dir(directory_path):
    Path(directory_path).mkdir(parents=True, exist_ok=True)

def generate_dummy_data(data_dir, num_files=5, lines_per_file=10000): # Increased dummy data size
    ensure_dir(data_dir)
    if not any(Path(data_dir).iterdir()):
        print(f"Generating dummy data in {data_dir}...")
        for i in range(num_files):
            with open(Path(data_dir) / f"dummy_data_{i}.txt", "w", encoding="utf-8") as f:
                for j in range(lines_per_file):
                    f.write(f"This is line {j+1} of GrugV2 dummy file {i+1}. The quick brown fox jumps over the lazy dog. 0123456789. áéíóúñü. " * 5 + "\n")
        print("Dummy data generated.")
    else:
        print(f"Directory {data_dir} is not empty. Skipping dummy data generation.")

# --- Custom Dataset ---
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
        
        # .copy() is important here if using mmap, to avoid issues with shared memory when batching
        input_sequence_np = self.all_bytes[idx : idx + self.sequence_length].copy()
        target_np = self.all_bytes[idx + self.sequence_length].copy() # Ensure target is also copied
        
        input_tensor = torch.tensor(input_sequence_np, dtype=torch.long)
        target_tensor = torch.tensor(target_np, dtype=torch.long)
            
        return input_tensor, target_tensor

# --- Data Processor ---
class DataProcessor:
    def __init__(self, data_dir, processed_data_dir, sequence_length, force_reprocess=False):
        self.data_dir = Path(data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.sequence_length = sequence_length
        self.force_reprocess = force_reprocess
        ensure_dir(self.processed_data_dir)
        self.all_bytes_path = Path(self.processed_data_dir) / "all_bytes_grug_v2.npy"


    def load_or_create_all_bytes(self):
        if not self.force_reprocess and self.all_bytes_path.exists():
            print(f"Loading cached {self.all_bytes_path.name} using memory-mapping...")
            try:
                # Use 'r' for read-only mmap, which is generally safer
                all_bytes_mmap = np.load(self.all_bytes_path, mmap_mode='r')
                print(f"Successfully memory-mapped (Length: {len(all_bytes_mmap):,} bytes).")
                return all_bytes_mmap
            except Exception as e:
                print(f"Error memory-mapping {self.all_bytes_path}: {e}. Reprocessing...")

        print(f"Processing text files from {self.data_dir} to create {self.all_bytes_path.name}...")
        text_files = glob.glob(str(self.data_dir / "*.txt")) 
        if not text_files:
            print(f"No .txt files found in {self.data_dir}.")
            if CONFIG["generate_dummy_data_if_empty"]:
                generate_dummy_data(str(self.data_dir)) 
                text_files = glob.glob(str(self.data_dir / "*.txt"))
                if not text_files:
                    raise FileNotFoundError(f"Still no .txt files found in {self.data_dir} after dummy data generation.")
            else:
                raise FileNotFoundError(f"No .txt files found in {self.data_dir}.")

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
        # It's crucial that the encoded bytes are what you intend.
        # UTF-8 is standard. 'replace' handles encoding errors gracefully.
        encoded_bytes = final_text_string.encode('utf-8', errors='replace')
        all_bytes_np_array = np.array(list(encoded_bytes), dtype=np.uint8)

        if len(all_bytes_np_array) == 0:
            raise ValueError("Processed data resulted in an empty byte array after encoding.")

        np.save(self.all_bytes_path, all_bytes_np_array)
        print(f"Saved {self.all_bytes_path.name} (Length: {len(all_bytes_np_array):,} bytes). Now loading with memory-mapping...")
        all_bytes_mmap = np.load(self.all_bytes_path, mmap_mode='r') # mmap_mode='r'
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
            val_indices = np.array([]) # Empty array for no validation set
        else:
            print(f"Generating and shuffling indices for {num_total_sequences} sequences...")
            indices = np.arange(num_total_sequences)
            np.random.shuffle(indices) # Shuffle once before splitting
            print("Indices shuffled.")

            num_val_sequences = int(val_split_ratio * num_total_sequences)
            num_train_sequences = num_total_sequences - num_val_sequences

            if num_val_sequences == 0 or num_train_sequences == 0:
                print(f"Warning: Dataset size ({num_total_sequences}) too small for val_split_ratio ({val_split_ratio}). Adjusting split.")
                # Fallback: if one set would be empty, use all for training or adjust.
                # Here, we'll use all for training if val set is too small.
                if num_train_sequences < batch_size : # If train set is too small for even one batch
                     print(f"Warning: num_train_sequences ({num_train_sequences}) is smaller than batch_size ({batch_size}). This might lead to issues.")
                train_indices = indices 
                val_indices = np.array([]) # No validation if split is problematic
            else:
                train_indices = indices[:num_train_sequences]
                val_indices = indices[num_train_sequences:]
        
        train_dataset = Subset(full_dataset, train_indices.tolist())
        val_dataset = Subset(full_dataset, val_indices.tolist()) # Will be empty if val_indices is empty
        print(f"Training set size: {len(train_dataset)} sequences")
        print(f"Validation set size: {len(val_dataset)} sequences")

        pin_memory_flag = (num_workers > 0 and torch.cuda.is_available())
        # drop_last=True for training is common to ensure consistent batch sizes, especially for some model types.
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                      num_workers=num_workers, pin_memory=pin_memory_flag, drop_last=True)
        # drop_last=False for validation to evaluate on all validation data.
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                                    num_workers=num_workers, pin_memory=pin_memory_flag, drop_last=False) if len(val_dataset) > 0 else None
        
        if len(train_dataloader) == 0 and len(train_dataset) > 0 :
            print(f"Warning: Training DataLoader is empty. Batch size ({batch_size}) might be too large for training set ({len(train_dataset)}).")
        if val_dataloader is None and len(val_dataset) > 0:
             print(f"Info: Validation DataLoader not created as validation set is empty or too small.")
        elif val_dataloader and len(val_dataloader) == 0 and len(val_dataset) > 0:
            print(f"Warning: Validation DataLoader is empty. Batch size ({batch_size}) might be too large for validation set ({len(val_dataset)}).")
        return train_dataloader, val_dataloader

    def get_vocab_size(self):
        return 256 # For byte-level models

# --- Model Architecture Components ---
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        self.dropout = nn.Dropout(p=dropout)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        # register_buffer ensures 'position_ids' is part of the model's state_dict
        # and moved to the correct device, but not updated by the optimizer.
        self.register_buffer("position_ids", torch.arange(max_len).expand((1, -1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        if seq_len > self.pos_embedding.num_embeddings: # num_embeddings is max_len
            raise ValueError(
                f"Input sequence length ({seq_len}) exceeds max_len "
                f"({self.pos_embedding.num_embeddings}) for positional embeddings."
            )
        # Slicing position_ids to match the current sequence length
        pos_enc = self.pos_embedding(self.position_ids[:, :seq_len]) # Shape: (1, seq_len, d_model)
        x = x + pos_enc # Broadcasting pos_enc across batch dimension
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
            # Calculate padding for 'same' output length with stride 1
            # For stride > 1, this padding might not maintain exact length, but is common.
            padding = (k_size - 1) // 2 
            conv_layer = nn.Conv1d(
                in_channels=current_in_channels,
                out_channels=o_channels,
                kernel_size=k_size,
                stride=stride, 
                padding=padding, 
                padding_mode=padding_mode, # 'zeros' or 'replicate' or 'circular'
                bias=True # Bias is typically True for Conv layers unless followed by BatchNorm
            )
            self.conv_layers.append(conv_layer)
            current_in_channels = o_channels # Update for next layer

        if activation.upper() == "RELU": self.activation_fn = nn.ReLU()
        elif activation.upper() == "GELU": self.activation_fn = nn.GELU()
        # Add other activations as needed, e.g., SiLU/Swish
        # elif activation.upper() == "SILU": self.activation_fn = nn.SiLU() 
        else: raise ValueError(f"Unsupported activation: {activation}")

        self.dropout_fn = nn.Dropout(cnn_dropout)
        self.use_layernorm = use_layernorm
        if self.use_layernorm:
            # LayerNorm is applied on the feature dimension (channels after permuting)
            self.layer_norms = nn.ModuleList([nn.LayerNorm(ch) for ch in out_channels_list])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x shape: (batch_size, seq_len, embedding_dim)
        x = x.permute(0, 2, 1) # (batch_size, embedding_dim, seq_len) for Conv1D
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            x = self.activation_fn(x)
            if self.use_layernorm:
                # LayerNorm expects (batch, ..., features)
                x_permuted = x.permute(0, 2, 1) # (batch_size, seq_len, channels)
                x_normed = self.layer_norms[i](x_permuted)
                x = x_normed.permute(0, 2, 1) # Back to (batch_size, channels, seq_len)
            x = self.dropout_fn(x)
        x = x.permute(0, 2, 1) # (batch_size, seq_len, final_cnn_channels)
        return x

# --- Grug V2 Model Architecture ---
class ByteLLM_GrugV2(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        if not MAMBA_AVAILABLE:
            raise RuntimeError("Mamba library not available. Cannot instantiate ByteLLM_GrugV2.")
            
        self.config = model_config # Store config for reference
        vocab_size = model_config["vocab_size"]
        embedding_dim = model_config["embedding_dim"]
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        current_dim_after_embedding = embedding_dim
        if model_config.get("use_cnn_frontend", False):
            self.cnn_frontend = CNNFrontend(
                in_channels=embedding_dim, # CNN input is the embedding dim
                out_channels_list=model_config["cnn_out_channels_list"],
                kernel_sizes=model_config["cnn_kernel_sizes"],
                stride=model_config.get("cnn_stride",1),
                cnn_dropout=model_config.get("cnn_dropout", 0.1),
                activation=model_config.get("cnn_activation", "GELU"),
                use_layernorm=model_config.get("cnn_use_layernorm", True),
                padding_mode=model_config.get("cnn_padding_mode", "zeros")
            )
            # The output dimension of the CNN becomes the input to PE
            current_dim_after_cnn = model_config["cnn_out_channels_list"][-1]
        else:
            self.cnn_frontend = None
            current_dim_after_cnn = embedding_dim # No CNN, dim remains embedding_dim

        # Positional encoding input dimension matches output of CNN (or embedding if no CNN)
        dim_for_pe = current_dim_after_cnn
        self.positional_encoder = LearnablePositionalEncoding(
            d_model=dim_for_pe, 
            dropout=model_config.get("pe_dropout", 0.1),
            max_len=model_config["max_positional_encoding_len"]
        )
        
        # Dimension after PE is input to Mamba (potentially through a projection)
        dim_before_mamba = dim_for_pe
        if dim_before_mamba != model_config["mamba_d_model"]:
            # Project if PE output dim doesn't match Mamba's d_model
            self.input_projection_to_mamba = nn.Linear(dim_before_mamba, model_config["mamba_d_model"])
            dim_into_mamba = model_config["mamba_d_model"]
            print(f"Info: Projecting input for Mamba from {dim_before_mamba} to {dim_into_mamba}")
        else:
            self.input_projection_to_mamba = None # No projection needed
            dim_into_mamba = dim_before_mamba

        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=dim_into_mamba, # This is the Mamba internal dimension
                d_state=model_config.get("mamba_d_state", 16),
                d_conv=model_config.get("mamba_d_conv", 4),
                expand=model_config.get("mamba_expand", 2),
                bias=model_config.get("mamba_bias", False),
                conv_bias=model_config.get("mamba_conv_bias", True),
                # layer_idx is used by Mamba's internal init, if it supports it.
                # Check Mamba documentation for exact parameter names if issues arise.
                layer_idx=i 
            ) for i in range(model_config["num_mamba_layers"])
        ])
        
        self.output_dropout = nn.Dropout(model_config.get("output_dropout", 0.1))
        # The Mamba layers output dim_into_mamba, which is then fed to the final linear layer
        self.fc_out = nn.Linear(dim_into_mamba, vocab_size)

        print(f"ByteLLM_GrugV2 Initialized. Embedding Dim: {embedding_dim}, CNN Out (if used): {current_dim_after_cnn if self.cnn_frontend else 'N/A'}, PE Dim: {dim_for_pe}, Mamba d_model: {dim_into_mamba}, Vocab Size: {vocab_size}")

    def forward(self, x: torch.Tensor, inference_params=None):
        # x shape: (batch_size, seq_len)
        x = self.embedding(x) # (batch_size, seq_len, embedding_dim)
        
        if self.cnn_frontend:
            x = self.cnn_frontend(x) # (batch_size, seq_len, cnn_out_channels_list[-1])
        
        x = self.positional_encoder(x) # (batch_size, seq_len, dim_for_pe)
        
        if self.input_projection_to_mamba:
            x = self.input_projection_to_mamba(x) # (batch_size, seq_len, mamba_d_model)
            
        # Mamba layers
        # The inference_params are passed along if provided (for generation)
        for mamba_layer in self.mamba_layers:
            if inference_params is not None:
                x = mamba_layer(x, inference_params=inference_params)
            else:
                x = mamba_layer(x) # (batch_size, seq_len, mamba_d_model)
        
        # For next-token prediction, typically use the representation of the *last* token in the sequence
        output_representation = x[:, -1, :] # (batch_size, mamba_d_model)
        output_representation = self.output_dropout(output_representation)
        logits = self.fc_out(output_representation) # (batch_size, vocab_size)
        
        return logits

# --- Predictor ---
class Predictor:
    def __init__(self, model, device, generation_config, model_internal_config):
        self.model = model.to(device).eval() # Ensure model is on device and in eval mode
        self.device = device
        self.temperature = generation_config.get("generation_temperature", 1.0)
        self.top_k = generation_config.get("generation_top_k", 0) # 0 means no top-k filtering
        self.model_internal_config = model_internal_config # For max_seqlen etc.
        if self.temperature <= 0: raise ValueError("Temperature must be positive.")
        print(f"Predictor initialized: Temp={self.temperature}, TopK={self.top_k}")
        print(f"Predictor using model's max_pos_len: {self.model_internal_config['max_positional_encoding_len']}")
        print(f"Predictor using model's training sequence_length for context: {self.model_internal_config['sequence_length']}")


    @torch.no_grad() # Essential for inference
    def generate_sequence(self, seed_bytes, length=100):
        self.model.eval() # Redundant if constructor ensures it, but good practice
        
        if isinstance(seed_bytes, bytes):
            current_sequence_values = list(seed_bytes)
        elif isinstance(seed_bytes, list) and all(isinstance(x, int) for x in seed_bytes):
            current_sequence_values = list(seed_bytes) # Make a mutable copy
        else:
            raise ValueError("seed_bytes must be bytes or list of ints.")

        generated_values = list(current_sequence_values) # Keep track of all generated tokens

        # Initialize Mamba's inference parameters
        # max_seqlen should be the model's maximum supported sequence length (from positional encoding)
        # max_batch_size is 1 for iterative generation.
        inference_params = InferenceParams(
            max_seqlen=self.model_internal_config['max_positional_encoding_len'], 
            max_batch_size=1 
        )
        inference_params.seqlen_offset = 0 # Initialize offset

        # Prime the model with the seed sequence if it's not empty
        if current_sequence_values:
            # Input tensor for priming: (batch_size=1, seed_len)
            # Ensure seed is not longer than model's max_positional_encoding_len
            prime_seq = current_sequence_values[:self.model_internal_config['max_positional_encoding_len']]
            input_tensor = torch.tensor([prime_seq], dtype=torch.long).to(self.device)
            
            # Pass the entire seed sequence through the model to set up Mamba's internal states
            # The logits output here is not used for generation, only for updating Mamba states
            _ = self.model(input_tensor, inference_params=inference_params)
            # Mamba's internal logic should update inference_params.seqlen_offset based on input_tensor.shape[1]
            # For subsequent steps, only the last token is needed as input.

        # Generate new tokens one by one
        for _ in range(length):
            if not generated_values: # Handle case of empty seed (generate from scratch)
                # You might want a default start token, e.g., a PAD token or a common byte like space.
                # For byte models, 0 is a valid byte.
                last_token_val = 0 # Or some other default start token if desired
            else:
                last_token_val = generated_values[-1]
            
            # Input for next token prediction: (batch_size=1, seq_len=1)
            input_tensor = torch.tensor([[last_token_val]], dtype=torch.long).to(self.device)
            
            # Get logits for the next token
            # inference_params will be used by Mamba layers to manage state across calls
            logits = self.model(input_tensor, inference_params=inference_params) # Logits shape: (1, vocab_size)
            # Mamba layer's forward pass should use and update inference_params.seqlen_offset correctly

            # Apply temperature scaling
            logits_scaled = logits / self.temperature

            # Apply top-k filtering
            if self.top_k > 0:
                k = min(max(1, self.top_k), logits_scaled.size(-1)) # Ensure k is valid
                top_k_vals, top_k_indices = torch.topk(logits_scaled, k, dim=-1)
                # Create a new tensor filled with -inf, then scatter top_k_vals
                filtered_logits = torch.full_like(logits_scaled, -float('Inf'))
                filtered_logits.scatter_(-1, top_k_indices, top_k_vals)
            else:
                filtered_logits = logits_scaled # No top-k filtering

            # Convert logits to probabilities
            probabilities = torch.softmax(filtered_logits, dim=-1)
            
            # Handle potential NaN/Inf in probabilities (e.g., from extreme temperatures or model issues)
            if torch.isnan(probabilities).any() or probabilities.sum() < 1e-6: # Check for invalid probability distribution
                print("Warning: Invalid probabilities detected during generation. Using argmax as fallback.")
                next_byte_val = torch.argmax(logits_scaled, dim=-1).item() # Fallback to greedy
            else:
                # Sample from the probability distribution
                next_byte_val = torch.multinomial(probabilities, 1).item()

            generated_values.append(next_byte_val)
            
            # If generated_values exceeds max_positional_encoding_len, Mamba's state handling
            # with inference_params should manage this (e.g., by effectively using a sliding window).
            # The input to the model is always the single last token.

        return bytes(generated_values) # Convert list of int byte values to a bytes object

# --- Trainer ---
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
        self.train_config = train_config if train_config else CONFIG # Use global CONFIG if not provided
        ensure_dir(self.checkpoint_dir)
        ensure_dir(self.train_config.get("profiler_log_dir", "./profiler_logs_grug_v2")) # Ensure profiler log dir exists
        
        self.current_config_for_checkpoint = CONFIG # Save the global CONFIG at the time of training
        self.current_global_step = 0 # For LR warmup and potentially other step-based logic

        # AMP: Initialize GradScaler if AMP is enabled
        self.use_amp = self.train_config.get("use_amp", False) and self.device.type == 'cuda'
        self.scaler = GradScaler(enabled=self.use_amp)
        if self.use_amp:
            print("Automatic Mixed Precision (AMP) is ENABLED for training.")
        else:
            print("Automatic Mixed Precision (AMP) is DISABLED for training.")


    def _run_profiler_step(self, profiler_context, epoch_num, batch_idx, inputs, targets):
        """Helper to run a single training step under profiler context."""
        inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True) # non_blocking for pinned memory
        
        self.optimizer.zero_grad(set_to_none=True) # set_to_none=True can improve performance slightly

        # AMP: Use autocast context manager
        with autocast(enabled=self.use_amp):
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
        
        # AMP: Scale loss and call backward()
        self.scaler.scale(loss).backward()
        
        # Gradient clipping (unscale gradients first if using AMP)
        clip_val = self.train_config.get("clip_grad_norm_value")
        if clip_val is not None and clip_val > 0:
            self.scaler.unscale_(self.optimizer) # Unscale before clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_val)
        
        # AMP: optimizer.step() through scaler
        self.scaler.step(self.optimizer)
        self.scaler.update() # Update scaler for next iteration

        if profiler_context: # only call profiler.step if it's active
             profiler_context.step()
        return loss.item()

    def _perform_lr_warmup(self):
        """Performs learning rate warmup if configured."""
        if self.train_config.get("use_lr_warmup", False) and \
           self.current_global_step < self.train_config.get("lr_warmup_steps", 0):
            
            warmup_steps = self.train_config["lr_warmup_steps"]
            target_lr = self.train_config["learning_rate"] # This is the LR *after* warmup
            init_factor = self.train_config.get("lr_warmup_init_factor", 0.01)
            initial_lr = target_lr * init_factor

            if warmup_steps == 0: # Avoid division by zero if warmup_steps is misconfigured
                lr_scale = 1.0
            elif self.current_global_step == 0: # First step, set to initial LR
                 lr_scale = init_factor # This makes current_lr = target_lr * init_factor
            else:
                # Linear warmup from initial_lr to target_lr
                # lr = initial_lr + (target_lr - initial_lr) * (self.current_global_step / warmup_steps)
                # So, lr = target_lr * [init_factor + (1 - init_factor) * (step / warmup_steps)]
                lr_scale = init_factor + (1.0 - init_factor) * (self.current_global_step / warmup_steps)
            
            # Ensure lr_scale doesn't exceed 1.0 if current_global_step somehow overshoots
            lr_scale = min(lr_scale, 1.0) 

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = target_lr * lr_scale # Scale the target_lr
            
            if self.current_global_step == 0 or (self.current_global_step + 1) % (warmup_steps // 10 if warmup_steps >= 10 else 1) == 0 or self.current_global_step == warmup_steps -1:
                print(f"Warmup Step {self.current_global_step+1}/{warmup_steps}, Current LR: {self.optimizer.param_groups[0]['lr']:.2e}")
        
        elif self.train_config.get("use_lr_warmup", False) and \
             self.current_global_step == self.train_config.get("lr_warmup_steps", 0):
            # Ensure target LR is set exactly after warmup, before main scheduler might take over
            target_lr = self.train_config["learning_rate"]
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = target_lr
            print(f"Warmup finished. LR set to target: {target_lr:.2e}")


    def run_interim_test(self, epoch_num, batch_idx):
        print(f"\n--- Interim Test @ Epoch {epoch_num+1}, Batch {batch_idx+1} ---")
        self.model.eval() # Set model to evaluation mode
        interim_gen_config = {
            "generation_temperature": self.train_config.get("interim_test_temperature", 1.0),
            "generation_top_k": self.train_config.get("interim_test_top_k", 0)
        }
        # Pass relevant parts of the main model's config to the predictor
        model_cfg_for_pred = {
            "max_positional_encoding_len": self.train_config["max_positional_encoding_len"],
            "sequence_length": self.train_config["sequence_length"], # Training sequence_length for context
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
        
        self.model.train() # Set model back to training mode
        print("--- End Interim Test ---\n")

    def train_epoch(self, epoch_num):
        self.model.train() # Ensure model is in training mode
        epoch_loss = 0
        num_batches = len(self.train_dataloader)
        if num_batches == 0:
            print(f"Epoch {epoch_num+1}: Training dataloader is empty. Skipping.")
            return float('inf') # Return inf if no training happened

        # Profiler setup for the target epoch
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
            prof_log_dir = self.train_config.get("profiler_log_dir", "./profiler_logs_grug_v2")
            ensure_dir(prof_log_dir) # Ensure directory exists

            prof_context = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=prof_schedule,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(Path(prof_log_dir) / "train"), # Suffix for train
                record_shapes=True,
                profile_memory=True, # Enable memory profiling
                with_stack=True # Enable stack tracing (can add overhead)
            )
            prof_context.start()

        for batch_idx, (inputs, targets) in enumerate(self.train_dataloader):
            self._perform_lr_warmup() # Perform LR warmup at the start of the step

            current_loss = self._run_profiler_step(prof_context, epoch_num, batch_idx, inputs, targets)
            epoch_loss += current_loss
            self.current_global_step += 1 # Increment global step

            if (batch_idx + 1) % self.train_config["print_every"] == 0 or (batch_idx + 1) == num_batches:
                print(f"Epoch {epoch_num+1}/{self.train_config['num_epochs']}, Batch {batch_idx+1}/{num_batches}, Train Loss: {current_loss:.4f}, Current LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            test_interval = self.train_config.get("test_every_batches", 0)
            if test_interval > 0 and (self.current_global_step % test_interval == 0) and self.current_global_step > 0: # Use global_step
                self.run_interim_test(epoch_num, batch_idx)
        
        if prof_context:
            prof_context.stop()
            print(f"--- Profiler stopped for Training, Epoch {epoch_num+1} ---")
            # You can print key averages or direct users to TensorBoard
            print(f"Training Profiler traces saved to: {Path(prof_log_dir) / 'train'}")
            # Example: print(prof_context.key_averages().table(sort_by="cuda_time_total", row_limit=10))

        if self.device.type == 'cuda': torch.cuda.empty_cache() # Clear cache at end of epoch
        
        # LR scheduler step (for epoch-based schedulers like CosineAnnealingLR)
        # This is typically done *after* the validation epoch if the scheduler depends on validation metrics (like ReduceLROnPlateau)
        # For CosineAnnealingLR, it's often stepped per epoch after training.
        # The current logic places scheduler step in evaluate_epoch, which is fine for both types.
        return epoch_loss / num_batches if num_batches > 0 else float('inf')

    def evaluate_epoch(self, epoch_num):
        self.model.eval() # Set model to evaluation mode
        val_loss = 0
        
        if not self.val_dataloader: # Handle case where val_dataloader might be None
            print(f"Epoch {epoch_num+1}: Validation dataloader is not available. Skipping validation.")
            # If no validation, scheduler step for epoch-based schedulers might need to happen here
            # or after train_epoch, if not ReduceLROnPlateau.
            # For simplicity, we'll assume if val_dataloader is None, ReduceLROnPlateau isn't used or won't step.
            # CosineAnnealingLR should still step.
            if self.scheduler and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                is_after_warmup = not self.train_config.get("use_lr_warmup", False) or \
                                  self.current_global_step >= self.train_config.get("lr_warmup_steps", 0)
                if is_after_warmup:
                    current_lr_before_step = self.optimizer.param_groups[0]['lr']
                    self.scheduler.step()
                    print(f"LR Scheduler ({type(self.scheduler).__name__}) step (no validation). Current LR: {current_lr_before_step:.2e} -> New LR: {self.optimizer.param_groups[0]['lr']:.2e}")
                else:
                     print(f"LR Scheduler ({type(self.scheduler).__name__}) step skipped during warmup (no validation).")
            return float('inf') # Return inf if no validation happened

        num_val_batches = len(self.val_dataloader)
        if num_val_batches == 0:
            print(f"Epoch {epoch_num+1}: Validation dataloader is empty. Skipping validation.")
            # Similar scheduler logic as above if val_dataloader is empty but was created.
            if self.scheduler and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                is_after_warmup = not self.train_config.get("use_lr_warmup", False) or \
                                  self.current_global_step >= self.train_config.get("lr_warmup_steps", 0)
                if is_after_warmup:
                    current_lr_before_step = self.optimizer.param_groups[0]['lr']
                    self.scheduler.step()
                    print(f"LR Scheduler ({type(self.scheduler).__name__}) step (empty validation). Current LR: {current_lr_before_step:.2e} -> New LR: {self.optimizer.param_groups[0]['lr']:.2e}")
                else:
                    print(f"LR Scheduler ({type(self.scheduler).__name__}) step skipped during warmup (empty validation).")
            return float('inf') 

        profiler_active_this_epoch = (self.train_config.get("enable_profiler", False) and 
                                      epoch_num == self.train_config.get("profile_epoch_target", 0))
        prof_context_eval = None
        if profiler_active_this_epoch and num_val_batches > 0 : 
            print(f"--- Profiler activated for Validation, Epoch {epoch_num+1} ---")
            p_active_eval = min(5, num_val_batches) 
            prof_log_dir = self.train_config.get("profiler_log_dir", "./profiler_logs_grug_v2")
            
            prof_context_eval = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=0, warmup=0, active=p_active_eval, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(Path(prof_log_dir) / "eval"),
                record_shapes=True, profile_memory=True, with_stack=True
            )
            prof_context_eval.start()

        with torch.no_grad(): # Ensure no gradients are computed during validation
            for batch_idx_eval, (inputs, targets) in enumerate(self.val_dataloader):
                inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                # AMP: autocast for evaluation if model was trained with it, though not strictly necessary for `no_grad`
                # It ensures consistency if any layers behave differently with mixed precision.
                with autocast(enabled=self.use_amp): 
                    outputs = self.model(inputs) 
                    loss = self.criterion(outputs, targets)
                val_loss += loss.item()
                if prof_context_eval and batch_idx_eval < p_active_eval: 
                    prof_context_eval.step()
        
        if prof_context_eval:
            prof_context_eval.stop()
            print(f"--- Profiler stopped for Validation, Epoch {epoch_num+1} ---")
            print(f"Validation Profiler traces saved to: {Path(prof_log_dir) / 'eval'}")

        avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else float('inf')
        print(f"Epoch {epoch_num+1}/{self.train_config['num_epochs']}, Validation Loss: {avg_val_loss:.4f}")
        
        # LR Scheduler Step
        current_lr_before_step = self.optimizer.param_groups[0]['lr']
        if self.scheduler:
            is_after_warmup = not self.train_config.get("use_lr_warmup", False) or \
                              self.current_global_step >= self.train_config.get("lr_warmup_steps", 0)

            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(avg_val_loss) # ReduceLROnPlateau steps based on metric
                print(f"LR Scheduler (ReduceLROnPlateau) step. Current LR: {current_lr_before_step:.2e} -> New LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            elif is_after_warmup: # For schedulers like CosineAnnealingLR that step each epoch
                self.scheduler.step()
                print(f"LR Scheduler ({type(self.scheduler).__name__}) step. Current LR: {current_lr_before_step:.2e} -> New LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            else: # Scheduler exists but it's warmup phase for an epoch-stepping one
                print(f"LR Scheduler ({type(self.scheduler).__name__}) step skipped during warmup. Current LR: {current_lr_before_step:.2e}")
        else:
            print(f"No LR Scheduler. Current LR: {current_lr_before_step:.2e}")


        self.model.train() # Set model back to training mode for the next epoch
        if self.device.type == 'cuda': torch.cuda.empty_cache()
        return avg_val_loss

    def train(self, num_epochs):
        print("Starting GrugV2 training..."); self.model.to(self.device)
        
        start_epoch = 0
        best_val_loss = float('inf')
        
        loaded_info = self.load_checkpoint(self.train_config.get("resume_from_checkpoint"))
        
        if loaded_info:
            loaded_epoch = loaded_info.get('epoch', -1) # epoch is 0-indexed
            # Use the loaded global step directly if available
            self.current_global_step = loaded_info.get('current_global_step', 0)
            
            print(f"Resuming training from epoch {loaded_epoch + 1}. Global step set to {self.current_global_step}")
            start_epoch = loaded_epoch + 1 # Next epoch to run
            
            if self.train_config.get("reset_best_val_loss_on_resume", False):
                print("Resetting best_val_loss to infinity for this run due to config.")
                best_val_loss = float('inf')
            elif loaded_info.get('loss') is not None and loaded_info.get('loss') != float('inf'):
                best_val_loss = loaded_info['loss']
                print(f"Previous best validation loss loaded: {best_val_loss:.4f}")
            else:
                print("No valid previous validation loss in checkpoint or reset_best_val_loss is true. Best_val_loss is infinity.")
            
            # AMP: Load scaler state if resuming and AMP was used
            if self.use_amp and 'scaler_state_dict' in loaded_info and loaded_info['scaler_state_dict'] is not None:
                try:
                    self.scaler.load_state_dict(loaded_info['scaler_state_dict'])
                    print("GradScaler state loaded successfully.")
                except Exception as e:
                    print(f"Warning: Could not load GradScaler state: {e}. Initializing fresh scaler.")
            elif self.use_amp:
                 print("Warning: Resuming with AMP but no GradScaler state found in checkpoint. Initializing fresh scaler.")


        else: # Not resuming from checkpoint
            self.current_global_step = 0 # Start from 0 if not resuming
            print("No checkpoint loaded or specified, starting training from scratch. Global step set to 0.")


        for epoch in range(start_epoch, num_epochs):
            avg_train_loss = self.train_epoch(epoch)
            current_val_loss = self.evaluate_epoch(epoch)

            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                print(f"New best validation loss: {best_val_loss:.4f}. Saving best model...")
                self.save_checkpoint(epoch, best_val_loss, is_best=True)
            
            # Optionally save checkpoint every N epochs or always
            # Here, saving every epoch as an example
            epoch_checkpoint_filename = f"{self.model_name}_epoch_{epoch+1}.pth"
            print(f"Saving checkpoint for epoch {epoch+1} to {epoch_checkpoint_filename}...")
            self.save_checkpoint(epoch, current_val_loss, is_best=False, custom_filename=epoch_checkpoint_filename)
        
        print("Training finished.")

    def save_checkpoint(self, epoch, val_loss, is_best=False, custom_filename=None):
        checkpoint = {
            'epoch': epoch, # 0-indexed epoch number that just completed
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': val_loss, # Validation loss for this epoch
            'config': self.current_config_for_checkpoint, # Save the config used for this training run
            'current_global_step': self.current_global_step # Save global step
        }
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # AMP: Save GradScaler state if using AMP
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        if is_best:
            filename = f"{self.model_name}_best.pth"
        elif custom_filename:
            filename = custom_filename
        else: # Fallback if not best and no custom name (should ideally have one)
            filename = f"{self.model_name}_epoch_{epoch+1}_chkpt.pth"

        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath} (Epoch {epoch+1}, Val Loss: {val_loss:.4f}, Global Step: {self.current_global_step})")

    def load_checkpoint(self, specific_checkpoint_path_str=None):
        load_path = None
        if specific_checkpoint_path_str:
            p = Path(specific_checkpoint_path_str)
            if p.is_file(): 
                load_path = p
            else: 
                print(f"Warning: Specified resume_from_checkpoint path '{p}' not found.")
        
        # If no specific path, don't try to load default best automatically here.
        # Let main logic decide if it wants to load a 'best' model for prediction later.
        # This function is primarily for resuming training.

        if not load_path: 
            # If specific_checkpoint_path_str was None or path not found, don't attempt to load.
            if specific_checkpoint_path_str: # Only print if a path was given but not found
                 print(f"Checkpoint '{specific_checkpoint_path_str}' not found. Starting from scratch or as configured.")
            return None # No checkpoint loaded

        try:
            print(f"Loading checkpoint from: {load_path}")
            # map_location ensures model loads to the correct device, esp. if saved on GPU and loading on CPU
            checkpoint = torch.load(load_path, map_location=self.device)
            
            chkpt_config = checkpoint.get('config', {}) # Get config from checkpoint
            if not chkpt_config:
                print("Warning: Checkpoint does not contain configuration. Model architecture may be incompatible if current CONFIG differs significantly.")
                # Could fall back to current CONFIG, but this is risky.
                # For safety, one might choose to raise an error or require manual config alignment.
            else:
                # Optional: Compare critical architectural params between chkpt_config and current self.train_config (which is CONFIG)
                # This is complex because the model is already initialized.
                # A robust way is to re-initialize model, optimizer, scheduler based on chkpt_config if they differ.
                # For now, we assume the user ensures compatibility or the model is re-initialized before calling load_checkpoint.
                # The current ByteLLM_GrugV2 is initialized with global CONFIG. If chkpt_config differs, there could be issues.
                # A safer approach:
                # 1. Load chkpt_config first.
                # 2. Initialize model, optimizer, scheduler with chkpt_config.
                # 3. Then load state_dicts.
                # This is handled in the main script flow before Trainer initialization for robustness.
                # Here, we'll just print a warning if `use_amp` differs, as it affects scaler.
                if self.train_config.get("use_amp") != chkpt_config.get("use_amp"):
                    print(f"Warning: AMP setting mismatch. Current config use_amp={self.train_config.get('use_amp')}, checkpoint config use_amp={chkpt_config.get('use_amp')}. GradScaler might behave unexpectedly if not re-initialized.")


            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            if 'optimizer_state_dict' in checkpoint and self.optimizer:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print("Optimizer state loaded successfully.")
                except ValueError as e: 
                    print(f"Warning: Could not load optimizer state: {e}. Optimizer state will be fresh. This can happen if model parameters changed.")
            
            if 'scheduler_state_dict' in checkpoint and self.scheduler:
                try:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    print("Scheduler state loaded successfully.")
                except Exception as e: # Catch more general exceptions for scheduler
                     print(f"Warning: Could not load scheduler state: {e}. Scheduler state will be fresh.")

            epoch = checkpoint.get('epoch', -1) # epoch is 0-indexed
            loss = checkpoint.get('loss', float('inf')) # val_loss at that epoch
            loaded_global_step = checkpoint.get('current_global_step', (epoch + 1) * (len(self.train_dataloader) if self.train_dataloader else 1)) # Fallback for older checkpoints
            
            # Scaler state will be loaded in the train() method after this returns.
            # We return the full checkpoint so train() can access scaler_state_dict.
            
            print(f"Checkpoint loaded: Epoch {epoch+1} (completed), Saved Loss: {loss:.4f}, Global Step: {loaded_global_step}")
            # Return all relevant info, including the scaler state if present
            return {
                'epoch': epoch, 
                'loss': loss, 
                'config': chkpt_config, 
                'current_global_step': loaded_global_step,
                'scaler_state_dict': checkpoint.get('scaler_state_dict') # Pass scaler state
            }

        except RuntimeError as e: # Often due to model key mismatches
            print(f"Error loading state_dict for checkpoint {load_path}: {e}")
            traceback.print_exc()
        except Exception as e: # Generic catch-all
            print(f"Generic error loading checkpoint {load_path}: {e}")
            traceback.print_exc()
        
        print("Failed to load checkpoint or error occurred. Training will start from scratch or as configured.")
        return None

# --- Helper Functions for Main Execution ---
def setup_environment(config_dict):
    ensure_dir(config_dict["data_dir"]) 
    ensure_dir(config_dict["checkpoint_dir"])
    ensure_dir(config_dict["processed_data_dir"])
    if config_dict.get("profiler_log_dir") and config_dict.get("enable_profiler"):
        ensure_dir(config_dict["profiler_log_dir"]) 
        ensure_dir(Path(config_dict["profiler_log_dir"]) / "train") # For train traces
        ensure_dir(Path(config_dict["profiler_log_dir"]) / "eval") # For eval traces

    if config_dict.get("generate_dummy_data_if_empty", True):
        generate_dummy_data(config_dict["data_dir"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
        if config_dict.get("cudnn_benchmark", False):
            torch.backends.cudnn.benchmark = True
            print("torch.backends.cudnn.benchmark = True (May speed up training if input sizes are constant)")
    
    print(f"Model name for checkpoints: {config_dict['model_name']}")
    return device

def load_data_components(config_dict):
    current_seq_len = config_dict["sequence_length"]
    print(f"Initializing DataProcessor with sequence_length: {current_seq_len}")
    
    data_processor = DataProcessor(
        config_dict["data_dir"], 
        config_dict["processed_data_dir"], 
        current_seq_len, 
        force_reprocess=config_dict.get("force_reprocess_data", False)
    )
    train_dataloader, val_dataloader = data_processor.get_dataloaders(
        config_dict["batch_size"], 
        config_dict["val_split_ratio"],
        config_dict.get("num_workers", 0),
        current_seq_len # Pass current_sequence_length
    )
    
    vocab_size = data_processor.get_vocab_size() # Should be 256 for bytes
    if config_dict.get("vocab_size") != vocab_size:
        print(f"Warning: CONFIG vocab_size {config_dict.get('vocab_size')} differs from DataProcessor's {vocab_size}. Using DataProcessor's.")
        config_dict["vocab_size"] = vocab_size # Update config to match actual
    return train_dataloader, val_dataloader

def initialize_optimizer(model, optim_config):
    lr = optim_config.get("learning_rate", 1e-3) 
    optimizer_type = optim_config.get("optimizer_type", "AdamW").lower()

    # Filter out parameters that do not require gradients
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    if optimizer_type == "adamw":
        return optim.AdamW(
            trainable_params, 
            lr=lr, # This is the target LR; warmup will adjust it initially
            betas=(optim_config.get("adam_beta1", 0.9), optim_config.get("adam_beta2", 0.98)),
            eps=optim_config.get("adam_eps", 1e-9),
            weight_decay=optim_config.get("weight_decay", 0.01)
        )
    elif optimizer_type == "adam":
        return optim.Adam(
            trainable_params, 
            lr=lr, 
            betas=(optim_config.get("adam_beta1", 0.9), optim_config.get("adam_beta2", 0.999)), # Default Adam betas
            eps=optim_config.get("adam_eps", 1e-8) # Default Adam eps
            # Adam doesn't typically use weight_decay in the same way as AdamW.
            # If weight_decay is desired with Adam, it's often applied manually or handled by specific implementations.
        )
    else:
        raise ValueError(f"Unsupported optimizer_type: {optimizer_type}")

def initialize_scheduler(optimizer, scheduler_config, batches_per_epoch=None):
    scheduler_type = scheduler_config.get("scheduler_type")
    if not scheduler_type: return None # No scheduler

    if scheduler_type.lower() == "reducelronplateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', # Assumes minimizing validation loss
            factor=scheduler_config.get("lr_scheduler_factor", 0.1),
            patience=scheduler_config.get("lr_scheduler_patience", 10)
            # verbose parameter was removed/deprecated. Monitor LR changes through logging.
        )
    elif scheduler_type.lower() == "cosineannealinglr":
        # T_max is the number of iterations for one half of a cosine cycle.
        # Typically set to total number of training steps (num_epochs * batches_per_epoch)
        # Or a fraction if you want multiple cycles.
        
        # Use pre-calculated T_max from config if available (e.g., from main logic)
        T_max_config_key_main = "lr_scheduler_T_max_calculated_in_main" # Key used in main if calculated there
        T_max_config_key_direct = "lr_scheduler_T_max" # Direct config key

        if T_max_config_key_direct in scheduler_config and scheduler_config[T_max_config_key_direct] is not None:
            T_max = scheduler_config[T_max_config_key_direct]
            print(f"Using T_max from config: {T_max}")
        elif T_max_config_key_main in scheduler_config and scheduler_config[T_max_config_key_main] is not None:
             T_max = scheduler_config[T_max_config_key_main]
             print(f"Using T_max calculated in main: {T_max}")
        elif batches_per_epoch is not None and batches_per_epoch > 0:
            num_epochs_for_scheduler = scheduler_config.get("num_epochs", 50) # Total epochs
            T_max = num_epochs_for_scheduler * batches_per_epoch
            print(f"Calculated T_max for CosineAnnealingLR: {T_max} (Epochs: {num_epochs_for_scheduler}, Batches/Epoch: {batches_per_epoch})")
        else:
            # Fallback if T_max cannot be determined (e.g., batches_per_epoch is None or 0)
            # This might happen if dataloader is empty or not yet initialized fully.
            # A large default or an error might be appropriate.
            # Using num_epochs * a guess for batches_per_epoch (e.g., 1000) as a last resort.
            num_epochs_for_scheduler = scheduler_config.get("num_epochs", 50)
            fallback_batches_per_epoch = 1000 
            T_max = num_epochs_for_scheduler * fallback_batches_per_epoch
            print(f"Warning: batches_per_epoch not available for CosineAnnealingLR T_max. Using fallback T_max: {T_max}")
        
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(T_max), # Ensure T_max is an integer
            eta_min=scheduler_config.get("lr_scheduler_eta_min", 0) # Minimum LR
        )
    else:
        print(f"Unsupported scheduler_type: {scheduler_type}. No scheduler will be used.")
        return None

def initialize_training_components(config_dict_for_model, config_dict_for_optim_sched, device, batches_per_epoch_for_scheduler=None):
    # Model uses its specific config (could be from checkpoint or current global)
    model = ByteLLM_GrugV2(config_dict_for_model).to(device)
    
    # In initialize_training_components, after model is created:
    if hasattr(torch, 'compile'):
        print("Attempting to compile the model with torch.compile()...")
        # model = torch.compile(model, mode="reduce-overhead") # Good for smaller models/batch sizes
        # model = torch.compile(model, mode="max-autotune") # More aggressive, longer compile time initially
    
    # Optimizer and Scheduler use the current run's config for their parameters (LR, type, etc.)
    optimizer = initialize_optimizer(model, config_dict_for_optim_sched)
    scheduler = initialize_scheduler(optimizer, config_dict_for_optim_sched, batches_per_epoch_for_scheduler)
    
    criterion = nn.CrossEntropyLoss() # Standard for classification/next-token prediction
    return model, optimizer, criterion, scheduler

def perform_training(current_run_config, model, train_dataloader, val_dataloader, optimizer, criterion, scheduler, device):
    if not current_run_config.get("DO_TRAINING", True):
        print("\n--- Skipping Training Phase (DO_TRAINING set to False) ---")
        return

    print("\n--- GrugV2 Training Phase ---")
    # Trainer uses the current_run_config for its operational parameters (epochs, print_every, AMP settings, etc.)
    trainer = Trainer(
        model, train_dataloader, val_dataloader, optimizer, criterion, device, 
        current_run_config["checkpoint_dir"], current_run_config["model_name"], scheduler,
        train_config=current_run_config # Pass the complete current run's config to trainer
    )
    try:
        trainer.train(current_run_config["num_epochs"])
    except Exception as e:
        print(f"An error occurred during GrugV2 training: {e}")
        traceback.print_exc()

def perform_prediction_scenarios(current_run_config, device):
    if not current_run_config.get("DO_PREDICTION", True):
        print("\n--- Skipping Prediction Phase (DO_PREDICTION set to False) ---")
        return

    print("\n--- GrugV2 Prediction/Generation (using best model) ---")
    # Prediction should use the model name from the current run's config to find the corresponding best checkpoint
    best_ckpt_path = Path(current_run_config["checkpoint_dir"]) / f"{current_run_config['model_name']}_best.pth"
    
    if not best_ckpt_path.exists():
        print(f"No best model checkpoint ({best_ckpt_path}) found for model '{current_run_config['model_name']}'. Skipping prediction.")
        return

    try:
        print(f"Loading best GrugV2 model for prediction: {best_ckpt_path}")
        ckpt = torch.load(best_ckpt_path, map_location=device)
        
        # IMPORTANT: For prediction, the model should be instantiated with the CONFIGURATION SAVED IN THE CHECKPOINT.
        loaded_model_config_from_ckpt = ckpt.get('config')
        if not loaded_model_config_from_ckpt:
            print("ERROR: Checkpoint does not contain its configuration. Cannot reliably perform prediction.")
            print("Please ensure checkpoints are saved with their 'config'. Falling back to current run's config (HIGHLY RISKY).")
            loaded_model_config_from_ckpt = current_run_config # Risky fallback
        
        # Instantiate model for prediction using the checkpoint's config
        predictor_model = ByteLLM_GrugV2(loaded_model_config_from_ckpt).to(device)
        predictor_model.load_state_dict(ckpt['model_state_dict'])
        print("Best GrugV2 model weights loaded successfully for prediction.")

        # Use generation parameters from the *checkpoint's config* if available, otherwise current run's config
        generation_params_for_predictor = {
            "generation_temperature": loaded_model_config_from_ckpt.get("generation_temperature", current_run_config.get("generation_temperature", 1.0)),
            "generation_top_k": loaded_model_config_from_ckpt.get("generation_top_k", current_run_config.get("generation_top_k", 0))
        }
        # Model internal parameters (like max_len) must come from the checkpoint's config
        model_internals_for_predictor = {
            "max_positional_encoding_len": loaded_model_config_from_ckpt["max_positional_encoding_len"],
            "sequence_length": loaded_model_config_from_ckpt["sequence_length"], # The sequence length it was trained on
        }
        predictor = Predictor(predictor_model, device, generation_params_for_predictor, model_internals_for_predictor)

        seeds_to_try = {
            "Philosophical": "The meaning of life is",
            "Technical": "Mamba architecture is based on",
            "Narrative Start": "Once upon a time, in a land of bytes,",
            "Code Snippet": "import torch\nclass MyModel(torch.nn.Module):"
        }
        
        for seed_name, seed_text in seeds_to_try.items():
            seed_bytes = seed_text.encode('utf-8')
            print(f"\nSeed ({seed_name}): '{seed_text}' (Length: {len(seed_bytes)} bytes)")
            generated_bytes = predictor.generate_sequence(seed_bytes, length=150) # Generate 150 new bytes
            try:
                # The generated_bytes includes the seed.
                full_text = generated_bytes.decode('utf-8', errors='replace')
                # To show only newly generated part:
                # newly_generated_text = generated_bytes[len(seed_bytes):].decode('utf-8', errors='replace')
                # print(f"Newly Generated: {newly_generated_text}")
                print(f"Full Text (Seed + Generated):\n---\n{full_text}\n---")
            except UnicodeDecodeError as ude: 
                print(f"Could not decode generated sequence: {ude}. Raw bytes: {generated_bytes}")
    
    except Exception as e: 
        print(f"An error occurred during the GrugV2 prediction phase: {e}")
        traceback.print_exc()

# --- Main Orchestration ---
def main():
    global CONFIG # Allow CONFIG to be modified (e.g., vocab_size update)
    
    if not MAMBA_AVAILABLE:
        print("Critical: Mamba library is not installed. GrugV2 script cannot run effectively.")
        return

    # For performance profiling, disable anomaly detection if it's on and profiler is active
    if CONFIG.get("enable_profiler", False) and torch.is_anomaly_enabled():
        print("INFO: Disabling autograd anomaly detection for profiling run for accurate performance metrics.")
        # torch.set_anomaly_enabled(False) # PyTorch 2.x way (preferred if available)
        torch.autograd.set_detect_anomaly(False) # Older way, still works

    try:
        device = setup_environment(CONFIG)
        
        print("\n--- GrugV2 Data Loading and Processing ---")
        train_dataloader, val_dataloader = load_data_components(CONFIG)
        
        # Calculate batches_per_epoch for scheduler T_max if needed
        batches_per_epoch = None
        if train_dataloader and len(train_dataloader) > 0:
            batches_per_epoch = len(train_dataloader)
            # Update T_max in CONFIG if CosineAnnealingLR is used and T_max is not explicitly set
            if CONFIG.get("scheduler_type", "").lower() == "cosineannealinglr" and "lr_scheduler_T_max" not in CONFIG:
                calculated_T_max = CONFIG["num_epochs"] * batches_per_epoch
                CONFIG["lr_scheduler_T_max_calculated_in_main"] = calculated_T_max 
                print(f"Calculated T_max for CosineAnnealingLR in main: {calculated_T_max} (Epochs: {CONFIG['num_epochs']}, Batches/Epoch: {batches_per_epoch})")
        else: # train_dataloader is empty or None
            if CONFIG.get("scheduler_type", "").lower() == "cosineannealinglr" and "lr_scheduler_T_max" not in CONFIG:
                print("Warning: train_dataloader is empty. CosineAnnealingLR T_max might be misconfigured if not explicitly set in CONFIG. Scheduler init will use a fallback.")
        

        print("\n--- GrugV2 Model and Optimizer Initialization ---")
        # Model, Optimizer, Scheduler are initialized based on the CURRENT CONFIG.
        # If resuming, the Trainer's load_checkpoint method handles loading state_dicts.
        # Architectural mismatches between current CONFIG and checkpoint CONFIG are a concern.
        # A more robust resume would load checkpoint config first, then init model with it.
        # For simplicity here, we assume if resuming, the current CONFIG is compatible or user manages this.
        model, optimizer, criterion, scheduler = initialize_training_components(
            CONFIG, # Model uses current CONFIG
            CONFIG, # Optimizer/Scheduler use current CONFIG for their params
            device, 
            batches_per_epoch_for_scheduler=batches_per_epoch
        )
        
        # Training uses the current CONFIG for its operational parameters
        perform_training(CONFIG, model, train_dataloader, val_dataloader, optimizer, criterion, scheduler, device)
        
        # Prediction uses the current CONFIG to find the model checkpoint, but then loads the checkpoint's config for model instantiation.
        perform_prediction_scenarios(CONFIG, device)

    except ValueError as ve:
        print(f"Configuration or Value Error in GrugV2 main: {ve}")
        traceback.print_exc()
    except FileNotFoundError as fnfe:
        print(f"File Not Found Error in GrugV2 main: {fnfe}")
        traceback.print_exc()
    except ImportError as ie:
        print(f"Import Error (likely mamba-ssm or a dependency not installed): {ie}")
        traceback.print_exc()
    except RuntimeError as rte: # Catch PyTorch runtime errors, often CUDA related
        print(f"PyTorch Runtime Error in GrugV2 main: {rte}")
        if "CUDA out of memory" in str(rte):
            print("Hint: This is a CUDA Out of Memory error. Try reducing batch_size, sequence_length, or model dimensions (embedding_dim, mamba_d_model, cnn_out_channels_list). Using AMP (use_amp=True) might also help reduce memory.")
        traceback.print_exc()
    except Exception as e:
        print(f"An unexpected critical error occurred in GrugV2 main execution: {e}")
        traceback.print_exc()
    finally:
        print("\nGrugV2 script finished.")

if __name__ == "__main__":
    main()
