import torch
import torch.nn as nn
import torch.nn.functional as F

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

# --- Model Architecture Components (Partially from GrugV2) ---

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

    def forward(self, x: torch.Tensor):
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
