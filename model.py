# model.py (Fully Modified)

import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. MODIFY THIS IMPORT
from model_components import SinusoidalPositionalEncoding, CNNFrontend, CustomTransformerEncoderLayer
from config import CONFIG_V3

class ByteLLM_GrugV3(nn.Module):
    """
    The main ByteLLM_GrugV3 model architecture.
    This class now supports two modes controlled by 'use_parallel_stream_model' in the config:
    1. A standard single-stream Transformer (original architecture).
    2. A parallel-stream model with a byte-level path and a CNN-aggregated path.
    Both architectures can now use Curved FFNs via CustomTransformerEncoderLayer.
    """
    def __init__(self, model_config: dict):
        super().__init__()
        self.config = model_config.copy()

        # --- 1. Common Embedding Layer ---
        self.embedding = nn.Embedding(
            self.config["vocab_size"],
            self.config["embedding_dim"]
        )

        # Optional CNN Frontend
        if self.config.get("use_cnn_frontend", False):
            self.cnn_frontend = CNNFrontend(
                in_channels=self.config["embedding_dim"],
                out_channels_list=self.config["cnn_out_channels_list"],
                kernel_sizes=self.config["cnn_kernel_sizes"],
                stride=self.config.get("cnn_stride", 1),
                cnn_dropout=self.config.get("cnn_dropout", 0.1),
                activation=self.config.get("cnn_activation", "GELU"),
                use_layernorm=self.config.get("cnn_use_layernorm", True),
                padding_mode=self.config.get("cnn_padding_mode", "zeros")
            )
        else:
            self.cnn_frontend = None

        # --- ARCHITECTURE-SPECIFIC INITIALIZATION ---
        if self.config.get("use_parallel_stream_model", False):
            self._init_parallel_stream()
        else:
            self._init_single_stream()

        total_params = sum(p.numel() for p in self.parameters())
        print(f"ByteLLM_GrugV3 Initialized. Total Parameters: {total_params:,}")

    def _init_single_stream(self):
        """Initializes the single-stream Transformer architecture."""
        print("Initializing Single-Stream Architecture.")
        d_model = self.config["attention_d_model"]

        if self.config["embedding_dim"] != d_model:
            self.input_projection = nn.Linear(self.config["embedding_dim"], d_model)
        else:
            self.input_projection = nn.Identity()

        self.positional_encoder = SinusoidalPositionalEncoding(
            d_model=d_model,
            dropout=self.config["pe_dropout"],
            max_len=self.config["max_positional_encoding_len"]
        )

        # 2. REPLACE THE TRANSFORMER LAYER INITIALIZATION HERE
        self.transformer_encoder_layers = nn.ModuleList([
            CustomTransformerEncoderLayer(
                d_model=d_model,
                nhead=self.config["attention_num_heads"],
                dim_feedforward=d_model * self.config.get("ffn_dim_multiply", 4),
                dropout=self.config.get("transformer_dropout", 0.1),
                activation=F.gelu,
                batch_first=True,
                norm_first=self.config.get("transformer_norm_first", False),
                use_curved_ffn=self.config.get("use_curved_ffn", False),
                ffn_gamma=self.config.get("ffn_gamma", 0.0)
            ) for _ in range(self.config["num_attention_layers"])
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.output_dropout = nn.Dropout(self.config.get("output_dropout", 0.1))
        self.fc_out = nn.Linear(d_model, self.config["vocab_size"])

    def _init_parallel_stream(self):
        """Initializes the parallel-stream architecture."""
        print("Initializing Parallel-Stream Architecture.")
        embedding_dim = self.config["embedding_dim"]
        d_model = self.config["attention_d_model"]
        pe_dropout = self.config.get("pe_dropout", 0.1)
        max_len = self.config.get("max_positional_encoding_len", 4096)
        transformer_dropout = self.config.get("transformer_dropout", 0.1)
        norm_first = self.config.get("transformer_norm_first", False)
        ffn_multiply = self.config.get("ffn_dim_multiply", 4)
        n_heads = self.config["attention_num_heads"]

        # --- Path 1: Byte Stream Components ---
        self.byte_stream_projection = nn.Linear(embedding_dim, d_model)
        self.byte_positional_encoder = SinusoidalPositionalEncoding(d_model, pe_dropout, max_len)

        # 3. REPLACE THE TRANSFORMER LAYER INITIALIZATION HERE
        self.byte_stream_encoder = nn.ModuleList([
            CustomTransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model * ffn_multiply,
                dropout=transformer_dropout, activation=F.gelu, batch_first=True, norm_first=norm_first,
                use_curved_ffn=self.config.get("use_curved_ffn", False),
                ffn_gamma=self.config.get("ffn_gamma", 0.0)
            ) for _ in range(self.config["num_byte_stream_layers"])
        ])

        # --- Path 2: Aggregated Stream Components ---
        agg_cnn_out_dim = self.config["agg_cnn_out_dim"]
        self.aggregator_cnn = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=agg_cnn_out_dim,
            kernel_size=self.config["agg_cnn_kernel_size"],
            stride=self.config["agg_cnn_stride"]
        )
        agg_max_len = max_len // self.config["agg_cnn_stride"]
        self.agg_positional_encoder = SinusoidalPositionalEncoding(agg_cnn_out_dim, pe_dropout, agg_max_len)

        # 4. REPLACE THE TRANSFORMER LAYER INITIALIZATION HERE
        self.agg_stream_encoder = nn.ModuleList([
            CustomTransformerEncoderLayer(
                d_model=agg_cnn_out_dim, nhead=n_heads, dim_feedforward=agg_cnn_out_dim * ffn_multiply,
                dropout=transformer_dropout, activation=F.gelu, batch_first=True, norm_first=norm_first,
                use_curved_ffn=self.config.get("use_curved_ffn", False),
                ffn_gamma=self.config.get("ffn_gamma", 0.0)
            ) for _ in range(self.config["num_agg_stream_layers"])
        ])

        # --- Final Combination Layers ---
        combined_input_dim = d_model + agg_cnn_out_dim
        self.final_norm = nn.LayerNorm(combined_input_dim)
        self.output_dropout = nn.Dropout(self.config.get("output_dropout", 0.1))
        self.fc_out = nn.Linear(combined_input_dim, self.config["vocab_size"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Main forward pass, dispatches to the correct architecture."""
        if self.config.get("use_parallel_stream_model", False):
            return self._forward_parallel_stream(x)
        else:
            return self._forward_single_stream(x)

    def _forward_single_stream(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the single-stream model."""
        x = self.embedding(x)
        if self.cnn_frontend:
            x = self.cnn_frontend(x)
        
        x = self.input_projection(x)

        x = self.positional_encoder(x)
        for layer in self.transformer_encoder_layers:
            x = layer(x)
        x = self.final_norm(x)
        output_representation = x[:, -1, :]
        output_representation = self.output_dropout(output_representation)
        logits = self.fc_out(output_representation)
        return logits

    def _forward_parallel_stream(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the parallel-stream model."""
        x_embed = self.embedding(x)
        if self.cnn_frontend:
            x_embed = self.cnn_frontend(x_embed)

        # --- Path 1: Fine-grained Byte Stream ---
        byte_path_projected = self.byte_stream_projection(x_embed)
        byte_path_input = self.byte_positional_encoder(byte_path_projected)
        
        byte_path_output = byte_path_input
        for layer in self.byte_stream_encoder:
            byte_path_output = layer(byte_path_output)
        final_byte_representation = byte_path_output[:, -1, :]

        # --- Path 2: Coarse-grained Aggregated Stream ---
        agg_input_cnn = x_embed.permute(0, 2, 1)
        agg_output_cnn = self.aggregator_cnn(agg_input_cnn)
        agg_output_cnn = agg_output_cnn.permute(0, 2, 1)

        agg_path_input = self.agg_positional_encoder(agg_output_cnn)
        agg_path_output = agg_path_input
        for layer in self.agg_stream_encoder:
            agg_path_output = layer(agg_path_output)
        final_agg_representation = agg_path_output[:, -1, :]

        # 3. Combine, Normalize, and Predict
        combined_representation = torch.cat([final_byte_representation, final_agg_representation], dim=-1)
        normed_representation = self.final_norm(combined_representation)
        dropped_representation = self.output_dropout(normed_representation)
        logits = self.fc_out(dropped_representation)

        return logits

if __name__ == '__main__':
    # ... (The __main__ block for testing remains unchanged) ...
    print("--- Testing model.py ---")
    test_config = CONFIG_V3.copy()
    batch_size = 2
    vocab_size = test_config["vocab_size"]

    # --- Test 1: Original Single-Stream Model ---
    print("\n--- Testing Single-Stream Architecture ---")
    single_stream_config = test_config.copy()
    single_stream_config["use_parallel_stream_model"] = False
    try:
        model = ByteLLM_GrugV3(single_stream_config)
        sequence_length = single_stream_config["sequence_length"]
        dummy_input = torch.randint(0, vocab_size, (batch_size, sequence_length), dtype=torch.long)
        with torch.no_grad():
            logits_output = model(dummy_input)
        assert logits_output.shape == (batch_size, vocab_size), f"Output shape mismatch. Expected {(batch_size, vocab_size)}, got {logits_output.shape}"
        print("Single-stream forward pass successful and output shape is correct.")
    except Exception as e:
        import traceback
        traceback.print_exc()

    # --- Test 2: New Parallel-Stream Model ---
    print("\n--- Testing Parallel-Stream Architecture ---")
    parallel_stream_config = test_config.copy()
    parallel_stream_config["use_parallel_stream_model"] = True
    parallel_stream_config.setdefault("num_byte_stream_layers", 4)
    parallel_stream_config.setdefault("num_agg_stream_layers", 4)
    parallel_stream_config.setdefault("agg_cnn_kernel_size", 4)
    parallel_stream_config.setdefault("agg_cnn_stride", 4)
    parallel_stream_config.setdefault("agg_cnn_out_dim", parallel_stream_config["attention_d_model"])
    try:
        model_parallel = ByteLLM_GrugV3(parallel_stream_config)
        sequence_length = parallel_stream_config["sequence_length"]
        dummy_input = torch.randint(0, vocab_size, (batch_size, sequence_length), dtype=torch.long)
        with torch.no_grad():
            logits_output_parallel = model_parallel(dummy_input)
        assert logits_output_parallel.shape == (batch_size, vocab_size), f"Output shape mismatch. Expected {(batch_size, vocab_size)}, got {logits_output_parallel.shape}"
        print("Parallel-stream forward pass successful and output shape is correct.")
    except Exception as e:
        import traceback
        traceback.print_exc()

    print("\n--- model.py all tests completed ---")