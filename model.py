# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_components import SinusoidalPositionalEncoding, LearnablePositionalEncoding, CNNFrontend
from config import CONFIG_V3

class ByteLLM_GrugV3(nn.Module):
    """
    The main ByteLLM_GrugV3 model architecture, combining an embedding layer,
    an optional CNN frontend, positional encoding, Transformer encoder blocks,
    and an output layer.
    """
    def __init__(self, model_config: dict):
        super().__init__()
        self.config = model_config.copy()

        # --- 1. Embedding Layer ---
        self.embedding = nn.Embedding(
            self.config["vocab_size"],
            self.config["embedding_dim"]
        )
        current_processing_dim = self.config["embedding_dim"]

        # --- 2. Optional CNN Frontend ---
        if self.config.get("use_cnn_frontend", False):
            if not all(k in self.config for k in ["cnn_out_channels_list", "cnn_kernel_sizes"]):
                raise ValueError("CNN frontend is enabled but required configs "
                                 "('cnn_out_channels_list', 'cnn_kernel_sizes') are missing.")
            self.cnn_frontend = CNNFrontend(
                in_channels=current_processing_dim,
                out_channels_list=self.config["cnn_out_channels_list"],
                kernel_sizes=self.config["cnn_kernel_sizes"],
                stride=self.config.get("cnn_stride", 1),
                cnn_dropout=self.config.get("cnn_dropout", 0.1),
                activation=self.config.get("cnn_activation", "GELU"),
                use_layernorm=self.config.get("cnn_use_layernorm", True),
                padding_mode=self.config.get("cnn_padding_mode", "zeros")
            )
            current_processing_dim = self.config["cnn_out_channels_list"][-1]
        else:
            self.cnn_frontend = None

        # --- Adjust attention_d_model if necessary ---
        if self.config["attention_d_model"] != current_processing_dim:
            print(f"Info: Model's attention_d_model in config was {self.config['attention_d_model']}. "
                  f"Adjusting to {current_processing_dim} to match the preceding layer's output dimension.")
            self.config["attention_d_model"] = current_processing_dim

        d_model = self.config["attention_d_model"]

        # --- 3. Positional Encoding ---
        self.positional_encoder = SinusoidalPositionalEncoding(
            d_model=d_model,
            dropout=self.config.get("pe_dropout", 0.1),
            max_len=self.config["max_positional_encoding_len"]
        )

        # --- 4. Transformer Encoder Layers ---
        self.transformer_encoder_layers = nn.ModuleList()
        num_layers = self.config.get("num_attention_layers", 4)
        if num_layers <= 0:
            raise ValueError(f"num_attention_layers must be positive, got {num_layers}")

        for _ in range(num_layers):
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=self.config["attention_num_heads"],
                dim_feedforward=d_model * self.config.get("ffn_dim_multiply", 4),
                dropout=self.config.get("transformer_dropout", 0.1),
                activation=F.gelu,
                batch_first=True,
                norm_first=self.config.get("transformer_norm_first", False)
            )
            self.transformer_encoder_layers.append(encoder_layer)

        # --- 5. Final Layer Normalization ---
        self.final_norm = nn.LayerNorm(d_model)

        # --- 6. Output Layer ---
        self.output_dropout = nn.Dropout(self.config.get("output_dropout", 0.1))
        self.fc_out = nn.Linear(d_model, self.config["vocab_size"])

        print(f"ByteLLM_GrugV3 Initialized. Effective d_model for Transformer: {d_model}. "
              f"Vocab Size: {self.config['vocab_size']}.")
        if self.cnn_frontend:
            print("CNN Frontend is ENABLED.")
        else:
            print("CNN Frontend is DISABLED.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        if self.cnn_frontend:
            x = self.cnn_frontend(x)
        x = self.positional_encoder(x)
        for layer in self.transformer_encoder_layers:
            x = layer(x)
        x = self.final_norm(x)
        output_representation = x[:, -1, :]
        output_representation = self.output_dropout(output_representation)
        logits = self.fc_out(output_representation)
        return logits

if __name__ == '__main__':
    print("--- Testing model.py ---")

    default_test_config = CONFIG_V3.copy()
    print(f"Using base model config for tests: {default_test_config}")

    # --- Standard Model Test (respecting config's use_cnn_frontend) ---
    print("\n--- Standard Model Test ---")
    try:
        print(f"Instantiating model with use_cnn_frontend: {default_test_config.get('use_cnn_frontend', False)}")
        model = ByteLLM_GrugV3(default_test_config)
        print("Model Instantiation Successful.")

        batch_size = 2 # Smaller batch for tests
        # Use sequence_length from the config for this dummy input
        sequence_length = default_test_config["sequence_length"]
        vocab_size = default_test_config["vocab_size"]

        dummy_input = torch.randint(0, vocab_size, (batch_size, sequence_length), dtype=torch.long)
        print(f"Dummy input shape: {dummy_input.shape}")

        with torch.no_grad():
            logits_output = model(dummy_input)
        print(f"Logits output shape: {logits_output.shape}")
        assert logits_output.shape == (batch_size, vocab_size), \
            f"Output shape mismatch. Expected {(batch_size, vocab_size)}, got {logits_output.shape}"
        print("Forward pass successful and output shape is correct.")

    except Exception as e:
        print(f"\nAn error occurred during standard model testing:")
        import traceback
        traceback.print_exc()
    print("\n--- Standard Model Test completed ---")

    # --- Test with CNN explicitly disabled (if not already disabled by default) ---
    if default_test_config.get("use_cnn_frontend", False): # Only run if default had CNN enabled
        print("\n--- Testing with CNN frontend explicitly disabled ---")
        config_no_cnn = default_test_config.copy()
        config_no_cnn["use_cnn_frontend"] = False
        # Adjust attention_d_model to embedding_dim as CNN is off
        config_no_cnn["attention_d_model"] = config_no_cnn["embedding_dim"]

        try:
            model_no_cnn = ByteLLM_GrugV3(config_no_cnn)
            print("Model (CNN disabled) Instantiation Successful.")
            # Recreate dummy_input if sequence_length might change
            sequence_length_no_cnn = config_no_cnn["sequence_length"]
            dummy_input_no_cnn = torch.randint(0, vocab_size, (batch_size, sequence_length_no_cnn), dtype=torch.long)
            with torch.no_grad():
                logits_no_cnn = model_no_cnn(dummy_input_no_cnn)
            print(f"Logits output shape (CNN disabled): {logits_no_cnn.shape}")
            assert logits_no_cnn.shape == (batch_size, vocab_size), "Output shape mismatch (CNN disabled)."
            print("Forward pass (CNN disabled) successful.")
        except Exception as e:
            print(f"\nAn error occurred during CNN disabled test:")
            traceback.print_exc()
        print("\n--- CNN disabled test completed ---")

    # --- Test with CNN explicitly enabled (if not already enabled by default) ---
    if not default_test_config.get("use_cnn_frontend", True): # Only run if default had CNN disabled
        print("\n--- Testing with CNN frontend explicitly enabled ---")
        config_with_cnn = default_test_config.copy()
        config_with_cnn["use_cnn_frontend"] = True
        # Ensure CNN params are sensible if they weren't in the original test_model_config
        if "cnn_out_channels_list" not in config_with_cnn or not config_with_cnn["cnn_out_channels_list"]:
             config_with_cnn["cnn_out_channels_list"] = [config_with_cnn["embedding_dim"], config_with_cnn["embedding_dim"] // 2]
             config_with_cnn["cnn_kernel_sizes"] = [3,3] # Ensure kernels match channels list
        # attention_d_model should match the last cnn_out_channel (handled by __init__)

        try:
            model_with_cnn = ByteLLM_GrugV3(config_with_cnn)
            print("Model (CNN enabled) Instantiation Successful.")
            sequence_length_cnn = config_with_cnn["sequence_length"]
            dummy_input_cnn = torch.randint(0, vocab_size, (batch_size, sequence_length_cnn), dtype=torch.long)
            with torch.no_grad():
                logits_with_cnn = model_with_cnn(dummy_input_cnn)
            print(f"Logits output shape (CNN enabled): {logits_with_cnn.shape}")
            assert logits_with_cnn.shape == (batch_size, vocab_size), "Output shape mismatch (CNN enabled)."
            print("Forward pass (CNN enabled) successful.")
        except Exception as e:
            print(f"\nAn error occurred during CNN enabled test:")
            traceback.print_exc()
        print("\n--- CNN enabled test completed ---")

    # --- SinusoidalPositionalEncoding Extrapolation Tests (Looping for CNN ON and OFF) ---
    print("\n--- Testing SinusoidalPositionalEncoding Extrapolation (CNN ON/OFF) ---")
    initial_max_pos_len = 32 # Keep it small for faster test
    extrap_len = initial_max_pos_len * 2
    test_batch_size_extrap = 2

    for cnn_enabled_case in [False, True]:
        print(f"\n--- Extrapolation Test - CNN Frontend: {'ENABLED' if cnn_enabled_case else 'DISABLED'} ---")
        extrap_test_config = CONFIG_V3.copy() # Start fresh from defaults
        extrap_test_config["max_positional_encoding_len"] = initial_max_pos_len
        extrap_test_config["use_cnn_frontend"] = cnn_enabled_case
        vocab_size_extrap = extrap_test_config["vocab_size"]

        if cnn_enabled_case:
            # Ensure CNN parameters are valid and set attention_d_model accordingly
            # If not in default config, add some minimal valid ones
            if not extrap_test_config.get("cnn_out_channels_list"):
                extrap_test_config["cnn_out_channels_list"] = [extrap_test_config["embedding_dim"], extrap_test_config["embedding_dim"] // 2]
                extrap_test_config["cnn_kernel_sizes"] = [3, 3] # Must match length of out_channels_list
            # attention_d_model will be set by __init__ based on CNN output
            print(f"Configuring for CNN ENABLED: cnn_out_channels_list={extrap_test_config['cnn_out_channels_list']}, kernels={extrap_test_config['cnn_kernel_sizes']}")
        else:
            # CNN is disabled, attention_d_model should match embedding_dim
            extrap_test_config["attention_d_model"] = extrap_test_config["embedding_dim"]
            print("Configuring for CNN DISABLED.")

        print(f"Instantiating model with max_positional_encoding_len = {initial_max_pos_len}.")
        model_for_extrapolation = ByteLLM_GrugV3(extrap_test_config)
        print(f"Model for extrapolation (CNN {'ENABLED' if cnn_enabled_case else 'DISABLED'}) instantiated.")

        # Test 1: Sequence length equals initial max_len
        print(f"1. Testing seq_len ({initial_max_pos_len}) equal to initial max_len.")
        dummy_input_normal = torch.randint(0, vocab_size_extrap, (test_batch_size_extrap, initial_max_pos_len), dtype=torch.long)
        try:
            with torch.no_grad():
                logits_normal = model_for_extrapolation(dummy_input_normal)
            assert logits_normal.shape == (test_batch_size_extrap, vocab_size_extrap)
            print("  Forward pass (normal len) successful.")
            assert isinstance(model_for_extrapolation.positional_encoder, SinusoidalPositionalEncoding)
            pe_buffer_len = model_for_extrapolation.positional_encoder.pe.size(0)
            assert pe_buffer_len == initial_max_pos_len, f"PE buffer len {pe_buffer_len} != initial {initial_max_pos_len}"
            print(f"  PE buffer length ({pe_buffer_len}) correct.")
        except Exception as e:
            print(f"  Error (normal len): {e}")
            traceback.print_exc()

        # Test 2: Sequence length greater than initial max_len (extrapolation)
        print(f"2. Testing seq_len ({extrap_len}) greater than initial max_len ({initial_max_pos_len}).")
        dummy_input_extrap = torch.randint(0, vocab_size_extrap, (test_batch_size_extrap, extrap_len), dtype=torch.long)
        try:
            with torch.no_grad():
                logits_extrap = model_for_extrapolation(dummy_input_extrap)
            assert logits_extrap.shape == (test_batch_size_extrap, vocab_size_extrap)
            print("  Forward pass (extrap len) successful.")
            assert isinstance(model_for_extrapolation.positional_encoder, SinusoidalPositionalEncoding)
            pe_buffer_len_extrap = model_for_extrapolation.positional_encoder.pe.size(0)
            assert pe_buffer_len_extrap == extrap_len, f"PE buffer len {pe_buffer_len_extrap} != extrap {extrap_len}"
            print(f"  PE buffer successfully extended to {pe_buffer_len_extrap}.")
        except Exception as e:
            print(f"  Error (extrap len): {e}")
            traceback.print_exc()

    print("\n--- model.py all tests completed ---")
