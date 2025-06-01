# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming model_components.py is in the same directory or accessible in Python path
from model_components import LearnablePositionalEncoding, CNNFrontend
# Assuming config.py is in the same directory or accessible
from config import CONFIG_V3 # Used for default values in __main__ test

class ByteLLM_GrugV3(nn.Module):
    """
    The main ByteLLM_GrugV3 model architecture, combining an embedding layer,
    an optional CNN frontend, positional encoding, Transformer encoder blocks,
    and an output layer.
    """
    def __init__(self, model_config: dict):
        """
        Initializes the ByteLLM_GrugV3 model.

        Args:
            model_config (dict): A dictionary containing the model's hyperparameters.
                                 Expected keys include:
                                 - "vocab_size": Size of the vocabulary (256 for bytes).
                                 - "embedding_dim": Dimensionality of the byte embeddings.
                                 - "use_cnn_frontend": Boolean, whether to use the CNN frontend.
                                 - "cnn_out_channels_list", "cnn_kernel_sizes", etc. (if use_cnn_frontend is True)
                                 - "attention_d_model": Dimensionality of the model for attention layers.
                                 - "max_positional_encoding_len": Max length for positional encodings.
                                 - "pe_dropout": Dropout for positional encoding.
                                 - "num_attention_layers": Number of Transformer encoder layers.
                                 - "attention_num_heads": Number of heads in MultiHeadAttention.
                                 - "ffn_dim_multiply": Multiplier for FFN hidden dim in Transformer.
                                 - "transformer_dropout": Dropout in Transformer layers.
                                 - "output_dropout": Dropout before the final output layer.
        """
        super().__init__()
        self.config = model_config.copy() # Store a copy of the config

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
                in_channels=current_processing_dim, # Output of embedding
                out_channels_list=self.config["cnn_out_channels_list"],
                kernel_sizes=self.config["cnn_kernel_sizes"],
                stride=self.config.get("cnn_stride", 1),
                cnn_dropout=self.config.get("cnn_dropout", 0.1),
                activation=self.config.get("cnn_activation", "GELU"),
                use_layernorm=self.config.get("cnn_use_layernorm", True),
                padding_mode=self.config.get("cnn_padding_mode", "zeros")
            )
            # The output dimension of the CNN frontend becomes the input to the next stage
            current_processing_dim = self.config["cnn_out_channels_list"][-1]
        else:
            self.cnn_frontend = None

        # --- Adjust attention_d_model if necessary ---
        # The d_model for the transformer part must match the output dimension of the preceding layer (CNN or embedding)
        if self.config["attention_d_model"] != current_processing_dim:
            print(f"Info: Model's attention_d_model in config was {self.config['attention_d_model']}. "
                  f"Adjusting to {current_processing_dim} to match the preceding layer's output dimension.")
            self.config["attention_d_model"] = current_processing_dim
        
        d_model = self.config["attention_d_model"] # This is the d_model for the Transformer part

        # --- 3. Positional Encoding ---
        self.positional_encoder = LearnablePositionalEncoding(
            d_model=d_model, # Must match the dimension of the data entering the transformer
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
                dropout=self.config.get("transformer_dropout", 0.1), # This is dropout inside TransformerEncoderLayer
                activation=F.gelu, # Common choice, can be configured
                batch_first=True,  # Crucial: input format is (batch, seq, feature)
                norm_first=False   # Standard Transformer uses LayerNorm after Add&Norm
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
        """
        Performs the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of byte indices, shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: Output logits, shape (batch_size, vocab_size).
                          Predicts the next byte based on the *last* hidden state of the sequence.
        """
        # x: (B, L) where L is sequence_length
        
        x = self.embedding(x)  # x: (B, L, EmbDim)

        if self.cnn_frontend:
            x = self.cnn_frontend(x) # x: (B, L_new, CNN_out_dim)
        
        # After CNN (if any), x has shape (B, L_processed, current_processing_dim)
        # This current_processing_dim should now be equal to self.config["attention_d_model"]
        x = self.positional_encoder(x) # x: (B, L_processed, d_model)

        # Pass through Transformer encoder layers
        # No explicit source mask needed for nn.TransformerEncoderLayer if we process the whole sequence
        # and only use the last output. If causal masking is needed for training (e.g. autoregressive on all positions),
        # a mask would be required here. For this setup, we predict one next token from the whole input sequence.
        for layer in self.transformer_encoder_layers:
            x = layer(x) # x: (B, L_processed, d_model)
        
        x = self.final_norm(x) # x: (B, L_processed, d_model)

        # We typically use the representation of the *last token* in the sequence for next-token prediction.
        # If CNN stride reduced sequence length, L_processed might be different from original L.
        output_representation = x[:, -1, :]  # x: (B, d_model) - taking the last hidden state
        
        output_representation = self.output_dropout(output_representation)
        logits = self.fc_out(output_representation)  # logits: (B, vocab_size)

        return logits

if __name__ == '__main__':
    print("--- Testing model.py ---")

    # Use a subset of CONFIG_V3 for testing, or the full one
    # Ensure all necessary keys for ByteLLM_GrugV3 are present
    test_model_config = CONFIG_V3.copy() # Start with the global config

    # Modify for a quicker test if needed, e.g., smaller dimensions/layers
    # test_model_config["embedding_dim"] = 64
    # test_model_config["attention_d_model"] = 64 # Will be adjusted if CNN is used
    # test_model_config["num_attention_layers"] = 2
    # test_model_config["attention_num_heads"] = 4
    # if test_model_config.get("use_cnn_frontend"):
    #     test_model_config["cnn_out_channels_list"] = [64, 64] # Match attention_d_model

    print(f"Using model config for test: {test_model_config}")

    try:
        model = ByteLLM_GrugV3(test_model_config)
        print("\nModel Instantiation Successful.")
        print(model)

        # Prepare a dummy input batch
        batch_size = 4
        # Use sequence_length from the config used by the model
        sequence_length = test_model_config["sequence_length"] 
        vocab_size = test_model_config["vocab_size"]

        dummy_input = torch.randint(0, vocab_size, (batch_size, sequence_length), dtype=torch.long)
        print(f"\nDummy input shape: {dummy_input.shape}")

        # Perform a forward pass
        print("Performing forward pass...")
        with torch.no_grad(): # No need to track gradients for a simple forward pass test
            logits_output = model(dummy_input)
        
        print(f"Logits output shape: {logits_output.shape}")
        
        # Expected output shape: (batch_size, vocab_size)
        assert logits_output.shape == (batch_size, vocab_size), \
            f"Output shape mismatch. Expected {(batch_size, vocab_size)}, got {logits_output.shape}"
        print("Forward pass successful and output shape is correct.")

        # Test with CNN disabled if it was enabled, and vice-versa (optional)
        print("\n--- Testing with CNN frontend explicitly disabled ---")
        config_no_cnn = test_model_config.copy()
        config_no_cnn["use_cnn_frontend"] = False
        # If CNN was used, attention_d_model might need to revert to embedding_dim
        config_no_cnn["attention_d_model"] = config_no_cnn["embedding_dim"] 
        
        model_no_cnn = ByteLLM_GrugV3(config_no_cnn)
        print("Model (no CNN) Instantiation Successful.")
        with torch.no_grad():
            logits_no_cnn = model_no_cnn(dummy_input)
        print(f"Logits output shape (no CNN): {logits_no_cnn.shape}")
        assert logits_no_cnn.shape == (batch_size, vocab_size), "Output shape mismatch (no CNN)."
        print("Forward pass (no CNN) successful.")


        if not test_model_config.get("use_cnn_frontend"): # If default was no CNN, test with CNN
            print("\n--- Testing with CNN frontend explicitly enabled (if not default) ---")
            config_with_cnn = test_model_config.copy()
            config_with_cnn["use_cnn_frontend"] = True
            # Ensure CNN params are sensible if they weren't in the original test_model_config
            if "cnn_out_channels_list" not in config_with_cnn:
                 config_with_cnn["cnn_out_channels_list"] = [config_with_cnn["embedding_dim"], config_with_cnn["embedding_dim"] // 2]
                 config_with_cnn["cnn_kernel_sizes"] = [3,3]
            # attention_d_model should match the last cnn_out_channel
            config_with_cnn["attention_d_model"] = config_with_cnn["cnn_out_channels_list"][-1]

            model_with_cnn = ByteLLM_GrugV3(config_with_cnn)
            print("Model (with CNN) Instantiation Successful.")
            with torch.no_grad():
                logits_with_cnn = model_with_cnn(dummy_input)
            print(f"Logits output shape (with CNN): {logits_with_cnn.shape}")
            assert logits_with_cnn.shape == (batch_size, vocab_size), "Output shape mismatch (with CNN)."
            print("Forward pass (with CNN) successful.")


    except Exception as e:
        print(f"\nAn error occurred during model.py testing:")
        import traceback
        traceback.print_exc()

    print("\n--- model.py tests completed (check for assertions) ---")
