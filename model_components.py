# model_components.py

import torch
import torch.nn as nn
import torch.nn.functional as F # Required for F.gelu if used, though not directly in these classes

class LearnablePositionalEncoding(nn.Module):
    """
    Learnable Positional Encoding module.
    Instead of fixed sinusoidal positional encodings, this module uses an Embedding layer
    to learn positional representations.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        """
        Initializes the LearnablePositionalEncoding module.

        Args:
            d_model (int): The dimensionality of the model's embeddings.
            dropout (float): Dropout probability.
            max_len (int): The maximum length of the input sequences.
        """
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if not (0 <= dropout <= 1):
            raise ValueError(f"dropout must be between 0 and 1, got {dropout}")
        if max_len <= 0:
            raise ValueError(f"max_len must be positive, got {max_len}")

        self.dropout = nn.Dropout(p=dropout)
        # Create an embedding layer for positions. The size is max_len, meaning it can
        # learn an embedding for each position up to max_len.
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
        # register_buffer ensures 'position_ids' is part of the model's state_dict,
        # is moved to the correct device with .to(device), but is not a learnable parameter.
        # It creates a tensor of position indices [0, 1, ..., max_len-1].
        self.register_buffer("position_ids", torch.arange(max_len).expand((1, -1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encodings to the input tensor.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, d_model).

        Returns:
            torch.Tensor: The input tensor with added positional encodings,
                          of the same shape as input x.
        """
        seq_len = x.size(1) # Get the actual sequence length from the input
        
        if seq_len > self.pos_embedding.num_embeddings:
            raise ValueError(
                f"Input sequence length ({seq_len}) exceeds max_len "
                f"({self.pos_embedding.num_embeddings}) for positional embeddings. "
                f"Increase max_len in config or ensure input sequences are truncated."
            )
        
        # Get the positional embeddings for the current sequence length
        # self.position_ids is (1, max_len), so self.position_ids[:, :seq_len] is (1, seq_len)
        pos_enc = self.pos_embedding(self.position_ids[:, :seq_len]) # Shape: (1, seq_len, d_model)
        
        # Add positional encoding to the input tensor.
        # Broadcasting will make pos_enc match the batch size of x.
        x = x + pos_enc 
        return self.dropout(x)

class CNNFrontend(nn.Module):
    """
    A 1D CNN frontend for processing sequences before a Transformer.
    This can help in capturing local features and reducing sequence length if stride > 1.
    """
    def __init__(self, in_channels: int, out_channels_list: list[int], kernel_sizes: list[int],
                 stride: int = 1, cnn_dropout: float = 0.1, activation: str = "GELU",
                 use_layernorm: bool = True, padding_mode: str = "zeros"):
        """
        Initializes the CNNFrontend module.

        Args:
            in_channels (int): Number of input channels (typically embedding dimension).
            out_channels_list (list[int]): List of output channels for each CNN layer.
            kernel_sizes (list[int]): List of kernel sizes for each CNN layer.
            stride (int): Stride for the convolutions.
            cnn_dropout (float): Dropout probability after each CNN layer.
            activation (str): Activation function to use ("RELU" or "GELU").
            use_layernorm (bool): Whether to use LayerNorm after activation in each CNN block.
            padding_mode (str): Padding mode for Conv1d (e.g., "zeros", "reflect", "replicate", "circular").
                                'zeros' is common, 'causal' might be 'replicate' then slice.
                                For non-causal, 'same' padding is often desired.
        """
        super().__init__()
        if not (len(out_channels_list) == len(kernel_sizes)):
            raise ValueError("out_channels_list and kernel_sizes must have the same length.")
        if not out_channels_list:
            raise ValueError("out_channels_list cannot be empty.")
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")

        self.conv_layers = nn.ModuleList()
        current_in_channels = in_channels
        
        for i, (k_size, o_channels) in enumerate(zip(kernel_sizes, out_channels_list)):
            if k_size <= 0 or k_size % 2 == 0: # Kernels are usually odd for 'same' padding
                print(f"Warning: CNN kernel size {k_size} is not a positive odd number. Padding might behave unexpectedly.")

            # Calculate padding for 'same' output length, assuming stride=1.
            # If stride > 1, output length will change.
            # For a 1D convolution, padding P = (K - 1) / 2 gives same output length if S=1, D=1.
            padding = (k_size - 1) // 2 
            
            conv_layer = nn.Conv1d(
                in_channels=current_in_channels,
                out_channels=o_channels,
                kernel_size=k_size,
                stride=stride, # Stride will reduce sequence length if > 1
                padding=padding,
                padding_mode=padding_mode,
                bias=not use_layernorm # If using LayerNorm, bias in Conv can be redundant
            )
            self.conv_layers.append(conv_layer)
            current_in_channels = o_channels # Output of this layer is input to next

        if activation.upper() == "RELU": self.activation_fn = nn.ReLU()
        elif activation.upper() == "GELU": self.activation_fn = nn.GELU()
        else: raise ValueError(f"Unsupported activation: {activation}. Choose 'RELU' or 'GELU'.")

        self.dropout_fn = nn.Dropout(cnn_dropout)
        self.use_layernorm = use_layernorm
        if self.use_layernorm:
            # LayerNorm is applied on the feature dimension (channels after permuting)
            self.layer_norms = nn.ModuleList([nn.LayerNorm(ch) for ch in out_channels_list])
        else:
            self.layer_norms = None # Explicitly set to None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes the input through the CNN layers.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embedding_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, new_sequence_length, last_out_channels).
                          new_sequence_length depends on stride.
        """
        # Input x: (B, L, Emb)
        # Conv1D expects (B, Emb, L)
        x = x.permute(0, 2, 1) 
        
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x) # (B, C_out, L_out)
            x = self.activation_fn(x)
            
            if self.use_layernorm and self.layer_norms:
                # LayerNorm expects (B, ..., Features) or (Features) for elementwise_affine=False
                # For Conv1D output (B, C, L), to normalize over channels C, we need to permute
                # so that C becomes the last dimension for LayerNorm.
                x = x.permute(0, 2, 1) # (B, L_out, C_out)
                x = self.layer_norms[i](x)
                x = x.permute(0, 2, 1) # Permute back to (B, C_out, L_out)
            
            x = self.dropout_fn(x)
            
        # Final permutation to (B, L_out, C_out) for subsequent Transformer layers
        x = x.permute(0, 2, 1) 
        return x

if __name__ == '__main__':
    print("--- Testing model_components.py ---")
    
    # Test Configuration
    batch_size = 4
    seq_len = 64
    d_model_test = 128 # Corresponds to embedding_dim or in_channels
    max_pos_len = 256
    
    # 1. Test LearnablePositionalEncoding
    print("\n1. Testing LearnablePositionalEncoding...")
    try:
        pos_encoder = LearnablePositionalEncoding(d_model=d_model_test, dropout=0.1, max_len=max_pos_len)
        print(f"Positional Encoder initialized: {pos_encoder}")
        
        dummy_input_x = torch.randn(batch_size, seq_len, d_model_test)
        print(f"Input shape for Positional Encoder: {dummy_input_x.shape}")
        
        output_pe = pos_encoder(dummy_input_x)
        print(f"Output shape from Positional Encoder: {output_pe.shape}")
        assert output_pe.shape == dummy_input_x.shape, "PE output shape mismatch."
        
        # Test with sequence length exceeding max_len (should raise error)
        try:
            failing_input_x = torch.randn(batch_size, max_pos_len + 1, d_model_test)
            pos_encoder(failing_input_x)
            print("PE Test FAILED: Did not raise error for seq_len > max_len.")
        except ValueError as e:
            print(f"PE Test PASSED: Correctly raised ValueError for seq_len > max_len: {e}")
            
    except Exception as e:
        print(f"Error during LearnablePositionalEncoding test: {e}")
        import traceback
        traceback.print_exc()

    # 2. Test CNNFrontend
    print("\n2. Testing CNNFrontend...")
    cnn_config_test = {
        "in_channels": d_model_test, # From previous d_model
        "out_channels_list": [d_model_test // 2, d_model_test // 4], # Example channel reduction
        "kernel_sizes": [5, 3],
        "stride": 1, # No sequence length reduction with stride 1
        "cnn_dropout": 0.1,
        "activation": "GELU",
        "use_layernorm": True,
        "padding_mode": "zeros"
    }
    try:
        cnn_frontend = CNNFrontend(**cnn_config_test)
        print(f"CNN Frontend initialized: {cnn_frontend}")

        dummy_input_cnn = torch.randn(batch_size, seq_len, cnn_config_test["in_channels"])
        print(f"Input shape for CNN Frontend: {dummy_input_cnn.shape}")

        output_cnn = cnn_frontend(dummy_input_cnn)
        print(f"Output shape from CNN Frontend: {output_cnn.shape}")
        
        expected_cnn_output_channels = cnn_config_test["out_channels_list"][-1]
        expected_cnn_output_seq_len = seq_len # Since stride is 1
        assert output_cnn.shape == (batch_size, expected_cnn_output_seq_len, expected_cnn_output_channels), \
            f"CNN output shape mismatch. Expected {(batch_size, expected_cnn_output_seq_len, expected_cnn_output_channels)}, got {output_cnn.shape}"

        # Test with stride > 1
        cnn_config_stride_test = {**cnn_config_test, "stride": 2}
        cnn_frontend_stride = CNNFrontend(**cnn_config_stride_test)
        print(f"\nCNN Frontend with stride=2 initialized: {cnn_frontend_stride}")
        output_cnn_stride = cnn_frontend_stride(dummy_input_cnn)
        print(f"Output shape from CNN Frontend (stride=2): {output_cnn_stride.shape}")
        
        # Calculate expected sequence length after stride
        # L_out = floor((L_in + 2*P - K) / S) + 1
        # With 'same' padding (P=(K-1)/2 for S=1), L_out = L_in for S=1.
        # If S > 1, it's roughly L_in / S.
        # For multiple layers, it's more complex, but for a single layer or all layers having same stride:
        expected_cnn_output_seq_len_stride = seq_len
        for _ in cnn_config_stride_test["kernel_sizes"]: # Assuming stride applies to each layer
             # This calculation is simplified; true L_out depends on padding and kernel at each stage
             # For P = (K-1)//2, L_out = ceil(L_in/S) if K is odd.
             # Or more generally, L_out = floor((L_in + 2*P - K)/S + 1)
             # If padding = (K-1)//2, then L_out = floor((L_in + K - 1 - K)/S + 1) = floor((L_in - 1)/S + 1)
             # For PyTorch Conv1d with padding=(k-1)//2 and stride=S: L_out = floor((L_in -1)/S) + 1 if k is odd.
             # If L_in = 64, S=2, K=5 -> P=2. L_out = floor((64+4-5)/2)+1 = floor(63/2)+1 = 31+1 = 32
             # If L_in = 32, S=2, K=3 -> P=1. L_out = floor((32+2-3)/2)+1 = floor(31/2)+1 = 15+1 = 16
             
             # Let's re-evaluate based on PyTorch's formula with 'same'-like padding:
             # L_out = floor((L_in + 2*padding[0] - dilation[0]*(kernel_size[0]-1) -1)/stride[0] + 1)
             # If padding = (kernel_size-1)//2 (our case for 'same' effect with stride 1)
             # L_out = floor((L_in + kernel_size-1 - (kernel_size-1) -1)/stride + 1)
             # L_out = floor((L_in - 1)/stride + 1)
             
             # Let's trace L_in through the layers with stride 2
             temp_l_in = expected_cnn_output_seq_len_stride
             k_s = cnn_config_stride_test["kernel_sizes"][0 if temp_l_in == seq_len else 1] # Simplified
             p_s = (k_s - 1) // 2
             expected_cnn_output_seq_len_stride = (temp_l_in + 2 * p_s - k_s) // cnn_config_stride_test["stride"] + 1


        # After 1st conv (k=5, s=2, p=2): L_out = (64 + 2*2 - 5)//2 + 1 = (64+4-5)//2 + 1 = 63//2 + 1 = 31+1 = 32
        # After 2nd conv (k=3, s=2, p=1, L_in=32): L_out = (32 + 2*1 - 3)//2 + 1 = (32+2-3)//2+1 = 31//2 + 1 = 15+1 = 16
        # So, if stride applies per layer:
        l_out_s2 = seq_len
        for k_val in cnn_config_stride_test["kernel_sizes"]:
            l_out_s2 = (l_out_s2 + 2*((k_val-1)//2) - k_val) // cnn_config_stride_test["stride"] + 1
        
        expected_cnn_output_seq_len_stride_final = l_out_s2

        assert output_cnn_stride.shape == (batch_size, expected_cnn_output_seq_len_stride_final, expected_cnn_output_channels), \
            f"CNN (stride=2) output shape mismatch. Expected {(batch_size, expected_cnn_output_seq_len_stride_final, expected_cnn_output_channels)}, got {output_cnn_stride.shape}"

    except Exception as e:
        print(f"Error during CNNFrontend test: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n--- model_components.py tests completed (check for assertions) ---")
