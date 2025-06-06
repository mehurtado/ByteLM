import torch
import torch.nn as nn
import torch.nn.functional as F # Required for F.gelu
import math # Required for SinusoidalPositionalEncoding

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        # self.max_len = max_len # Not strictly needed to store if pe is the source of truth for current max_len

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0: # Handle odd d_model by ensuring the cos part doesn't go out of bounds
            pe[:, 1::2] = torch.cos(position * div_term[:-1] if div_term.size(0) > d_model // 2 else position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        current_max_len = self.pe.size(0)

        if seq_len > current_max_len:
            print(f"Extending PE from {current_max_len} to {seq_len}")
            # Dynamically extend pe
            new_pe = torch.zeros(seq_len, self.d_model, device=self.pe.device, dtype=self.pe.dtype)
            new_pe[:current_max_len, :] = self.pe # Copy old values

            position = torch.arange(current_max_len, seq_len, dtype=torch.float, device=self.pe.device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * \
                                 (-math.log(10000.0) / self.d_model)).to(self.pe.device)

            new_pe[current_max_len:, 0::2] = torch.sin(position * div_term)
            if self.d_model % 2 != 0:
                 new_pe[current_max_len:, 1::2] = torch.cos(position * div_term[:-1] if div_term.size(0) > self.d_model // 2 else position * div_term)
            else:
                 new_pe[current_max_len:, 1::2] = torch.cos(position * div_term)

            self.register_buffer('pe', new_pe) # Re-registering with the same name updates the buffer

        # Ensure pe slice is on the same device as x for the addition
        x = x + self.pe[:seq_len, :].to(x.device)
        return self.dropout(x)

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
        self.pos_embedding = nn.Embedding(max_len, d_model)
        # Prioritize using register_buffer for tensors not updated by optimizer
        # but part of the model state. Using torch.arange directly in forward
        # is also fine if max_len isn't excessively large causing recomputation overhead.
        # For very large max_len, buffering position_ids might be slightly more efficient.
        # self.register_buffer("position_ids", torch.arange(max_len).expand((1, -1)))
        # However, nn.Embedding handles its own weight (which includes these positions implicitly)
        # So, direct use of arange in forward is common.
        # Let's stick to a simpler self.max_len for the check in forward for now.
        self.max_len = max_len


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encodings to the input tensor.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, d_model).

        Returns:
            torch.Tensor: The input tensor with added positional encodings,
                          of the same shape as input x.
        """
        seq_len = x.size(1)

        if seq_len > self.max_len: # Check against stored max_len
            raise ValueError(
                f"Input sequence length ({seq_len}) exceeds max_len "
                f"({self.max_len}) for positional embeddings. "
                f"Increase max_len in config or ensure input sequences are truncated."
            )

        # Create position_ids on the fly, on the same device as x
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0) # Shape: (1, seq_len)
        pos_enc = self.pos_embedding(position_ids) # Shape: (1, seq_len, d_model)
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
            padding_mode (str): Padding mode for Conv1d.
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
            if k_size <= 0 : # Removed k_size % 2 == 0 check, padding calculation handles it
                raise ValueError(f"CNN kernel size {k_size} must be positive.")
            # Calculate padding for 'same' output length assuming stride=1 for padding calculation basis
            # For stride > 1, output length will be reduced.
            # This padding aims to keep features centered.
            padding = (k_size - 1) // 2

            conv_layer = nn.Conv1d(
                in_channels=current_in_channels,
                out_channels=o_channels,
                kernel_size=k_size,
                stride=stride, # Stride applied here
                padding=padding, # Padding calculated for k_size
                padding_mode=padding_mode,
                bias=not use_layernorm # Bias is False if LayerNorm is used, True otherwise
            )
            self.conv_layers.append(conv_layer)
            current_in_channels = o_channels # Update for next layer

        if activation.upper() == "RELU": self.activation_fn = nn.ReLU()
        elif activation.upper() == "GELU": self.activation_fn = nn.GELU()
        else: raise ValueError(f"Unsupported activation: {activation}. Choose 'RELU' or 'GELU'.")

        self.dropout_fn = nn.Dropout(cnn_dropout)
        self.use_layernorm = use_layernorm
        if self.use_layernorm:
            # LayerNorms are applied on the channel dimension (which is d_model here)
            self.layer_norms = nn.ModuleList([nn.LayerNorm(o_channels) for o_channels in out_channels_list])
        else:
            self.layer_norms = None # Explicitly set to None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes the input through the CNN layers.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embedding_dim).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, new_sequence_length, last_out_channel).
        """
        # Conv1d expects (batch_size, channels, sequence_length)
        x = x.permute(0, 2, 1)

        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            x = self.activation_fn(x)

            if self.use_layernorm and self.layer_norms:
                # LayerNorm expects (batch_size, sequence_length, channels) for direct application on last dim
                x = x.permute(0, 2, 1)
                x = self.layer_norms[i](x)
                x = x.permute(0, 2, 1) # Permute back to (batch_size, channels, sequence_length)

            x = self.dropout_fn(x)

        # Output shape (batch_size, sequence_length_new, channels_last)
        x = x.permute(0, 2, 1)
        return x

if __name__ == '__main__':
    print("--- Testing SinusoidalPositionalEncoding ---")

    d_model = 512
    batch_size = 4
    seq_len = 64
    dropout_rate = 0.1

    # 1. Initialization Test
    print("\n1. Initialization Test")
    try:
        pos_encoder = SinusoidalPositionalEncoding(d_model, dropout_rate, max_len=seq_len*2)
        print("SinusoidalPositionalEncoding initialized successfully.")
        print(f"Initial PE shape: {pos_encoder.pe.shape}")
        assert pos_encoder.pe.shape == (seq_len*2, d_model)
    except Exception as e:
        print(f"Initialization failed: {e}")
        raise

    # Prepare dummy input
    dummy_input = torch.randn(batch_size, seq_len, d_model)
    print(f"\nDummy input shape: {dummy_input.shape}")

    # 2. Output Shape Test
    print("\n2. Output Shape Test")
    try:
        output = pos_encoder(dummy_input)
        print(f"Output shape: {output.shape}")
        assert output.shape == dummy_input.shape, \
            f"Output shape mismatch. Expected {dummy_input.shape}, got {output.shape}"
        print("Output shape test passed.")
    except Exception as e:
        print(f"Output shape test failed: {e}")
        raise

    # 3. Encoding Values Test
    print("\n3. Encoding Values Test")
    try:
        pe_slice = pos_encoder.pe
        assert not torch.allclose(pe_slice[0, :], pe_slice[1, :]), \
            "Encoding for position 0 and 1 should be different."

        assert torch.allclose(pe_slice[0, 0::2], torch.zeros(pe_slice[0, 0::2].shape)), \
            "PE(0, even_dims) should be sin(0) = 0"
        assert torch.allclose(pe_slice[0, 1::2], torch.ones(pe_slice[0, 1::2].shape)), \
            "PE(0, odd_dims) should be cos(0) = 1"
        print("Encoding values at pos=0 are correct (sin(0)=0, cos(0)=1).")

        if pe_slice.size(0) > 1:
            assert not torch.allclose(pe_slice[1, 0::2], torch.zeros(pe_slice[1, 0::2].shape)), \
                "PE(1, even_dims) should largely be non-zero."
            assert not torch.allclose(pe_slice[1, 1::2], torch.zeros(pe_slice[1, 1::2].shape)), \
                "PE(1, odd_dims) should largely be non-zero."
            print("Encoding values at pos=1 are generally non-zero.")

        if d_model > 2:
             assert not torch.allclose(pe_slice[0,0], pe_slice[0,1]), \
                 "PE(pos, 2i) and PE(pos, 2i+1) should differ for pos=0 (0 vs 1)."
        print("Encoding values test passed.")
    except Exception as e:
        print(f"Encoding values test failed: {e}")
        raise

    # 4. Extrapolation Test
    print("\n4. Extrapolation Test")
    initial_max_len = 32
    extrapolation_seq_len = initial_max_len * 2

    try:
        pos_encoder_extrap = SinusoidalPositionalEncoding(d_model, dropout_rate, max_len=initial_max_len)
        print(f"Initialized for extrapolation test with max_len={initial_max_len}. PE shape: {pos_encoder_extrap.pe.shape}")
        assert pos_encoder_extrap.pe.size(0) == initial_max_len

        extrap_input = torch.randn(batch_size, extrapolation_seq_len, d_model)
        print(f"Extrapolation input shape: {extrap_input.shape}")

        output_extrap = pos_encoder_extrap(extrap_input)
        print(f"Output shape after extrapolation: {output_extrap.shape}")

        assert output_extrap.shape == extrap_input.shape
        print(f"PE buffer shape after extrapolation: {pos_encoder_extrap.pe.shape}")
        assert pos_encoder_extrap.pe.size(0) == extrapolation_seq_len
        print("Extrapolation test passed.")

        even_longer_seq_len = extrapolation_seq_len * 2
        extrap_input_2 = torch.randn(batch_size, even_longer_seq_len, d_model)
        output_extrap_2 = pos_encoder_extrap(extrap_input_2)
        assert pos_encoder_extrap.pe.size(0) == even_longer_seq_len
        print("Second extrapolation (re-extension) test passed.")

    except Exception as e:
        print(f"Extrapolation test failed: {e}")
        raise

    # 5. Device Handling Test
    print("\n5. Device Handling Test")
    device_to_test = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing device handling on: {device_to_test}")

    try:
        pos_encoder_device = SinusoidalPositionalEncoding(d_model, dropout_rate, max_len=seq_len)
        pos_encoder_device.to(device_to_test)

        device_input = torch.randn(batch_size, seq_len, d_model).to(device_to_test)
        print(f"Input tensor device: {device_input.device}")
        print(f"Module PE buffer device before forward: {pos_encoder_device.pe.device}")

        output_device = pos_encoder_device(device_input)

        assert output_device.device == device_input.device
        assert pos_encoder_device.pe.device == device_input.device
        print("Output and PE buffer are on the correct device after forward pass.")

        extrap_len_device = seq_len * 2
        device_input_extrap = torch.randn(batch_size, extrap_len_device, d_model).to(device_to_test)
        print(f"Extrapolation input tensor device: {device_input_extrap.device}")

        output_device_extrap = pos_encoder_device(device_input_extrap)

        assert output_device_extrap.device == device_input_extrap.device
        assert pos_encoder_device.pe.device == device_input_extrap.device
        print("Device handling test passed (including extrapolation).")

    except Exception as e:
        print(f"Device handling test failed: {e}")
        raise

    print("\n--- All SinusoidalPositionalEncoding tests completed successfully! ---")

    # Minimal tests for LearnablePositionalEncoding (can be expanded)
    print("\n--- Testing LearnablePositionalEncoding (Minimal) ---")
    try:
        lpe = LearnablePositionalEncoding(d_model=d_model, max_len=seq_len)
        print("LearnablePositionalEncoding initialized.")
        lpe_output = lpe(dummy_input)
        assert lpe_output.shape == dummy_input.shape
        print("LearnablePositionalEncoding forward pass and shape test passed.")
    except Exception as e:
        print(f"LearnablePositionalEncoding test failed: {e}")
        raise

    # Minimal tests for CNNFrontend (can be expanded)
    print("\n--- Testing CNNFrontend (Minimal) ---")
    try:
        cnn_out_channels = [d_model // 2, d_model]
        cnn_kernels = [3, 3]
        cnn_frontend = CNNFrontend(in_channels=d_model,
                                   out_channels_list=cnn_out_channels,
                                   kernel_sizes=cnn_kernels)
        print("CNNFrontend initialized.")
        # Dummy input for CNN: (batch_size, seq_len, d_model)
        cnn_output = cnn_frontend(dummy_input)
        # Expected output shape: (batch_size, seq_len_after_stride, cnn_out_channels[-1])
        # If stride is 1, seq_len should be the same.
        expected_cnn_shape = (batch_size, seq_len, cnn_out_channels[-1])
        assert cnn_output.shape == expected_cnn_shape, \
            f"CNNFrontend output shape mismatch. Expected {expected_cnn_shape}, got {cnn_output.shape}"
        print("CNNFrontend forward pass and shape test passed.")
    except Exception as e:
        print(f"CNNFrontend test failed: {e}")
        raise

    print("\n--- All model_components.py tests completed successfully! ---")
