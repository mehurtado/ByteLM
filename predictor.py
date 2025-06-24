# predictor.py

import torch
import torch.nn.functional as F # For softmax

# Assuming model.py and config.py are in the same directory or accessible
from model import ByteLLM_GrugV3
from config import CONFIG_V3 # For default generation/model parameters in tests

class Predictor:
    """
    Handles sequence generation using a trained ByteLLM_GrugV3 model.
    """
    def __init__(self, model: ByteLLM_GrugV3, device: torch.device,
                 generation_config: dict, model_internal_config: dict):
        """
        Initializes the Predictor.

        Args:
            model (ByteLLM_GrugV3): The trained model instance.
            device (torch.device): The device to run generation on (e.g., 'cuda', 'cpu').
            generation_config (dict): Configuration for generation, e.g.,
                                      {"generation_temperature": 1.0, "generation_top_k": 0}.
            model_internal_config (dict): Configuration related to the model's architecture
                                          that affects prediction, e.g.,
                                          {"max_positional_encoding_len": 512, "sequence_length": 16}.
                                          'sequence_length' here refers to the training context length.
        """
        self.model = model.to(device).eval() # Ensure model is on correct device and in eval mode
        self.device = device
        
        self.temperature = generation_config.get("generation_temperature", 1.0)
        self.top_k = generation_config.get("generation_top_k", 0) # 0 means no top-k filtering

        self.model_context_len = model_internal_config.get("sequence_length", 256)
        self.max_model_input_len = model_internal_config.get("max_positional_encoding_len", 4096)

        if self.temperature <= 0:
            raise ValueError("Temperature must be positive.")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative.")

        print(f"Predictor initialized: Temp={self.temperature}, TopK={self.top_k}, "
              f"ModelContextLen={self.model_context_len}, MaxModelInputLen={self.max_model_input_len}")

    @torch.no_grad() # Disable gradient calculations during generation
    def generate_sequence(self, seed_bytes: bytes | list[int], length: int = 100) -> bytes:
        """
        Generates a sequence of bytes starting from seed_bytes.
        This version primes the context by padding the seed to the model's context length
        to improve initial generation coherence.
        """
        self.model.eval()

        if isinstance(seed_bytes, bytes):
            seed_values = list(seed_bytes)
        elif isinstance(seed_bytes, list) and all(isinstance(x, int) and 0 <= x <= 255 for x in seed_bytes):
            seed_values = list(seed_bytes)
        else:
            raise ValueError("seed_bytes must be actual bytes or a list of integers (0-255).")

        if length <= 0:
            return bytes(seed_values)

        # --- CONTEXT PRIMING LOGIC ---
        # Prime the context by padding the seed to the model's training context length.
        # We use byte 0 (the NULL byte) as a neutral padding token.
        num_padding = self.model_context_len - len(seed_values)
        
        if num_padding > 0:
            # Pad on the left side to fill the context window
            current_sequence_values = ([0] * num_padding) + seed_values
        else:
            # If the seed is already longer than the context, just use the last part of it
            current_sequence_values = seed_values[-self.model_context_len:]
        
        generated_continuation = []
        for _ in range(length):
            # The input to the model will now always be the last `model_context_len` tokens
            input_sequence_for_model = current_sequence_values[-self.model_context_len:]
            input_tensor = torch.tensor([input_sequence_for_model], dtype=torch.long, device=self.device)
            
            logits = self.model(input_tensor) # (1, vocab_size)
            
            logits_scaled = logits / self.temperature

            if self.top_k > 0:
                k = min(max(1, self.top_k), logits_scaled.size(-1))
                top_k_vals, top_k_indices = torch.topk(logits_scaled, k, dim=-1)
                
                filtered_logits = torch.full_like(logits_scaled, float('-inf'))
                filtered_logits.scatter_(-1, top_k_indices, top_k_vals)
            else:
                filtered_logits = logits_scaled
            
            probabilities = F.softmax(filtered_logits, dim=-1)
            
            if torch.isnan(probabilities).any() or probabilities.sum() < 1e-6 :
                print("Warning: Invalid probabilities encountered during generation. Falling back to argmax.")
                next_byte_val = torch.argmax(logits_scaled, dim=-1).item()
            else:
                next_byte_val = torch.multinomial(probabilities, 1).item()
            
            # Append the new byte to our working sequence for the next iteration
            current_sequence_values.append(next_byte_val)
            # Also append to the list of newly generated bytes
            generated_continuation.append(next_byte_val)
        
        # Return the original seed plus the newly generated part (without the padding)
        return bytes(seed_values + generated_continuation)

if __name__ == '__main__':
    print("--- Testing predictor.py ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use a minimal version of CONFIG_V3 for the dummy model
    test_pred_config = CONFIG_V3.copy()
    test_pred_config["embedding_dim"] = 32
    test_pred_config["attention_d_model"] = 32
    test_pred_config["num_attention_layers"] = 1
    test_pred_config["attention_num_heads"] = 2
    test_pred_config["sequence_length"] = 16        # Training context length
    test_pred_config["max_positional_encoding_len"] = 32 # Max PE length
    test_pred_config["use_parallel_stream_model"] = False # Test with simpler model

    print(f"Test Predictor - Model Config: {test_pred_config}")
    
    try:
        dummy_model = ByteLLM_GrugV3(test_pred_config).to(device)
        print("Dummy model created.")

        generation_params = {
            "generation_temperature": 0.8,
            "generation_top_k": 5
        }
        model_internal_params_for_predictor = {
            "sequence_length": test_pred_config["sequence_length"],
            "max_positional_encoding_len": test_pred_config["max_positional_encoding_len"]
        }
        
        predictor = Predictor(dummy_model, device, generation_params, model_internal_params_for_predictor)
        print("Predictor initialized.")

        seed_text = "Hello"
        seed_bytes = seed_text.encode('utf-8')
        num_bytes_to_generate = 20

        print(f"\nGenerating sequence with seed: '{seed_text}' (bytes: {seed_bytes})")
        generated_output_bytes = predictor.generate_sequence(seed_bytes, length=num_bytes_to_generate)
        
        print(f"Generated output (bytes type): {generated_output_bytes}")
        expected_total_length = len(seed_bytes) + num_bytes_to_generate
        assert len(generated_output_bytes) == expected_total_length, \
            f"Generated sequence length mismatch. Expected {expected_total_length}, got {len(generated_output_bytes)}"
        print(f"Decoded generated text: \"{generated_output_bytes.decode('utf-8', 'replace')}\"")
        
        print("\n--- predictor.py tests completed successfully (if no assertions failed) ---")

    except Exception as e:
        print(f"\nAn error occurred during predictor.py testing:")
        import traceback
        traceback.print_exc()