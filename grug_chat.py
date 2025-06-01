# grug_chat.py
import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import traceback

# --- Configuration for Chat ---
CHAT_CONFIG = {
    "checkpoint_path": "./checkpoints/byte_llm_lstm_attn_best.pth", # IMPORTANT: Update this path
    "generation_length": 150,         # How many new bytes Grug should generate per turn
    "temperature": 0.9,               # Temperature for sampling (e.g., 0.7-1.5)
    "top_k": 40,                      # Top-k sampling (0 to disable)
    "max_history_bytes": 1024,        # Max bytes of conversation history to feed as context
                                      # Should ideally be related to model's sequence_length, but predictor handles padding/truncation.
    "user_prompt_prefix": "\nYou: ",
    "grug_response_prefix": "Grug: "
}

# --- Model Architecture (Copied from your training script) ---
class ByteLLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout, num_attention_heads):
        super(ByteLLM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim # LSTM hidden dim
        self.num_lstm_layers = num_layers # Renamed for clarity
        self.num_attention_heads = num_attention_heads

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, self.num_lstm_layers, 
                            dropout=(dropout if self.num_lstm_layers > 1 else 0),
                            batch_first=True)

        if hidden_dim % num_attention_heads != 0:
            raise ValueError(
                f"LSTM hidden_dim ({hidden_dim}) must be divisible by num_attention_heads ({num_attention_heads})"
            )
        
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.dropout_layer = nn.Dropout(dropout) 
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden_lstm_state_initial):
        embedded = self.embedding(x)
        lstm_out, hidden_lstm_state_final = self.lstm(embedded, hidden_lstm_state_initial)
        attn_output, _ = self.attention_layer(query=lstm_out, key=lstm_out, value=lstm_out)
        out_from_attention = attn_output[:, -1, :]
        out_after_dropout = self.dropout_layer(out_from_attention)
        final_out = self.fc(out_after_dropout)
        return final_out, hidden_lstm_state_final

    def init_hidden(self, batch_size, device):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.num_lstm_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.num_lstm_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden

# --- Predictor Class (Copied from your training script, with sampling) ---
class Predictor:
    def __init__(self, model, device, sequence_length_for_padding, temperature=1.0, top_k=0):
        self.model = model 
        self.device = device
        self.sequence_length = sequence_length_for_padding # Used for padding/truncating input to model
        
        if temperature <= 0:
            # Allow temperature of 0 for greedy, but very small positive for near-greedy
            if temperature == 0: 
                print("Warning: Temperature is 0, using a very small value (1e-8) for near-greedy sampling.")
                self.temperature = 1e-8
            else:
                 raise ValueError("Temperature must be positive.")
        else:
            self.temperature = temperature
        self.top_k = top_k
        print(f"Predictor initialized with temperature: {self.temperature:.2f}, top_k: {self.top_k}, sequence_length_for_padding: {self.sequence_length}")

    def predict_next_byte(self, byte_sequence_values):
        if not isinstance(byte_sequence_values, np.ndarray):
            byte_sequence_values = np.array(byte_sequence_values, dtype=np.int64)
        
        current_input_len = len(byte_sequence_values)
        # Pad or truncate the input sequence to match self.sequence_length
        if current_input_len == 0:
            current_sequence_padded = np.zeros(self.sequence_length, dtype=np.int64)
        elif current_input_len < self.sequence_length:
            padding = np.zeros(self.sequence_length - current_input_len, dtype=np.int64)
            current_sequence_padded = np.concatenate((padding, byte_sequence_values))
        else: # current_input_len >= self.sequence_length
            current_sequence_padded = byte_sequence_values[-self.sequence_length:]
        
        input_tensor = torch.tensor([current_sequence_padded], dtype=torch.long).to(self.device)
        
        self.model.eval()
        with torch.no_grad(): 
            hidden_lstm = self.model.init_hidden(1, self.device) 
            output_model, _ = self.model(input_tensor, hidden_lstm)
            
            logits = output_model.squeeze(0)
            logits = logits / self.temperature

            if self.top_k > 0:
                k = min(max(1, self.top_k), logits.size(-1)) 
                top_k_vals, top_k_indices = torch.topk(logits, k)
                filtered_logits = torch.full_like(logits, -float('Inf'))
                filtered_logits.scatter_(0, top_k_indices, top_k_vals)
            else:
                filtered_logits = logits

            probabilities = torch.softmax(filtered_logits, dim=-1)

            if torch.isnan(probabilities).any() or probabilities.sum() < 1e-6 :
                print(f"Warning: Invalid probabilities (sum: {probabilities.sum().item()}). Falling back to greedy on raw model output.")
                predicted_byte_value = torch.argmax(output_model.squeeze(0)).item() 
            else:
                predicted_byte_value = torch.multinomial(probabilities, 1).item()
            
        return predicted_byte_value

    def generate_sequence(self, seed_bytes, length=100):
        if isinstance(seed_bytes, bytes):
            current_sequence_values = list(seed_bytes)
        elif isinstance(seed_bytes, list) and all(isinstance(x, int) for x in seed_bytes):
            current_sequence_values = list(seed_bytes)
        else:
            raise ValueError("seed_bytes must be a bytes object or a list of integers representing byte values.")

        generated_values = list(current_sequence_values) 

        print(f"Generating {length} bytes. Initial seed length: {len(generated_values)}")
        for i in range(length):
            # The input for prediction is the current state of generated_values
            # The predictor's predict_next_byte will handle padding/truncation to its self.sequence_length
            input_for_prediction = np.array(generated_values, dtype=np.int64)
            
            next_byte_value = self.predict_next_byte(input_for_prediction)
            generated_values.append(next_byte_value)
            # Simple progress indicator
            if (i + 1) % 50 == 0:
                print(f"  Generated {i+1}/{length} bytes...")
        
        print(f"Finished generation. Total length: {len(generated_values)}")
        return bytes(generated_values)

# --- Utility Functions for Chat ---
def load_model_for_chat(checkpoint_path_str, device_str):
    """Loads the ByteLLM model from a checkpoint."""
    device = torch.device(device_str)
    checkpoint_path = Path(checkpoint_path_str)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Use config saved in checkpoint, fallback to a default if not present (less ideal)
    # For chat, we mainly need model architecture params.
    # The training CONFIG might have other keys not relevant here.
    model_config = checkpoint.get('config', {})
    if not model_config:
        raise ValueError("Checkpoint does not contain a 'config' dictionary. Cannot determine model parameters.")

    # Extract necessary parameters for model instantiation
    # Provide fallbacks from a default or raise error if critical ones are missing
    vocab_size = model_config.get('vocab_size', 256) # Should be 256 for byte model
    embedding_dim = model_config.get('embedding_dim')
    hidden_dim = model_config.get('hidden_dim')
    num_layers = model_config.get('num_layers') # LSTM layers
    dropout = model_config.get('dropout')
    num_attention_heads = model_config.get('num_attention_heads')
    
    # Ensure critical parameters are present
    critical_params = {
        'embedding_dim': embedding_dim, 'hidden_dim': hidden_dim, 
        'num_layers': num_layers, 'dropout': dropout, 
        'num_attention_heads': num_attention_heads
    }
    for param_name, param_val in critical_params.items():
        if param_val is None:
            raise ValueError(f"Critical model parameter '{param_name}' not found in checkpoint's config.")

    # The sequence_length from the checkpoint's config is the one the model was trained with.
    # This is important for the Predictor's internal padding/truncation logic.
    predictor_sequence_length = model_config.get('sequence_length')
    if predictor_sequence_length is None:
        raise ValueError("Checkpoint's config must contain 'sequence_length'.")


    print(f"Instantiating model with params from checkpoint config:")
    print(f"  Vocab Size: {vocab_size}, Embedding Dim: {embedding_dim}, Hidden Dim: {hidden_dim}")
    print(f"  LSTM Layers: {num_layers}, Dropout: {dropout}, Attention Heads: {num_attention_heads}")
    print(f"  Predictor will use sequence_length_for_padding: {predictor_sequence_length}")


    model = ByteLLM(
        vocab_size=vocab_size, 
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        num_attention_heads=num_attention_heads
    ).to(device)
    
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError as e:
        print(f"Error loading state_dict: {e}")
        print("This might be due to an architecture mismatch between the saved model and the ByteLLM class definition.")
        print("Ensure the ByteLLM class in this script matches the one used for training the checkpoint.")
        raise
        
    model.eval() # Set model to evaluation mode
    print("Model loaded successfully and set to evaluation mode.")
    return model, predictor_sequence_length # Return predictor_sequence_length as well

def chat_with_grug():
    """Main interactive chat loop."""
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device_str}")

    try:
        model, predictor_padding_seq_len = load_model_for_chat(CHAT_CONFIG["checkpoint_path"], device_str)
    except Exception as e:
        print(f"Could not load the model: {e}")
        traceback.print_exc()
        return

    predictor = Predictor(
        model, 
        torch.device(device_str), 
        predictor_padding_seq_len, # Use the sequence length the model was trained with for predictor's internal padding
        temperature=CHAT_CONFIG["temperature"], 
        top_k=CHAT_CONFIG["top_k"]
    )

    print("\n--- Grug Chat Initialized ---")
    print(f"Type your message and press Enter. Type 'quit', 'exit', or 'bye' to end.")
    print(f"Generation settings: Length={CHAT_CONFIG['generation_length']}, Temp={CHAT_CONFIG['temperature']:.2f}, Top-K={CHAT_CONFIG['top_k']}")
    print("Grug's context window for generation is effectively managed by the Predictor.")
    print("Conversation history is maintained up to `max_history_bytes`.")
    print("-------------------------------------\n")

    # Initialize with Grug's self-introduction if available and desired
    # For this example, we'll start with an empty history.
    # You could load the "grug_introduction_to_self" text here as an initial seed.
    # e.g., conversation_history_bytes = Path("path/to/grug_introduction_to_self.txt").read_bytes()
    conversation_history_bytes = b""

    while True:
        try:
            user_input_text = input(CHAT_CONFIG["user_prompt_prefix"])
            if user_input_text.lower() in ['quit', 'exit', 'bye']:
                print("Exiting chat. Goodbye!")
                break
            if not user_input_text.strip():
                continue

            # Encode user input and append to history
            user_input_bytes = (CHAT_CONFIG["user_prompt_prefix"] + user_input_text + "\n").encode('utf-8', errors='replace')
            
            # Combine history with new input for the seed
            current_seed_bytes = conversation_history_bytes + user_input_bytes
            
            # Truncate combined seed if it exceeds max_history_bytes (from the beginning)
            # This ensures the input to generate_sequence doesn't grow indefinitely,
            # though the Predictor itself will truncate/pad to its internal sequence_length.
            if len(current_seed_bytes) > CHAT_CONFIG["max_history_bytes"]:
                print(f"(Trimming conversation history from {len(current_seed_bytes)} to {CHAT_CONFIG['max_history_bytes']} bytes)")
                current_seed_bytes = current_seed_bytes[-CHAT_CONFIG["max_history_bytes"]:]

            print(f"\n{CHAT_CONFIG['grug_response_prefix']}", end="", flush=True) # Print prefix, no newline yet
            
            # Generate sequence based on the current seed
            # The predictor will take current_seed_bytes and internally pad/truncate its input
            # to predictor_padding_seq_len for the model.
            full_generated_bytes = predictor.generate_sequence(
                seed_bytes=current_seed_bytes, 
                length=CHAT_CONFIG["generation_length"]
            )
            
            # Extract only Grug's new response part
            grug_response_bytes = full_generated_bytes[len(current_seed_bytes):]
            
            try:
                grug_response_text = grug_response_bytes.decode('utf-8', errors='replace')
                print(grug_response_text) # Print the response
            except UnicodeDecodeError:
                print("[Error decoding Grug's response. Raw bytes displayed below]")
                print(grug_response_bytes)
                grug_response_text = "[Decoding Error]" # Placeholder for history

            # Update conversation history
            # Append the user's input and Grug's actual generated bytes
            conversation_history_bytes += user_input_bytes 
            conversation_history_bytes += (CHAT_CONFIG["grug_response_prefix"].encode('utf-8') + grug_response_bytes + b"\n")

            # Truncate overall conversation history if it exceeds max_history_bytes
            if len(conversation_history_bytes) > CHAT_CONFIG["max_history_bytes"]:
                conversation_history_bytes = conversation_history_bytes[-CHAT_CONFIG["max_history_bytes"]:]


        except KeyboardInterrupt:
            print("\nExiting chat due to interrupt. Goodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            traceback.print_exc()
            # Optionally, decide if you want to break the loop on general errors

if __name__ == "__main__":
    chat_with_grug()
