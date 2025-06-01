# main.py

import torch
import torch.optim as optim
import torch.nn as nn
import traceback
from pathlib import Path

# Import from our modularized files
from config import CONFIG_V3
from utils import ensure_dir, generate_dummy_data
from dataset import DataProcessor
from model import ByteLLM_GrugV3 # ByteLLM_GrugV3 uses model_components internally
from predictor import Predictor
from trainer import Trainer

# --- Helper Functions for Main Execution Orchestration ---

def setup_environment(config_dict: dict) -> torch.device:
    """
    Sets up necessary directories, the device, and CUDA backend settings.
    """
    ensure_dir(config_dict["data_dir"])
    ensure_dir(config_dict["checkpoint_dir"])
    ensure_dir(config_dict["processed_data_dir"])
    
    if config_dict.get("profiler_log_dir") and config_dict.get("enable_profiler", False):
        ensure_dir(config_dict["profiler_log_dir"])
        ensure_dir(Path(config_dict["profiler_log_dir"]) / "train")
        ensure_dir(Path(config_dict["profiler_log_dir"]) / "eval")

    if config_dict.get("generate_dummy_data_if_empty", True):
        # Pass the relevant part of the config if generate_dummy_data needs it
        generate_dummy_data(config_dict["data_dir"], config_dict)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
        if config_dict.get("cudnn_benchmark", False):
            torch.backends.cudnn.benchmark = True
            print("torch.backends.cudnn.benchmark = True (May speed up training if input sizes are constant)")
        # torch.backends.cuda.matmul.allow_tf32 = True # For Ampere and later, TF32 can speed up matmuls

    print(f"Model name for checkpoints: {config_dict['model_name']}")
    return device

def load_data_components(config_dict: dict) -> tuple[torch.utils.data.DataLoader | None, torch.utils.data.DataLoader | None]:
    """
    Initializes the DataProcessor and returns train and validation DataLoaders.
    Updates config_dict in place if vocab_size changes.
    """
    current_seq_len = config_dict["sequence_length"]
    print(f"Initializing DataProcessor with sequence_length: {current_seq_len}")

    data_processor = DataProcessor(
        data_dir=config_dict["data_dir"],
        processed_data_dir=config_dict["processed_data_dir"],
        sequence_length=current_seq_len, # This is the default sequence length for creating the .npy file
        force_reprocess=config_dict.get("force_reprocess_data", False),
        config_for_data_gen=config_dict # Pass the whole config for dummy data generation logic
    )
    
    # Get DataLoaders using the current_sequence_length from config for this run
    train_dataloader, val_dataloader = data_processor.get_dataloaders(
        batch_size=config_dict["batch_size"],
        val_split_ratio=config_dict["val_split_ratio"],
        num_workers=config_dict.get("num_workers", 0),
        current_sequence_length=current_seq_len # Explicitly pass sequence length for dataloader creation
    )

    vocab_size = data_processor.get_vocab_size() # Should be 256 for bytes
    if config_dict.get("vocab_size") != vocab_size:
        print(f"Warning: CONFIG_V3 vocab_size {config_dict.get('vocab_size')} differs from DataProcessor's {vocab_size}. "
              f"Using DataProcessor's ({vocab_size}).")
        config_dict["vocab_size"] = vocab_size # Update config in place
        
    return train_dataloader, val_dataloader

def initialize_optimizer(model: nn.Module, optim_config: dict) -> torch.optim.Optimizer:
    """Initializes the optimizer based on configuration."""
    lr = optim_config.get("learning_rate", 1e-3)
    optimizer_type = optim_config.get("optimizer_type", "AdamW").lower()
    
    # Filter parameters that require gradients
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    if optimizer_type == "adamw":
        optimizer = optim.AdamW(
            trainable_params, lr=lr,
            betas=(optim_config.get("adam_beta1", 0.9), optim_config.get("adam_beta2", 0.98)),
            eps=optim_config.get("adam_eps", 1e-9),
            weight_decay=optim_config.get("weight_decay", 0.01)
        )
    elif optimizer_type == "adam":
        optimizer = optim.Adam(
            trainable_params, lr=lr,
            betas=(optim_config.get("adam_beta1", 0.9), optim_config.get("adam_beta2", 0.999)), # Default Adam betas
            eps=optim_config.get("adam_eps", 1e-8) # Default Adam eps
            # Adam does not typically use weight_decay directly in its constructor like AdamW
        )
    else:
        raise ValueError(f"Unsupported optimizer_type: {optimizer_type}. Choose 'AdamW' or 'Adam'.")
    print(f"Optimizer initialized: {optimizer_type} with LR: {lr}")
    return optimizer

def initialize_scheduler(optimizer: torch.optim.Optimizer, scheduler_config: dict, 
                         batches_per_epoch: int = None) -> torch.optim.lr_scheduler._LRScheduler | None:
    """Initializes the learning rate scheduler based on configuration."""
    scheduler_type = scheduler_config.get("scheduler_type")
    if not scheduler_type:
        print("No scheduler_type specified. No LR scheduler will be used.")
        return None

    scheduler_type = scheduler_type.lower()
    num_epochs_for_scheduler = scheduler_config.get("num_epochs", 50) # Total epochs from main config

    if scheduler_type == "reducelronplateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', # Assuming we monitor validation loss
            factor=scheduler_config.get("lr_scheduler_factor", 0.1),
            patience=scheduler_config.get("lr_scheduler_patience", 10)
        )
        print(f"ReduceLROnPlateau scheduler initialized.")
    elif scheduler_type == "cosineannealinglr":
        # T_max can be directly set, or calculated if batches_per_epoch is known
        T_max_direct = scheduler_config.get("lr_scheduler_T_max") # Explicit T_max in steps
        
        if T_max_direct is not None:
            T_max = int(T_max_direct)
            print(f"Using T_max from config for CosineAnnealingLR: {T_max} steps.")
        elif batches_per_epoch is not None and batches_per_epoch > 0:
            # Calculate T_max based on total training steps (excluding warmup)
            warmup_steps = scheduler_config.get("lr_warmup_steps", 0) if scheduler_config.get("use_lr_warmup") else 0
            total_steps_for_scheduler = (num_epochs_for_scheduler * batches_per_epoch) - warmup_steps
            T_max = max(1, total_steps_for_scheduler) # Ensure T_max is at least 1
            print(f"Calculated T_max for CosineAnnealingLR: {T_max} steps (Epochs: {num_epochs_for_scheduler}, "
                  f"Batches/Epoch: {batches_per_epoch}, Warmup Steps: {warmup_steps})")
        else:
            # Fallback if T_max cannot be determined (e.g., dataloader not ready or empty)
            fallback_batches_per_epoch = 1000 # A reasonable guess
            T_max = num_epochs_for_scheduler * fallback_batches_per_epoch
            print(f"Warning: batches_per_epoch not available for CosineAnnealingLR T_max calculation. "
                  f"Using fallback T_max: {T_max} (based on {fallback_batches_per_epoch} batches/epoch). "
                  f"Consider setting 'lr_scheduler_T_max' directly in config if this is not intended.")

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max,
            eta_min=scheduler_config.get("lr_scheduler_eta_min", 0)
        )
        print(f"CosineAnnealingLR scheduler initialized with T_max={T_max}, eta_min={scheduler_config.get('lr_scheduler_eta_min', 0)}.")
    else:
        print(f"Unsupported scheduler_type: {scheduler_type}. No scheduler will be used.")
        return None
    return scheduler

def initialize_training_components(model_config_to_use: dict, optim_sched_config_to_use: dict, 
                                   device: torch.device, batches_per_epoch_for_scheduler: int = None
                                   ) -> tuple[ByteLLM_GrugV3, torch.optim.Optimizer, nn.Module, torch.optim.lr_scheduler._LRScheduler | None]:
    """Initializes model, optimizer, criterion, and scheduler."""
    model = ByteLLM_GrugV3(model_config_to_use).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total model parameters: {total_params:,}")
    print(f"Trainable model parameters: {trainable_params:,}")

    if hasattr(torch, 'compile') and model_config_to_use.get("use_torch_compile", False):
        print("Attempting to compile the GrugV3 model with torch.compile()...")
        try:
            # Common modes: "default", "reduce-overhead", "max-autotune"
            # model = torch.compile(model, mode="reduce-overhead") 
            model = torch.compile(model) # Default mode
            print("Model compiled successfully with torch.compile().")
        except Exception as e:
            print(f"torch.compile() failed: {e}. Proceeding with uncompiled model.")

    optimizer = initialize_optimizer(model, optim_sched_config_to_use)
    scheduler = initialize_scheduler(optimizer, optim_sched_config_to_use, batches_per_epoch_for_scheduler)
    criterion = nn.CrossEntropyLoss() # Standard for classification-like tasks (next byte prediction)
    
    return model, optimizer, criterion, scheduler

# --- Training and Prediction Orchestration Functions ---

def perform_training(current_run_config: dict, model: ByteLLM_GrugV3, 
                     train_dataloader: torch.utils.data.DataLoader, 
                     val_dataloader: torch.utils.data.DataLoader | None, 
                     optimizer: torch.optim.Optimizer, criterion: nn.Module, 
                     scheduler: torch.optim.lr_scheduler._LRScheduler | None, device: torch.device):
    """Orchestrates the training phase."""
    if not current_run_config.get("DO_TRAINING", True):
        print("\n--- Skipping Training Phase (DO_TRAINING set to False) ---")
        return

    print("\n--- GrugV3 Training Phase ---")
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_dir=current_run_config["checkpoint_dir"],
        model_name=current_run_config["model_name"],
        scheduler=scheduler,
        train_config=current_run_config # Pass the full current run config as train_config
    )
    try:
        trainer.train(num_epochs=current_run_config["num_epochs"])
    except Exception as e:
        print(f"An error occurred during GrugV3 training: {e}")
        traceback.print_exc()

def perform_prediction_scenarios(current_run_config: dict, device: torch.device):
    """Orchestrates the prediction/generation phase using the best model."""
    if not current_run_config.get("DO_PREDICTION", True):
        print("\n--- Skipping Prediction Phase (DO_PREDICTION set to False) ---")
        return

    print("\n--- GrugV3 Prediction/Generation (using best model) ---")
    best_ckpt_path = Path(current_run_config["checkpoint_dir"]) / f"{current_run_config['model_name']}_best.pth"

    if not best_ckpt_path.exists():
        print(f"No best model checkpoint ({best_ckpt_path}) found for model '{current_run_config['model_name']}'. "
              "Skipping prediction. Train a model first or ensure the path is correct.")
        return

    try:
        print(f"Loading best GrugV3 model for prediction from: {best_ckpt_path}")
        # map_location ensures model loads to the correct device specified for prediction
        ckpt = torch.load(best_ckpt_path, map_location=device)

        # Crucial: Use the configuration stored *within the checkpoint* to rebuild the model
        loaded_model_config_from_ckpt = ckpt.get('config') 
        if not loaded_model_config_from_ckpt:
            print("ERROR: Checkpoint does not contain its 'config' (model architecture). Cannot reliably perform prediction.")
            print("Attempting to fall back to current run's config (HIGHLY RISKY if architecture changed).")
            loaded_model_config_from_ckpt = current_run_config.copy() # Risky fallback
        
        # Ensure all necessary keys are present for model init, potentially merging with current if some are missing
        # This merge is a safeguard, ideally checkpoint 'config' is complete
        required_model_keys = {"vocab_size", "embedding_dim", "attention_d_model", "max_positional_encoding_len", "num_attention_layers", "attention_num_heads"}
        for key in required_model_keys:
            if key not in loaded_model_config_from_ckpt:
                print(f"Warning: Key '{key}' not found in checkpoint's model config. Using value from current run config: {current_run_config.get(key)}")
                loaded_model_config_from_ckpt[key] = current_run_config.get(key)


        predictor_model = ByteLLM_GrugV3(loaded_model_config_from_ckpt).to(device)
        predictor_model.load_state_dict(ckpt['model_state_dict'])
        print("Best GrugV3 model weights loaded successfully for prediction.")

        # Generation parameters for the Predictor instance
        # Prefer parameters from the checkpoint's train_config if available, else current_run_config
        ckpt_train_config = ckpt.get('train_config', {})
        generation_params_for_predictor = {
            "generation_temperature": ckpt_train_config.get("generation_temperature", current_run_config.get("generation_temperature", 1.0)),
            "generation_top_k": ckpt_train_config.get("generation_top_k", current_run_config.get("generation_top_k", 0))
        }
        # Model internal config for predictor, taken from the loaded model's actual config
        model_internals_for_predictor = {
            "max_positional_encoding_len": loaded_model_config_from_ckpt.get("max_positional_encoding_len"),
            "sequence_length": loaded_model_config_from_ckpt.get("sequence_length"), # Training context length
        }

        predictor = Predictor(predictor_model, device, generation_params_for_predictor, model_internals_for_predictor)

        seeds_to_try = {
            "Philosophical": "The meaning of life is",
            "Technical": "Multihead attention mechanism is",
            "Narrative Start": "Once upon a time, in a land of bytes,",
            "Code Snippet": "import torch\nclass MySimpleModel(torch.nn.Module):"
        }

        for seed_name, seed_text in seeds_to_try.items():
            seed_bytes = seed_text.encode('utf-8')
            print(f"\nSeed ({seed_name}): '{seed_text}' (Length: {len(seed_bytes)} bytes)")
            generated_bytes = predictor.generate_sequence(seed_bytes, length=150) # Generate 150 new bytes
            try:
                full_text = generated_bytes.decode('utf-8', errors='replace') # Includes seed + generated
                print(f"Full Text (Seed + Generated):\n---\n{full_text}\n---")
            except UnicodeDecodeError as ude:
                print(f"Could not decode generated sequence: {ude}. Raw bytes: {generated_bytes}")

    except Exception as e:
        print(f"An error occurred during the GrugV3 prediction phase: {e}")
        traceback.print_exc()

# --- Main Orchestration ---
def main():
    """
    Main function to orchestrate the GrugV3 model training and prediction.
    """
    # Use a mutable copy of the global config for this run
    current_run_config = CONFIG_V3.copy() 
    
    # Anomaly detection can be helpful during development, but has a performance overhead.
    # Disable it if profiler is enabled, as they can interact.
    if not current_run_config.get("enable_profiler", False):
        # torch.autograd.set_detect_anomaly(True)
        # print("INFO: PyTorch autograd anomaly detection is ENABLED.")
        pass # Disabled by default now; can be enabled for debugging.
    
    try:
        device = setup_environment(current_run_config)

        print("\n--- GrugV3 Data Loading and Processing ---")
        train_dataloader, val_dataloader = load_data_components(current_run_config) # Can modify current_run_config

        batches_per_epoch = None
        if train_dataloader and len(train_dataloader) > 0:
            batches_per_epoch = len(train_dataloader)
        elif train_dataloader is None or len(train_dataloader) == 0:
            print("Warning: Training DataLoader is empty or None. Training might not occur or scheduler may be misconfigured.")
            # If training is intended, this is likely an issue with data or batch_size.

        print("\n--- GrugV3 Model and Training Components Initialization ---")
        # Ensure use_torch_compile key exists, defaulting to False if not in config
        if "use_torch_compile" not in current_run_config: 
            current_run_config["use_torch_compile"] = False 

        model, optimizer, criterion, scheduler = initialize_training_components(
            model_config_to_use=current_run_config, 
            optim_sched_config_to_use=current_run_config, 
            device=device,
            batches_per_epoch_for_scheduler=batches_per_epoch
        )

        # --- Perform Training ---
        perform_training(current_run_config, model, train_dataloader, val_dataloader, 
                         optimizer, criterion, scheduler, device)
        
        # --- Perform Prediction (using the best model saved during training) ---
        perform_prediction_scenarios(current_run_config, device)

    except ValueError as ve:
        print(f"Configuration or Value Error in GrugV3 main: {ve}")
        traceback.print_exc()
    except FileNotFoundError as fnfe:
        print(f"File Not Found Error in GrugV3 main: {fnfe}")
        traceback.print_exc()
    except ImportError as ie:
        print(f"Import Error in GrugV3 main: {ie}. Ensure all modules are in PYTHONPATH.")
        traceback.print_exc()
    except RuntimeError as rte: # PyTorch specific runtime errors
        print(f"PyTorch Runtime Error in GrugV3 main: {rte}")
        if "CUDA out of memory" in str(rte):
            print("Hint: This is a CUDA Out of Memory error. Try reducing 'batch_size', 'sequence_length', "
                  "or model dimensions (e.g., 'embedding_dim', 'attention_d_model', 'num_attention_layers'). "
                  "Using AMP (current_run_config['use_amp']=True) might also help if not already enabled.")
        if "NaN" in str(rte) or "nan" in str(rte) or "Inf" in str(rte) or "inf" in str(rte):
            print("Hint: A RuntimeError involving NaN or Inf occurred. This often indicates numerical instability. "
                  "Check gradients (e.g., enable anomaly detection for debugging), learning rate, "
                  "model architecture (especially normalizations), or data scaling/preprocessing.")
        traceback.print_exc()
    except Exception as e:
        print(f"An unexpected critical error occurred in GrugV3 main execution: {e}")
        traceback.print_exc()
    finally:
        print("\nGrugV3 script finished.")

if __name__ == "__main__":
    main()
