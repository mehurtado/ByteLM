# config.py

# --- Configuration for GrugV3 ---
CONFIG_V3 = {
    # Data and General Settings
    "data_dir": "./dataset/USE",
    "processed_data_dir": "./dataset/USE_processed",
    "checkpoint_dir": "./checkpoints_grug_v3",
    "model_name": "grug_v3_cnn_attention",
    "resume_from_checkpoint": "./checkpoints_grug_v3/grug_v3_cnn_attention_best.pth",
    "sequence_length": 256,
    "batch_size": 8,
    "vocab_size": 256,
    "val_split_ratio": 0.1,
    "num_workers": 0,
    "generate_dummy_data_if_empty": True,
    "force_reprocess_data": True,
    "data_stride": 256,

    # Embedding Layer
    "embedding_dim": 512,

    # CNN Frontend (Optional)
    "use_cnn_frontend": True,
    "cnn_out_channels_list": [1024, 1024],
    "cnn_kernel_sizes": [9, 3],
    "cnn_stride": 1,
    "cnn_padding_mode": "zeros",
    "cnn_activation": "GELU",
    "cnn_dropout": 0.2,
    "cnn_use_layernorm": True,

    # Learnable Positional Encoding
    "max_positional_encoding_len": 4096,
    "pe_dropout": 0.3,

    # Transformer Encoder Block Parameters
    "attention_d_model": 1024,
    "attention_num_heads": 16,
    "attention_dropout": 0.2, # Dropout within MultiHeadAttention
    "num_attention_layers": 4,
    "ffn_dim_multiply": 4,
    "transformer_dropout": 0.2, # Dropout for FFN and after MHA in TransformerEncoderLayer

    # Output Layer
    "output_dropout": 0.2,

    # Training Parameters
    "num_epochs": 50,
    "learning_rate": 1e-6,
    "optimizer_type": "AdamW",
    "adam_beta1": 0.9,
    "adam_beta2": 0.98,
    "adam_eps": 1e-9,
    "weight_decay": 0.01,
    "scheduler_type": "ReduceLROnPlateau",
    "lr_scheduler_T_max": 50 * 1000, # Placeholder, might be recalculated in main
    "lr_scheduler_eta_min": 1e-6,
    "lr_scheduler_patience": 5, # For ReduceLROnPlateau
    "lr_scheduler_factor": 0.1,  # For ReduceLROnPlateau
    "clip_grad_norm_value": 1.0,
    "print_every": 1000,
    "test_every_batches": 5000,
    "validate_and_checkpoint_best_every_batches": 20000, # Run validation every N batches. 0 or -1 to disable, runs at epoch end only.
    "reset_best_val_loss_on_resume": False,

    # Learning Rate Warmup
    "use_lr_warmup": True,
    "lr_warmup_steps": 20000,
    "lr_warmup_init_factor": 0.01,

    # Automatic Mixed Precision (AMP)
    "use_amp": True,

    # Generation / Prediction Settings
    "generation_temperature": 1.0,
    "generation_top_k": 50,
    "interim_test_temperature": 0.3,
    "interim_test_top_k": 20,

    # Profiling Settings
    "enable_profiler": False,
    "profiler_log_dir": "./profiler_logs_grug_v3",
    "profile_epoch_target": 0,
    "profiler_schedule_wait": 5,
    "profiler_schedule_warmup": 5,
    "profiler_schedule_active": 10,
    "profiler_schedule_repeat": 1,

    # Main script flow control
    "DO_TRAINING": True,
    "DO_PREDICTION": True,

    # CuDNN Benchmarking
    "cudnn_benchmark": True,
    "use_torch_compile": False # Flag to enable torch.compile
}
