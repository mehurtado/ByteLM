# config.py

# --- Configuration for GrugV3 ---
CONFIG_V3 = {
    # Data and General Settings
    "data_dir": "./dataset/USE",
    "processed_data_dir": "./dataset/USE_processed",
    "checkpoint_dir": "./checkpoints_grug_v3",
    "model_name": "grug_v3_parallel_stream", # New model name
    "resume_from_checkpoint": "./checkpoints_grug_v3/grug_v3_parallel_stream_best.pth", # Set to a path to resume, e.g., "./checkpoints_grug_v3/grug_v3_parallel_stream_best.pth"
    "sequence_length": 256,
    "batch_size": 8,
    "vocab_size": 256,
    "val_split_ratio": 0.1,
    "num_workers": 0,
    "generate_dummy_data_if_empty": True,
    "force_reprocess_data": False, # Set to True to force reprocessing on next run
    "data_stride": 64,

    # Embedding Layer
    "embedding_dim": 512,

    # --- ARCHITECTURE SWITCH ---
    "use_parallel_stream_model": True, # MASTER SWITCH for the new architecture

    # --- Original CNN Frontend (can be used by either model if desired) ---
    "use_cnn_frontend": False, # Best to keep this off to avoid confusion with the aggregator
    "cnn_out_channels_list": [1024, 1024],
    "cnn_kernel_sizes": [9, 3],
    "cnn_stride": 1,
    "cnn_padding_mode": "zeros",
    "cnn_activation": "GELU",
    "cnn_dropout": 0.2,
    "cnn_use_layernorm": True,

    # --- Parallel Stream Settings (used if use_parallel_stream_model is True) ---
    "num_byte_stream_layers": 4,      # Attention layers for the byte-level path
    "num_agg_stream_layers": 4,       # Attention layers for the aggregated-byte path
    "agg_cnn_kernel_size": 4,         # Kernel size for the aggregator CNN
    "agg_cnn_stride": 4,              # Stride for the aggregator CNN
    "agg_cnn_out_dim": 1024,          # Output dim of aggregator, should match attention_d_model

    # --- Original Single-Stream Transformer Settings ---
    # (used if use_parallel_stream_model is False)
    "num_attention_layers": 4,

    # --- Common Component Settings ---
    "max_positional_encoding_len": 4096,
    "pe_dropout": 0.3,
    "attention_d_model": 1024,
    "attention_num_heads": 16,
    "attention_dropout": 0.2,
    "ffn_dim_multiply": 4,
    "transformer_dropout": 0.2,
    "transformer_norm_first": False, # Set to True for Pre-LN Transformers

    # --- Output Layer ---
    "output_dropout": 0.2,

    # --- Training Parameters ---
    "num_epochs": 500,
    "learning_rate": 5e-5,
    "optimizer_type": "AdamW",
    "adam_beta1": 0.9,
    "adam_beta2": 0.98,
    "adam_eps": 1e-9,
    "weight_decay": 0.01,
    "scheduler_type": "ReduceLROnPlateau",
    "lr_scheduler_T_max": 50 * 1000,
    "lr_scheduler_eta_min": 1e-6,
    "lr_scheduler_patience": 5,
    "lr_scheduler_factor": 0.1,
    "clip_grad_norm_value": 1.0,
    "print_every": 1000,
    "test_every_batches": 5000,
    "validate_and_checkpoint_best_every_batches": 20000,
    "reset_best_val_loss_on_resume": False,
    "use_lr_warmup": True,
    "lr_warmup_steps": 20000,
    "lr_warmup_init_factor": 0.01,
    "use_amp": True,

    # --- Generation / Prediction Settings ---
    "generation_temperature": 1.0,
    "generation_top_k": 50,
    "interim_test_temperature": 0.3,
    "interim_test_top_k": 20,

    # --- Profiling Settings ---
    "enable_profiler": False,
    "profiler_log_dir": "./profiler_logs_grug_v3",
    # ... (other profiler settings) ...

    # --- Main script flow control ---
    "DO_TRAINING": True,
    "DO_PREDICTION": True,

    # --- Backend Settings ---
    "cudnn_benchmark": True,
    "use_torch_compile": False
}