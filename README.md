# GrugV3 Byte-Level Language Model

## Overview
GrugV3 is a PyTorch-based byte-level language model for text generation and sequence modeling. It processes raw UTF-8 byte sequences, handling diverse inputs.
**Status:** Actively under development; not yet stable.

## Core Features
- **Byte-Level Processing:** Operates directly on UTF-8 bytes.
- **Modular Architecture:** Key parameters defined in `config.py`.
- **Optional CNN Frontend:** 1D CNNs for initial sequence processing.
- **Transformer Core:** Multi-head self-attention and learnable positional encodings.
- **Comprehensive Tooling:** Includes checkpointing, LR scheduling, AMP, robust data handling, dummy data generation, profiler support, and text generation, all orchestrated by `main.py`.

## Architecture
The ByteLLM_GrugV3 model, defined in `model.py`, supports two distinct architectural configurations, selectable via the `use_parallel_stream_model` setting in `config.py`.

**1. Single-Stream Architecture (Default):**
This is the original architecture and is used when `use_parallel_stream_model` is `False`.
- **Embedding Layer:** Maps input byte tokens (0-255) to dense embedding vectors.
- **CNN Frontend (Optional):** If `use_cnn_frontend` is enabled in the config, the embeddings are first processed by a configurable 1D CNN layer (`CNNFrontend` from `model_components.py`) to capture local patterns.
- **Input Projection:** The (potentially CNN-processed) embeddings are linearly projected to match the `attention_d_model` dimension if different from `embedding_dim`.
- **Positional Encoding:** Sinusoidal positional encodings (`SinusoidalPositionalEncoding` from `model_components.py`) are added to inject sequence order information.
- **Transformer Encoder Core:** A stack of standard Transformer encoder layers (`nn.TransformerEncoderLayer`) processes the sequence. Each layer consists of multi-head self-attention and a feed-forward network.
- **Output Layer:** The output representation from the final Transformer layer (specifically, the representation of the last token) is passed through a LayerNorm, a dropout layer, and finally a linear layer to predict the logits for the next byte in the sequence (vocabulary size of 256).

**2. Parallel-Stream Architecture (New):**
Activated when `use_parallel_stream_model` is `True` in the config, this architecture processes input through two parallel streams which are then combined.
- **Common Input Processing:**
    - **Embedding Layer:** Same as the single-stream.
    - **CNN Frontend (Optional):** Same as the single-stream, applied before splitting into parallel paths.

- **Path 1: Byte Stream (Fine-grained):**
    - **Projection:** The (potentially CNN-processed) embeddings are linearly projected to `attention_d_model`.
    - **Positional Encoding:** Sinusoidal positional encodings are applied.
    - **Transformer Encoders:** A dedicated stack of Transformer encoder layers (`byte_stream_encoder`) processes this sequence, focusing on fine-grained byte-level details.

- **Path 2: Aggregated Stream (Coarse-grained):**
    - **CNN Aggregation:** The (potentially CNN-processed) embeddings are passed through an `aggregator_cnn` (a 1D Convolutional layer). This CNN typically uses a stride greater than 1, effectively downsampling the sequence and creating aggregated features over local windows. Its output dimension is `agg_cnn_out_dim`.
    - **Positional Encoding:** Sinusoidal positional encodings are applied to this aggregated, shorter sequence.
    - **Transformer Encoders:** Another dedicated stack of Transformer encoder layers (`agg_stream_encoder`) processes this coarser representation.

- **Combination and Output:**
    - The final hidden state (last token representation) from both the Byte Stream and the Aggregated Stream are concatenated.
    - This combined representation is then passed through a LayerNorm (`final_norm`), an output dropout layer, and finally a linear layer (`fc_out`) to predict the next-byte logits.

This dual-stream approach allows the model to simultaneously learn from both fine-grained byte sequences and more abstract, aggregated features from the input.

## Dataset and Data Preparation
- `DataProcessor` (in `dataset.py`): Reads `.txt` files from ``data_dir``, concatenates, UTF-8 encodes, and saves as ``all_bytes_grug_v3.npy`` in ``processed_data_dir``.
- `ByteSequenceDataset` (in `dataset.py`): Serves sequences for training/validation.
- Place UTF-8 `.txt` training files in ``data_dir`` (default: `./dataset/USE`).
- Dummy data is generated if ``data_dir`` is empty and ``generate_dummy_data_if_empty`` is `True` in `config.py`.
- Data is processed to ``all_bytes_grug_v3.npy`` on first run. Set ``force_reprocess_data``: `True` in `config.py` to rebuild.

## Configuration
All settings are in the `CONFIG_V3` dictionary in `config.py` (paths, model hyperparameters, training, generation, profiling).

**Running:**
1. Prepare your data as described in "Dataset and Data Preparation".
2. Execute: `python3 main.py`
3. Control training/prediction via `DO_TRAINING` and `DO_PREDICTION` in `config.py`.
4. Checkpoints are in `checkpoint_dir` (default: `./checkpoints_grug_v3`); best model: `[model_name]_best.pth`.

## Project Structure
- `main.py`: Main script for training and prediction.
- `config.py`: `CONFIG_V3` dictionary for all settings.
- `utils.py`: Utility functions.
- `dataset.py`: `DataProcessor` and `ByteSequenceDataset`.
- `model_components.py`: `LearnablePositionalEncoding`, `CNNFrontend`.
- `model.py`: `ByteLLM_GrugV3` model architecture.
- `predictor.py`: `Predictor` class for generation.
- `trainer.py`: `Trainer` class for training loop.

## Contributing
Contributions welcome via issues or pull requests.

## Collaborators
- Jules

## License
MIT License. See `LICENSE`.
