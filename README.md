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
- **Embedding Layer:** Maps bytes to dense vectors.
- **CNN Frontend (Optional):** (`model_components.py`) Captures local patterns from embeddings.
- **Learnable Positional Encoding:** (`model_components.py`) Adds positional context.
- **Multi-Head Attention Layers:** (`model.py`) Stacked Transformer encoders.
- **Output Layer:** Predicts next-byte logits (0-255).

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
MIT License. See `LICENSE.md`.
