# GrugV3 Byte-Level Language Model

## Overview
GrugV3 is a byte-level language model built with PyTorch. It is designed for tasks such as text generation and research into sequence modeling at the byte level. The model processes raw byte sequences from text data, allowing it to handle any type of text or character encoding.

## Current Status
**Important Note:** The GrugV3 model is currently under active development and training. It is not yet stable, and its performance may vary significantly. Use with caution, and expect changes.

## Features
*   **Byte-Level Processing:** Operates directly on UTF-8 byte sequences, making it agnostic to vocabulary and character sets.
*   **Configurable Architecture:** Key aspects of the model (embedding dimensions, CNN layers, attention heads, etc.) are configurable.
*   **CNN Frontend (Optional):** Can utilize a Convolutional Neural Network (CNN) frontend to process input sequences before the attention layers.
*   **Multi-Head Attention:** Employs multi-head self-attention mechanisms for learning contextual representations.
*   **Learnable Positional Encodings:** Uses learnable embeddings to represent token positions.
*   **Comprehensive Training Script (`grug_v3.py`):**
    *   Checkpointing: Saves and resumes training, storing the best model.
    *   Learning Rate Scheduling: Supports schedulers like Cosine Annealing and ReduceLROnPlateau.
    *   Learning Rate Warmup.
    *   Automatic Mixed Precision (AMP): Option for faster training on compatible GPUs.
    *   Data Processing: Includes robust data loading and preprocessing from text files.
    *   Dummy Data Generation: Can generate dummy data for quick testing if no dataset is provided.
    *   Profiling: Integrated PyTorch profiler support.
*   **Prediction/Generation:** Script includes functionality to generate text sequences using a trained model.

## Architecture
The `ByteLLM_GrugV3` model primarily consists of the following components:
1.  **Embedding Layer:** Maps input bytes to dense vector representations.
2.  **CNN Frontend (Optional):** A series of 1D convolutional layers that can process the embedded sequences to capture local patterns.
3.  **Learnable Positional Encoding:** Adds positional information to the sequences.
4.  **Multi-Head Attention Layers:** A stack of multi-head self-attention layers to learn long-range dependencies.
5.  **Output Layer:** A fully connected layer that projects the final representation to logits over the vocabulary (byte values 0-255).

The model predicts the next byte in a sequence given the preceding bytes.

## Dataset
GrugV3 is trained on sequences of bytes derived from text files.
*   The `DataProcessor` class in `grug_v3.py` handles the reading of `.txt` files from a specified directory.
*   It concatenates their content, encodes the text into a single stream of UTF-8 bytes, and then creates a memory-mapped NumPy array (`all_bytes_grug_v3.npy`) for efficient loading.
*   The `ByteSequenceDataset` class then serves sequences of a defined length from this byte stream for training and validation.

## Configuration (`CONFIG_V3`)
The behavior of the model, training process, and data handling is extensively controlled by the `CONFIG_V3` dictionary found at the beginning of the `grug_v3.py` script.

Key configuration categories include:
*   **Data and General Settings:** Paths for data, checkpoints, sequence length, batch size, etc.
*   **Model Architecture:** Embedding dimensions, CNN parameters (if used), attention mechanism details (number of heads, layers, dropout), positional encoding settings.
*   **Training Parameters:** Number of epochs, learning rate, optimizer settings, scheduler settings, gradient clipping, AMP usage.
*   **Generation Settings:** Temperature and top-k for sampling during text generation.
*   **Profiling Settings:** Options to enable and configure the PyTorch profiler.

Users are encouraged to inspect and modify the `CONFIG_V3` dictionary in `grug_v3.py` to suit their specific needs and dataset.

## Setup and Usage

### Prerequisites
*   Python 3.x
*   PyTorch (`torch`)
*   NumPy (`numpy`)

You can typically install the required libraries using pip:
```bash
pip install torch numpy
```

### Data Preparation
1.  Place your training text files (UTF-8 encoded `.txt` files) into the directory specified by `"data_dir"` in `CONFIG_V3` (default: `./dataset/USE`).
2.  If the data directory is empty and `"generate_dummy_data_if_empty"` is `True` (default), the script will automatically generate some dummy text files for testing purposes.
3.  The first time you run the script, the `DataProcessor` will process these files and save a `all_bytes_grug_v3.npy` file in the `"processed_data_dir"` (default: `./dataset/USE_processed`). Subsequent runs will load this processed file. Set `"force_reprocess_data": True` to re-process from raw .txt files.

### Running the Script
Execute the main script from your terminal:
```bash
python3 grug_v3.py
```

### Training
*   To start training, ensure `DO_TRAINING = True` in the `CONFIG_V3` dictionary (default).
*   The script will initialize the model, load data, and begin the training loop.
*   Progress, including loss and learning rate, will be printed to the console.
*   Checkpoints are saved in the directory specified by `"checkpoint_dir"` (default: `./checkpoints_grug_v3`). The best performing model (based on validation loss) will be saved as `[model_name]_best.pth`.

### Prediction/Generation
*   To run text generation, ensure `DO_PREDICTION = True` in `CONFIG_V3` (default).
*   This typically runs after the training phase (if both are enabled).
*   It will load the best saved checkpoint (`[model_name]_best.pth`) and generate sample text sequences based on predefined or configurable seeds.
*   The generation parameters (temperature, top-k) can be adjusted in `CONFIG_V3`.

## Key Script Components
*   **`grug_v3.py`**: The main Python script containing all the code for the model, data handling, training, and prediction.
*   **`CONFIG_V3`**: A Python dictionary at the beginning of `grug_v3.py` that holds all configurable parameters.
*   **`ByteLLM_GrugV3`**: The PyTorch `nn.Module` class defining the language model architecture.
*   **`DataProcessor`**: Class responsible for finding, reading, and preprocessing text files into a byte stream suitable for training.
*   **`ByteSequenceDataset`**: PyTorch `Dataset` class that provides training and validation samples (input byte sequence and target byte).
*   **`Trainer`**: Class that encapsulates the model training loop, including optimization, checkpointing, validation, and learning rate adjustments.
*   **`Predictor`**: Class used for generating new byte sequences from a trained model.

## Contributing
Contributions to GrugV3 are welcome! If you find any issues, have suggestions for improvements, or want to add new features, please feel free to:
1.  Open an issue on the project's repository.
2.  Fork the repository, make your changes, and submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE.md) file for details.
