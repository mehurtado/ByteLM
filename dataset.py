# dataset.py

import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from pathlib import Path

# Assuming utils.py is in the same directory or accessible in the Python path
from utils import ensure_dir, generate_dummy_data # Import necessary functions

class ByteSequenceDataset(Dataset):
    """
    A PyTorch Dataset for serving sequences of bytes.
    """
    def __init__(self, all_bytes_mmap_array: np.ndarray, sequence_length: int):
        """
        Initializes the dataset.

        Args:
            all_bytes_mmap_array (np.ndarray): A memory-mapped NumPy array containing all byte data.
            sequence_length (int): The length of the input sequences to be generated.
        """
        self.all_bytes = all_bytes_mmap_array
        self.sequence_length = sequence_length

        if not isinstance(self.all_bytes, np.ndarray):
            raise TypeError(f"all_bytes_mmap_array must be a NumPy array, got {type(self.all_bytes)}")
        if not isinstance(self.sequence_length, int) or self.sequence_length <= 0:
            raise ValueError(f"sequence_length must be a positive integer, got {self.sequence_length}")

        if len(self.all_bytes) <= self.sequence_length:
            raise ValueError(
                f"Total data length ({len(self.all_bytes)} bytes) is less than or equal to sequence_length "
                f"({self.sequence_length}). Not enough data to create any sequences."
            )
        # The number of possible sequences is len(data) - sequence_length,
        # because each sequence needs a subsequent byte as a target.
        self.num_sequences = len(self.all_bytes) - self.sequence_length

    def __len__(self) -> int:
        """
        Returns the total number of sequences in the dataset.
        """
        return self.num_sequences

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a single input sequence and its corresponding target byte.

        Args:
            idx (int): The index of the sequence to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - input_tensor: The input sequence as a LongTensor.
                - target_tensor: The target byte as a LongTensor.
        """
        if not (0 <= idx < self.num_sequences):
            raise IndexError(f"Index {idx} out of range for dataset of length {self.num_sequences}")

        # Input sequence is from idx to idx + sequence_length
        input_sequence_np = self.all_bytes[idx : idx + self.sequence_length].copy() # .copy() is crucial for mmap
        # Target is the byte immediately following the input sequence
        target_np = self.all_bytes[idx + self.sequence_length].copy()

        input_tensor = torch.tensor(input_sequence_np, dtype=torch.long)
        target_tensor = torch.tensor(target_np, dtype=torch.long)

        return input_tensor, target_tensor

class DataProcessor:
    """
    Handles loading, processing, and preparing data for the model.
    """
    def __init__(self, data_dir: str | Path, processed_data_dir: str | Path,
                 sequence_length: int, force_reprocess: bool = False,
                 config_for_data_gen: dict = None):
        """
        Initializes the DataProcessor.

        Args:
            data_dir (str or Path): Directory containing raw text files.
            processed_data_dir (str or Path): Directory to store/load processed data (e.g., memory-mapped array).
            sequence_length (int): The length of sequences for the ByteSequenceDataset.
            force_reprocess (bool): If True, reprocesses data even if a cached version exists.
            config_for_data_gen (dict, optional): Configuration dictionary passed to generate_dummy_data.
                                                 Required if dummy data generation is needed.
        """
        self.data_dir = Path(data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.sequence_length = sequence_length
        self.force_reprocess = force_reprocess
        self.config_for_data_gen = config_for_data_gen # Store for dummy data generation

        ensure_dir(self.processed_data_dir)
        self.all_bytes_path = self.processed_data_dir / "all_bytes_grug_v3.npy"

    def load_or_create_all_bytes(self) -> np.ndarray:
        """
        Loads the processed byte array from a memory-mapped .npy file if it exists
        and reprocessing is not forced. Otherwise, processes raw text files to create it.

        Returns:
            np.ndarray: A memory-mapped NumPy array of all bytes.
        """
        if not self.force_reprocess and self.all_bytes_path.exists():
            print(f"Loading cached {self.all_bytes_path.name} using memory-mapping...")
            try:
                # 'r' mode opens for reading, but allows mmap to work efficiently
                all_bytes_mmap = np.load(self.all_bytes_path, mmap_mode='r')
                print(f"Successfully memory-mapped (Length: {len(all_bytes_mmap):,} bytes).")
                if len(all_bytes_mmap) == 0:
                    print(f"Warning: Cached file {self.all_bytes_path} is empty. Reprocessing...")
                    return self._create_byte_array_from_text_files()
                return all_bytes_mmap
            except Exception as e:
                print(f"Error memory-mapping {self.all_bytes_path}: {e}. Reprocessing...")
                # Fall through to reprocessing
        
        return self._create_byte_array_from_text_files()

    def _create_byte_array_from_text_files(self) -> np.ndarray:
        """
        Private helper to process text files from data_dir into a single byte array.
        """
        print(f"Processing text files from {self.data_dir} to create {self.all_bytes_path.name}...")
        text_files = sorted(list(self.data_dir.glob("*.txt"))) # Sort for deterministic order

        if not text_files:
            print(f"No .txt files found in {self.data_dir}.")
            if self.config_for_data_gen and self.config_for_data_gen.get("generate_dummy_data_if_empty", False):
                print("Attempting to generate dummy data...")
                generate_dummy_data(str(self.data_dir), self.config_for_data_gen)
                text_files = sorted(list(self.data_dir.glob("*.txt")))
                if not text_files:
                    raise FileNotFoundError(f"Still no .txt files found in {self.data_dir} after dummy data generation attempt.")
            else:
                raise FileNotFoundError(
                    f"No .txt files found in {self.data_dir}. "
                    "Enable 'generate_dummy_data_if_empty' in config or provide data."
                )

        full_text_content_parts = []
        print(f"Found {len(text_files)} .txt files. Reading content...")
        for file_path in text_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    full_text_content_parts.append(f.read())
            except Exception as e:
                print(f"Warning: Could not read file {file_path} as text: {e}")
                continue
        
        if not full_text_content_parts:
            raise ValueError("No text data could be read from the files. Ensure files are UTF-8 encoded or adjust error handling.")

        final_text_string = "".join(full_text_content_parts)
        print(f"Total characters read: {len(final_text_string):,}")

        # Encode the entire string to bytes using UTF-8, replacing errors
        encoded_bytes = final_text_string.encode('utf-8', errors='replace')
        all_bytes_np_array = np.array(list(encoded_bytes), dtype=np.uint8)

        if len(all_bytes_np_array) == 0:
            raise ValueError("Processed data resulted in an empty byte array after encoding. Check input files.")

        try:
            np.save(self.all_bytes_path, all_bytes_np_array)
            print(f"Saved {self.all_bytes_path.name} (Length: {len(all_bytes_np_array):,} bytes). Now loading with memory-mapping...")
            # Load it back as memory-mapped
            all_bytes_mmap = np.load(self.all_bytes_path, mmap_mode='r')
            return all_bytes_mmap
        except Exception as e:
            raise IOError(f"Failed to save or mmap {self.all_bytes_path}: {e}")


    def get_dataloaders(self, batch_size: int, val_split_ratio: float,
                        num_workers: int, current_sequence_length: int) -> tuple[DataLoader, DataLoader | None]:
        """
        Creates and returns training and validation DataLoaders.

        Args:
            batch_size (int): Batch size for the DataLoaders.
            val_split_ratio (float): Fraction of data to use for validation (0.0 to 1.0).
            num_workers (int): Number of worker processes for DataLoader.
            current_sequence_length (int): Sequence length to use for this set of dataloaders.
                                           This overrides the one from init for flexibility if needed.

        Returns:
            tuple[DataLoader, DataLoader | None]: Training DataLoader and Validation DataLoader (or None if no validation split).
        """
        all_bytes_mmap = self.load_or_create_all_bytes()
        
        # Use the provided current_sequence_length for the dataset
        full_dataset = ByteSequenceDataset(all_bytes_mmap, current_sequence_length)
        
        num_total_sequences = len(full_dataset)
        if num_total_sequences == 0:
            raise ValueError("The full dataset resulted in 0 sequences. Cannot create dataloaders. "
                             "This might be due to insufficient data for the given sequence_length.")

        if not (0 < val_split_ratio < 1):
            print(f"Warning: val_split_ratio ({val_split_ratio}) is not strictly between 0 and 1. "
                  "Using all data for training and no validation set.")
            # Create indices for the entire dataset for training
            train_indices = np.arange(num_total_sequences)
            val_indices = np.array([]) # Empty array for validation
        else:
            print(f"Generating and shuffling indices for {num_total_sequences:,} sequences...")
            indices = np.arange(num_total_sequences)
            np.random.shuffle(indices) # Shuffle indices for random split
            print("Indices shuffled.")

            num_val_sequences = int(val_split_ratio * num_total_sequences)
            num_train_sequences = num_total_sequences - num_val_sequences

            if num_val_sequences == 0 or num_train_sequences == 0:
                print(f"Warning: Dataset size ({num_total_sequences:,}) is too small for val_split_ratio ({val_split_ratio}). "
                      "This results in an empty training or validation set. Adjusting to use all data for training.")
                train_indices = indices # Use all data for training
                val_indices = np.array([]) # No validation
            else:
                train_indices = indices[:num_train_sequences]
                val_indices = indices[num_train_sequences:]
        
        train_dataset = Subset(full_dataset, train_indices.tolist())
        val_dataset = Subset(full_dataset, val_indices.tolist()) if len(val_indices) > 0 else None

        print(f"Training set size: {len(train_dataset):,} sequences")
        if val_dataset:
            print(f"Validation set size: {len(val_dataset):,} sequences")
        else:
            print("Validation set size: 0 sequences (or not created)")

        pin_memory_flag = (num_workers > 0 and torch.cuda.is_available())
        
        if len(train_dataset) == 0:
            print("Warning: Training dataset is empty. Cannot create training DataLoader.")
            train_dataloader = None
        elif len(train_dataset) < batch_size:
            print(f"Warning: Training dataset size ({len(train_dataset)}) is less than batch_size ({batch_size}). "
                  "Training DataLoader will have drop_last=False to use all data, but this might be only one batch.")
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                          num_workers=num_workers, pin_memory=pin_memory_flag, drop_last=False)
        else:
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                          num_workers=num_workers, pin_memory=pin_memory_flag, drop_last=True)

        val_dataloader = None
        if val_dataset and len(val_dataset) > 0:
            if len(val_dataset) < batch_size:
                 print(f"Warning: Validation dataset size ({len(val_dataset)}) is less than batch_size ({batch_size}). "
                       "Validation DataLoader will have drop_last=False.")
                 val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                        num_workers=num_workers, pin_memory=pin_memory_flag, drop_last=False)
            else:
                val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                        num_workers=num_workers, pin_memory=pin_memory_flag, drop_last=False) # Typically False for validation
        
        if train_dataloader and len(train_dataloader) == 0 and len(train_dataset) > 0 :
            print(f"Critical Warning: Training DataLoader is empty despite having {len(train_dataset)} samples. "
                  f"Batch size ({batch_size}) might be too large or drop_last=True is problematic.")
        if val_dataloader and len(val_dataloader) == 0 and val_dataset and len(val_dataset) > 0:
            print(f"Critical Warning: Validation DataLoader is empty despite having {len(val_dataset)} samples. "
                  f"Batch size ({batch_size}) might be too large.")
            
        return train_dataloader, val_dataloader

    def get_vocab_size(self) -> int:
        """
        Returns the vocabulary size, which is fixed at 256 for byte-level models.
        """
        return 256

if __name__ == '__main__':
    # This block is for testing the dataset.py module directly.
    print("--- Testing dataset.py ---")

    # Dummy configuration for testing
    test_config = {
        "data_dir": "./temp_test_data_dataset",
        "processed_data_dir": "./temp_test_processed_dataset",
        "sequence_length": 10, # Shorter for faster testing
        "batch_size": 4,
        "val_split_ratio": 0.2,
        "num_workers": 0, # Easier for debugging, set to >0 for actual use if desired
        "generate_dummy_data_if_empty": True, # Enable dummy data generation
        "force_reprocess_data": True # Force reprocessing for consistent test runs
    }

    # 1. Setup test directories
    ensure_dir(test_config["data_dir"])
    ensure_dir(test_config["processed_data_dir"])

    # 2. (Optional) Pre-populate with a tiny dummy file if generate_dummy_data is complex to test here
    # For this test, we rely on DataProcessor's call to generate_dummy_data

    try:
        # 3. Initialize DataProcessor
        print("\nInitializing DataProcessor...")
        data_processor = DataProcessor(
            data_dir=test_config["data_dir"],
            processed_data_dir=test_config["processed_data_dir"],
            sequence_length=test_config["sequence_length"], # This is for ByteSequenceDataset default
            force_reprocess=test_config["force_reprocess_data"],
            config_for_data_gen=test_config # Pass the config for dummy data generation
        )
        print("DataProcessor initialized.")

        # 4. Test load_or_create_all_bytes (which should trigger dummy data generation if dir is empty)
        print("\nTesting load_or_create_all_bytes...")
        all_bytes = data_processor.load_or_create_all_bytes()
        print(f"Loaded/Created all_bytes array with shape: {all_bytes.shape}, dtype: {all_bytes.dtype}")
        assert len(all_bytes) > test_config["sequence_length"], "Not enough data generated/loaded."

        # 5. Test ByteSequenceDataset directly
        print("\nTesting ByteSequenceDataset...")
        # Use a small slice of the loaded bytes for a direct dataset test
        sample_bytes_for_direct_test = all_bytes[:test_config["sequence_length"] + 5]
        direct_dataset = ByteSequenceDataset(sample_bytes_for_direct_test, test_config["sequence_length"])
        print(f"Direct ByteSequenceDataset length: {len(direct_dataset)}")
        assert len(direct_dataset) == 5, "Direct dataset length mismatch."
        input_seq, target_val = direct_dataset[0]
        print(f"First item - Input: {input_seq.tolist()}, Target: {target_val.item()}")
        assert len(input_seq) == test_config["sequence_length"], "Input sequence length mismatch."

        # 6. Test get_dataloaders
        print("\nTesting get_dataloaders...")
        # Use a different sequence length for get_dataloaders to test flexibility
        dataloader_seq_len = 8
        print(f"Requesting DataLoaders with sequence_length: {dataloader_seq_len}")
        train_dl, val_dl = data_processor.get_dataloaders(
            batch_size=test_config["batch_size"],
            val_split_ratio=test_config["val_split_ratio"],
            num_workers=test_config["num_workers"],
            current_sequence_length=dataloader_seq_len # Test with this specific length
        )

        if train_dl:
            print(f"Train DataLoader created. Number of batches: {len(train_dl)}")
            for i, (batch_inputs, batch_targets) in enumerate(train_dl):
                print(f"Train Batch {i+1} - Inputs shape: {batch_inputs.shape}, Targets shape: {batch_targets.shape}")
                assert batch_inputs.shape[1] == dataloader_seq_len, f"Train batch input seq length mismatch. Expected {dataloader_seq_len}, got {batch_inputs.shape[1]}"
                if i >= 1: break # Just check a couple of batches
        else:
            print("Train DataLoader was not created (or is empty).")


        if val_dl:
            print(f"Validation DataLoader created. Number of batches: {len(val_dl)}")
            for i, (batch_inputs, batch_targets) in enumerate(val_dl):
                print(f"Validation Batch {i+1} - Inputs shape: {batch_inputs.shape}, Targets shape: {batch_targets.shape}")
                assert batch_inputs.shape[1] == dataloader_seq_len, f"Validation batch input seq length mismatch. Expected {dataloader_seq_len}, got {batch_inputs.shape[1]}"
                if i >= 1: break
        else:
            print("Validation DataLoader was not created (or is empty).")
        
        print("\n--- dataset.py tests completed successfully (if no assertions failed) ---")

    except Exception as e:
        print(f"\n--- An error occurred during dataset.py testing ---")
        import traceback
        traceback.print_exc()

    finally:
        # Clean up test directories and files
        print("\nCleaning up test directories...")
        processed_file = Path(test_config["processed_data_dir"]) / "all_bytes_grug_v3.npy"
        if processed_file.exists():
            processed_file.unlink()
            print(f"Deleted {processed_file}")

        data_dir_path = Path(test_config["data_dir"])
        for f in data_dir_path.glob("*.txt"):
            f.unlink()
            print(f"Deleted {f}")
        
        if Path(test_config["processed_data_dir"]).exists():
            try: Path(test_config["processed_data_dir"]).rmdir() 
            except OSError: print(f"Could not remove {test_config['processed_data_dir']} (might not be empty if other files were created)")
        if data_dir_path.exists():
            try: data_dir_path.rmdir()
            except OSError: print(f"Could not remove {test_config['data_dir']} (might not be empty)")
        print("Cleanup attempt finished.")
