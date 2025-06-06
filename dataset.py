# dataset.py

import glob
import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from pathlib import Path
import tempfile # Added for robust test directory management
import time # Added for cache validation testing (modifying timestamps)

# Assuming utils.py is in the same directory or accessible in the Python path
from utils import ensure_dir, generate_dummy_data # Import necessary functions

class ByteSequenceDataset(Dataset):
    """
    A PyTorch Dataset for serving sequences of bytes from chunked data.
    """
    def __init__(self, chunk_manifest: list[dict], processed_data_dir: Path, sequence_length: int, stride: int):
        """
        Initializes the dataset with chunked data.

        Args:
            chunk_manifest (list[dict]): A list of dictionaries, where each dict contains
                                         metadata for a chunk (e.g., 'chunk_file', 'num_bytes').
            processed_data_dir (Path): The directory where chunk .npy files are stored.
            sequence_length (int): The length of the input sequences to be generated.
            stride (int): The step size between the start of consecutive sequences.
        """
        if not isinstance(chunk_manifest, list):
            raise TypeError(f"chunk_manifest must be a list, got {type(chunk_manifest)}")
        if not isinstance(processed_data_dir, Path):
            raise TypeError(f"processed_data_dir must be a Path object, got {type(processed_data_dir)}")
        if not isinstance(sequence_length, int) or sequence_length <= 0:
            raise ValueError(f"sequence_length must be a positive integer, got {sequence_length}")
        if not isinstance(stride, int) or stride <= 0:
            raise ValueError(f"stride must be a positive integer, got {stride}")

        self.chunk_manifest = chunk_manifest
        self.processed_data_dir = processed_data_dir
        self.sequence_length = sequence_length
        self.stride = stride
        self.cumulative_sequences: list[int] = []
        self.total_num_sequences = 0

        if not self.chunk_manifest:
            print("Warning: ByteSequenceDataset initialized with an empty chunk_manifest.")
            return

        current_cumulative_sum = 0
        for chunk_info in self.chunk_manifest:
            if 'num_bytes' not in chunk_info:
                raise ValueError(f"Chunk info {chunk_info.get('chunk_file', 'Unknown chunk')} is missing 'num_bytes'.")

            num_bytes_in_chunk = chunk_info['num_bytes']
            if num_bytes_in_chunk >= self.sequence_length:
                num_sequences_in_chunk = (num_bytes_in_chunk - self.sequence_length) // self.stride + 1
            else:
                num_sequences_in_chunk = 0

            chunk_info['num_sequences'] = num_sequences_in_chunk
            current_cumulative_sum += num_sequences_in_chunk
            self.cumulative_sequences.append(current_cumulative_sum)

        if self.cumulative_sequences:
            self.total_num_sequences = self.cumulative_sequences[-1]

        if self.total_num_sequences == 0:
            print(
                f"Warning: Total number of sequences is 0. "
                f"This might be due to all chunks being shorter than sequence_length ({self.sequence_length}), "
                f"or an empty/invalid chunk_manifest."
            )


    def __len__(self) -> int:
        """
        Returns the total number of sequences in the dataset across all chunks.
        """
        return self.total_num_sequences

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a single input sequence and its corresponding target byte from the appropriate chunk.
        """
        if not (0 <= idx < self.total_num_sequences):
            raise IndexError(f"Index {idx} out of range for dataset of length {self.total_num_sequences}")

        chunk_index = np.searchsorted(self.cumulative_sequences, idx, side='right')

        if chunk_index >= len(self.chunk_manifest):
             raise IndexError(f"Calculated chunk_index {chunk_index} is out of bounds for chunk_manifest length {len(self.chunk_manifest)} with idx {idx}.")

        if chunk_index == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_sequences[chunk_index - 1]

        chunk_info = self.chunk_manifest[chunk_index]
        chunk_filename = chunk_info['chunk_file']
        chunk_file_path = self.processed_data_dir / chunk_filename

        try:
            current_chunk_data = np.load(chunk_file_path, mmap_mode='r')
        except FileNotFoundError:
            raise FileNotFoundError(f"Chunk file {chunk_file_path} not found for index {idx}.")
        except Exception as e:
            raise IOError(f"Error loading chunk file {chunk_file_path} for index {idx}: {e}")

        if not (0 <= local_idx < chunk_info['num_sequences']):
             raise IndexError(
                f"Calculated local_idx {local_idx} is out of range for chunk {chunk_filename} "
                f"(num_sequences: {chunk_info['num_sequences']}) for global idx {idx} and chunk_idx {chunk_index}."
                f"Cumulative sequences: {self.cumulative_sequences}"
             )

        start_pos = local_idx * self.stride
        end_pos = start_pos + self.sequence_length
        target_pos = end_pos # Target is the byte immediately following the input sequence

        # Boundary check for target_pos
        if target_pos >= chunk_info['num_bytes']:
            # This should ideally not happen if num_sequences_in_chunk is calculated correctly
            # and local_idx is within bounds. But as a safeguard:
            raise IndexError(
                f"Calculated target_pos {target_pos} is out of bounds for chunk {chunk_filename} "
                f"(num_bytes: {chunk_info['num_bytes']}) with local_idx {local_idx}, stride {self.stride}, seq_len {self.sequence_length}."
            )

        input_sequence_np = current_chunk_data[start_pos : end_pos].copy()
        # The target is the single byte at target_pos, which is end_pos
        # However, the original code implies the target is the byte *after* the sequence.
        # If stride can be > 1,  local_idx + self.sequence_length might not be the correct target
        # if we are skipping bytes.
        # The new logic: target is simply the byte at `start_pos + self.sequence_length` (which is `end_pos`)
        # Let's stick to the original intention: target is the byte immediately following the sequence,
        # so current_chunk_data[end_pos] or current_chunk_data[start_pos + self.sequence_length]
        target_np = current_chunk_data[target_pos].copy()


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
        self.data_dir = Path(data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.sequence_length = sequence_length
        self.force_reprocess = force_reprocess
        self.config_for_data_gen = config_for_data_gen
        ensure_dir(self.processed_data_dir)
        ensure_dir(self.data_dir) # Ensure data_dir also exists

    def load_or_create_chunk_manifest(self) -> list[dict]:
        manifest_path = self.processed_data_dir / "chunks_manifest.json"
        if not self.force_reprocess and manifest_path.exists():
            print(f"Found cached chunk manifest {manifest_path.name}. Validating...")
            try:
                with open(manifest_path, 'r') as f:
                    chunk_metadata_list = json.load(f)
                all_chunks_exist = True
                if not chunk_metadata_list:
                    print("Manifest is empty. Will proceed to (re)processing.")
                else:
                    for chunk_info in chunk_metadata_list:
                        chunk_file_path = self.processed_data_dir / chunk_info["chunk_file"]
                        if not chunk_file_path.exists():
                            print(f"Cache invalid: Chunk file {chunk_file_path} listed in manifest not found. Reprocessing...")
                            all_chunks_exist = False
                            break
                if all_chunks_exist:
                    print(f"Cache valid: Successfully loaded manifest with {len(chunk_metadata_list)} chunks. All chunk files verified.")
                    return chunk_metadata_list
            except json.JSONDecodeError as e:
                print(f"Cache invalid: Error decoding JSON from {manifest_path}: {e}. Reprocessing...")
            except Exception as e:
                print(f"Cache invalid: Error loading or verifying manifest {manifest_path}: {e}. Reprocessing...")

        print("No valid cache found or reprocessing forced. Creating chunks...")
        return self._create_chunked_byte_arrays()

    def _create_chunked_byte_arrays(self) -> list[dict]:
        TARGET_CHUNK_SIZE_BYTES = 100 * 1024 * 1024  # 100MB (can be made smaller for testing)
        # For tests, allow overriding TARGET_CHUNK_SIZE_BYTES via config_for_data_gen
        if self.config_for_data_gen and 'target_chunk_size_bytes_override' in self.config_for_data_gen:
            TARGET_CHUNK_SIZE_BYTES = self.config_for_data_gen['target_chunk_size_bytes_override']
            print(f"Overriding TARGET_CHUNK_SIZE_BYTES to {TARGET_CHUNK_SIZE_BYTES} for testing.")

        chunk_metadata_list = []
        accumulated_text_content = ""
        chunk_index = 0
        print(f"Processing text files from {self.data_dir} to create chunked .npy files (target chunk size: {TARGET_CHUNK_SIZE_BYTES} bytes)...")
        text_files = sorted(list(self.data_dir.glob("*.txt")))

        if not text_files:
            print(f"No .txt files found in {self.data_dir}.")
            if self.config_for_data_gen and self.config_for_data_gen.get("generate_dummy_data_if_empty", False):
                print("Attempting to generate dummy data...")
                try:
                    generate_dummy_data(str(self.data_dir), self.config_for_data_gen)
                    text_files = sorted(list(self.data_dir.glob("*.txt")))
                    if not text_files:
                        print(f"Still no .txt files found in {self.data_dir} after dummy data generation attempt.")
                except Exception as e:
                    print(f"Error during dummy data generation: {e}")
            if not text_files:
                print("No .txt files available. No chunks will be created.")
                manifest_path = self.processed_data_dir / "chunks_manifest.json"
                with open(manifest_path, 'w') as f:
                    json.dump([], f, indent=4)
                print(f"Saved empty manifest to {manifest_path}")
                return []
        print(f"Found {len(text_files)} .txt files. Reading content and creating chunks...")

        def save_current_chunk(text_to_save: str, current_chunk_idx: int) -> tuple[dict | None, int]:
            if not text_to_save.strip(): return None, current_chunk_idx
            encoded_bytes = text_to_save.encode('utf-8', errors='replace')
            if not encoded_bytes: return None, current_chunk_idx
            chunk_np_array = np.array(list(encoded_bytes), dtype=np.uint8)
            chunk_filename = f"chunk_{current_chunk_idx:04d}.npy"
            chunk_filepath = self.processed_data_dir / chunk_filename
            np.save(chunk_filepath, chunk_np_array)
            metadata = {"chunk_file": chunk_filename, "num_bytes": len(encoded_bytes)}
            print(f"Saved chunk {chunk_filename} (Bytes: {len(encoded_bytes):,})")
            return metadata, current_chunk_idx + 1

        for file_path in text_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: file_content = f.read()
                current_accumulated_bytes = len(accumulated_text_content.encode('utf-8', errors='replace'))
                new_file_bytes_estimate = len(file_content.encode('utf-8', errors='replace'))
                if accumulated_text_content and (current_accumulated_bytes + new_file_bytes_estimate > TARGET_CHUNK_SIZE_BYTES):
                    metadata, chunk_index = save_current_chunk(accumulated_text_content, chunk_index)
                    if metadata: chunk_metadata_list.append(metadata)
                    accumulated_text_content = ""
                accumulated_text_content += file_content
                while len(accumulated_text_content.encode('utf-8', errors='replace')) >= TARGET_CHUNK_SIZE_BYTES:
                    text_for_this_chunk, remainder_text = "", ""
                    current_chunk_actual_bytes, split_char_idx = 0, 0
                    for char_idx, char_val in enumerate(accumulated_text_content):
                        char_bytes_len = len(char_val.encode('utf-8', errors='replace'))
                        if current_chunk_actual_bytes + char_bytes_len > TARGET_CHUNK_SIZE_BYTES:
                            temp_text_before_char = accumulated_text_content[:char_idx]
                            last_newline, last_space = temp_text_before_char.rfind('\n'), temp_text_before_char.rfind(' ')
                            split_pos = char_idx
                            if last_newline != -1 and (char_idx - last_newline) < (len(temp_text_before_char) * 0.2) and len(accumulated_text_content[:last_newline+1].encode('utf-8', errors='replace')) > 0: split_pos = last_newline + 1
                            elif last_space != -1 and (char_idx - last_space) < (len(temp_text_before_char) * 0.2) and len(accumulated_text_content[:last_space+1].encode('utf-8', errors='replace')) > 0: split_pos = last_space + 1
                            text_for_this_chunk, remainder_text = accumulated_text_content[:split_pos], accumulated_text_content[split_pos:]
                            break
                        current_chunk_actual_bytes += char_bytes_len
                    else:
                        if len(accumulated_text_content.encode('utf-8', errors='replace')) >= TARGET_CHUNK_SIZE_BYTES:
                            text_for_this_chunk, remainder_text = accumulated_text_content, ""
                        else: break
                    if text_for_this_chunk:
                        metadata, chunk_index = save_current_chunk(text_for_this_chunk, chunk_index)
                        if metadata: chunk_metadata_list.append(metadata)
                        accumulated_text_content = remainder_text
                    elif not remainder_text and not text_for_this_chunk and accumulated_text_content: break
                    elif not accumulated_text_content: break
            except Exception as e: print(f"Warning: Could not process file {file_path}: {e}")
        if accumulated_text_content:
            metadata, chunk_index = save_current_chunk(accumulated_text_content, chunk_index)
            if metadata: chunk_metadata_list.append(metadata)
        if not chunk_metadata_list and text_files: print("Warning: Text files were found, but no chunks were saved. Check content and chunking logic.")
        manifest_path = self.processed_data_dir / "chunks_manifest.json"
        try:
            with open(manifest_path, 'w') as f: json.dump(chunk_metadata_list, f, indent=4)
            print(f"Saved chunk manifest to {manifest_path} with {len(chunk_metadata_list)} entries.")
        except Exception as e: raise IOError(f"Failed to save chunk manifest {manifest_path}: {e}")
        return chunk_metadata_list

    def get_dataloaders(self, batch_size: int, val_split_ratio: float,
                        num_workers: int, current_sequence_length: int, stride: int) -> tuple[DataLoader, DataLoader | None]:
        chunk_manifest_data = self.load_or_create_chunk_manifest()
        if not chunk_manifest_data:
            raise ValueError("No data chunks found or created. Cannot create dataloaders.")
        full_dataset = ByteSequenceDataset(
            chunk_manifest=chunk_manifest_data,
            processed_data_dir=self.processed_data_dir,
            sequence_length=current_sequence_length,
            stride=stride
        )
        num_total_sequences = len(full_dataset)
        if num_total_sequences == 0:
            raise ValueError(f"The dataset resulted in 0 sequences for sequence_length {current_sequence_length}.")
        if not (0 < val_split_ratio < 1):
            train_indices, val_indices = np.arange(num_total_sequences), np.array([])
        else:
            indices = np.arange(num_total_sequences); np.random.shuffle(indices)
            num_val_sequences = int(val_split_ratio * num_total_sequences)
            if num_val_sequences == 0 or (num_total_sequences - num_val_sequences == 0):
                print(f"Warning: Small dataset size ({num_total_sequences}) or val_split_ratio ({val_split_ratio}) leads to empty train/val set. Using all data for training.")
                train_indices, val_indices = indices, np.array([])
            else:
                train_indices, val_indices = indices[num_val_sequences:], indices[:num_val_sequences]

        train_dataset = Subset(full_dataset, train_indices.tolist())
        val_dataset = Subset(full_dataset, val_indices.tolist()) if len(val_indices) > 0 else None
        print(f"Training set: {len(train_dataset)} sequences. Validation set: {len(val_dataset) if val_dataset else 0} sequences.")
        pin_memory_flag = (num_workers > 0 and torch.cuda.is_available())
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory_flag, drop_last=len(train_dataset) >= batch_size) if len(train_dataset) > 0 else None
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory_flag, drop_last=False) if val_dataset and len(val_dataset) > 0 else None
        return train_dataloader, val_dataloader

    def get_vocab_size(self) -> int: return 256

if __name__ == '__main__':
    print("--- Comprehensive Testing for dataset.py ---")

    # Default test configuration
    default_test_config = {
        "sequence_length": 10,
        "batch_size": 4,
        "val_split_ratio": 0.2,
        "num_workers": 0,
        "generate_dummy_data_if_empty": True,
        "force_reprocess_data": True,
        "num_files": 3,
        "min_size_kb": 1,    # Smaller min size for more varied testing
        "max_size_kb": 5,   # Smaller max size for faster tests, less complex chunking
        "target_chunk_size_bytes_override": 10 * 1024 # 10KB chunks for easier testing of multiple chunks
    }

    def run_test_scenario(scenario_name: str, config_overrides: dict = None):
        print(f"\n--- SCENARIO: {scenario_name} ---")
        current_config = default_test_config.copy()
        if config_overrides:
            current_config.update(config_overrides)

        with tempfile.TemporaryDirectory() as temp_data_dir_str, \
             tempfile.TemporaryDirectory() as temp_processed_dir_str:

            temp_data_dir = Path(temp_data_dir_str)
            temp_processed_dir = Path(temp_processed_dir_str)
            print(f"Using temp_data_dir: {temp_data_dir}")
            print(f"Using temp_processed_dir: {temp_processed_dir}")

            # --- Pre-test setup specific to scenarios ---
            if scenario_name == "Test Reprocessing Logic - Missing Chunk":
                # First, create data normally
                dp_initial = DataProcessor(
                    data_dir=temp_data_dir, processed_data_dir=temp_processed_dir,
                    sequence_length=current_config["sequence_length"], force_reprocess=True,
                    config_for_data_gen=current_config
                )
                initial_manifest = dp_initial.load_or_create_chunk_manifest()
                assert initial_manifest, "Initial manifest creation failed for 'Missing Chunk' scenario."
                # Delete a chunk file
                if initial_manifest:
                    chunk_to_delete = temp_processed_dir / initial_manifest[0]['chunk_file']
                    if chunk_to_delete.exists():
                        chunk_to_delete.unlink()
                        print(f"Deleted chunk {chunk_to_delete} for cache validation test.")
                    else:
                        print(f"Warning: Chunk {initial_manifest[0]['chunk_file']} not found for deletion in 'Missing Chunk' setup.")
                current_config["force_reprocess_data"] = False # Now try to load from cache

            if scenario_name == "Test Reprocessing Logic - Valid Cache":
                dp_initial = DataProcessor(
                    data_dir=temp_data_dir, processed_data_dir=temp_processed_dir,
                    sequence_length=current_config["sequence_length"], force_reprocess=True,
                    config_for_data_gen=current_config
                )
                dp_initial.load_or_create_chunk_manifest() # Create cache
                current_config["force_reprocess_data"] = False # Now try to load
                # To verify cache is used, one might check logs or, if possible, mock _create_chunked_byte_arrays
                # For this test, we'll assume if it runs faster or certain logs don't appear, cache was used.
                # A more direct test is difficult without deeper mocking/instrumentation.
                print("Cache created. Subsequent run should attempt to load from cache.")


            # --- DataProcessor Initialization & Chunk Creation ---
            print("\n1. Testing DataProcessor and Chunk Creation...")
            data_processor = DataProcessor(
                data_dir=temp_data_dir,
                processed_data_dir=temp_processed_dir,
                sequence_length=current_config["sequence_length"],
                force_reprocess=current_config["force_reprocess_data"],
                config_for_data_gen=current_config
            )

            manifest_list = data_processor.load_or_create_chunk_manifest()
            manifest_path = temp_processed_dir / "chunks_manifest.json"

            assert manifest_path.exists(), "chunks_manifest.json was not created."
            with open(manifest_path, 'r') as f:
                loaded_manifest_json = json.load(f)
            assert isinstance(loaded_manifest_json, list), "Manifest is not a list."
            assert len(manifest_list) == len(loaded_manifest_json), "Manifest length mismatch."

            if current_config.get("generate_dummy_data_if_empty", False) and not list(temp_data_dir.glob("*.txt")):
                 # If dummy data was supposed to be generated but no txt files exist, something is wrong with dummy data gen
                 if current_config["num_files"] > 0 and current_config["max_size_kb"] > 0 : # only if expected to generate
                    # This case should ideally be caught by generate_dummy_data itself or result in empty manifest
                    pass # `_create_chunked_byte_arrays` handles empty text_files list

            expected_min_chunks = 0
            if list(temp_data_dir.glob("*.txt")): # If there are source files
                total_source_bytes = sum(p.stat().st_size for p in temp_data_dir.glob("*.txt"))
                if total_source_bytes > 0 : # only expect chunks if there's data
                    expected_min_chunks = 1
                    # A more precise check for number of chunks based on total_source_bytes and TARGET_CHUNK_SIZE_BYTES could be added.

            if expected_min_chunks > 0 :
                assert len(manifest_list) >= expected_min_chunks, f"Expected at least {expected_min_chunks} chunk(s), got {len(manifest_list)}."
            else:
                assert len(manifest_list) == 0, f"Expected 0 chunks, got {len(manifest_list)}."


            # --- Ground Truth Data Preparation ---
            all_source_text_bytes_list = []
            for txt_file in sorted(list(temp_data_dir.glob("*.txt"))):
                with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                    all_source_text_bytes_list.append(f.read().encode('utf-8', errors='replace'))
            ground_truth_bytes = b"".join(all_source_text_bytes_list)
            ground_truth_np = np.array(list(ground_truth_bytes), dtype=np.uint8) if ground_truth_bytes else np.array([], dtype=np.uint8)


            # --- Manifest and Chunk File Verification ---
            concatenated_chunk_bytes_list = []
            for i, chunk_info in enumerate(manifest_list):
                assert "chunk_file" in chunk_info, f"Manifest entry {i} missing 'chunk_file'."
                assert "num_bytes" in chunk_info, f"Manifest entry {i} missing 'num_bytes'."
                chunk_file = temp_processed_dir / chunk_info["chunk_file"]
                assert chunk_file.exists(), f"Chunk file {chunk_info['chunk_file']} not found."
                loaded_chunk_data = np.load(chunk_file)
                assert loaded_chunk_data.dtype == np.uint8, f"Chunk {chunk_info['chunk_file']} dtype is not uint8."
                assert len(loaded_chunk_data) == chunk_info["num_bytes"], \
                    f"Num_bytes mismatch for {chunk_info['chunk_file']}: manifest says {chunk_info['num_bytes']}, file has {len(loaded_chunk_data)}."
                concatenated_chunk_bytes_list.append(loaded_chunk_data)

            if manifest_list: # Only if chunks were created
                all_chunks_np = np.concatenate(concatenated_chunk_bytes_list)
                assert np.array_equal(all_chunks_np, ground_truth_np), "Concatenated chunk data does not match ground truth source data."
            elif len(ground_truth_np) > 0 : # Manifest is empty, but there was source data
                 assert False, "Manifest is empty, but ground truth data was expected. Chunking might have failed."


            # --- ByteSequenceDataset Testing ---
            print("\n2. Testing ByteSequenceDataset...")
            dataset_seq_len = current_config["sequence_length"]
            bsd = ByteSequenceDataset(manifest_list, temp_processed_dir, dataset_seq_len)

            expected_total_sequences = 0
            for chunk_data_info in manifest_list:
                expected_total_sequences += max(0, chunk_data_info['num_bytes'] - dataset_seq_len)

            print(f"Expected total sequences: {expected_total_sequences}, Got from dataset __len__: {len(bsd)}")
            assert len(bsd) == expected_total_sequences, "ByteSequenceDataset __len__ is incorrect."

            if len(bsd) > 0:
                indices_to_test = sorted(list(set([0, len(bsd) // 2, len(bsd) - 1]))) # Test first, middle, last valid indices
                for global_idx in indices_to_test:
                    input_tensor, target_tensor = bsd[global_idx]
                    assert input_tensor.shape == (dataset_seq_len,), f"Input tensor shape incorrect for idx {global_idx}."
                    assert isinstance(target_tensor.item(), int), f"Target tensor is not a scalar int for idx {global_idx}."

                    # Verify content
                    expected_input_np = ground_truth_np[global_idx : global_idx + dataset_seq_len]
                    expected_target_val = ground_truth_np[global_idx + dataset_seq_len]

                    assert np.array_equal(input_tensor.numpy(), expected_input_np), \
                        f"Content mismatch for input sequence at global_idx {global_idx}."
                    assert target_tensor.item() == expected_target_val, \
                        f"Content mismatch for target value at global_idx {global_idx}."
                    print(f"  Successfully verified item at global_idx {global_idx}")
            else:
                print("Dataset is empty (0 sequences), skipping __getitem__ content tests.")
                try:
                    bsd[0] # Should raise IndexError
                    assert False, "Accessing item [0] in empty dataset should raise IndexError."
                except IndexError:
                    print("  Correctly got IndexError for item [0] in empty dataset.")


            # --- DataLoaders Testing ---
            print("\n3. Testing DataProcessor.get_dataloaders...")
            if len(bsd) > 0 : # Only if there are sequences to load
                train_dl, val_dl = data_processor.get_dataloaders(
                    batch_size=current_config["batch_size"],
                    val_split_ratio=current_config["val_split_ratio"],
                    num_workers=current_config["num_workers"],
                    current_sequence_length=dataset_seq_len
                )
                if train_dl:
                    print(f"  Train DataLoader: {len(train_dl)} batches.")
                    for i, (inputs, targets) in enumerate(train_dl):
                        assert inputs.shape[0] <= current_config["batch_size"] # Can be less if drop_last=False
                        assert inputs.shape[1] == dataset_seq_len
                        assert targets.shape[0] == inputs.shape[0]
                        if i >= 1: break # Check a couple of batches
                if val_dl:
                    print(f"  Validation DataLoader: {len(val_dl)} batches.")
                    for i, (inputs, targets) in enumerate(val_dl):
                        assert inputs.shape[0] <= current_config["batch_size"]
                        assert inputs.shape[1] == dataset_seq_len
                        assert targets.shape[0] == inputs.shape[0]
                        if i >= 1: break
            else:
                print("  Skipping DataLoader test as dataset has 0 sequences.")
                try:
                    data_processor.get_dataloaders(
                        batch_size=current_config["batch_size"], val_split_ratio=current_config["val_split_ratio"],
                        num_workers=current_config["num_workers"], current_sequence_length=dataset_seq_len
                    )
                    # This should raise ValueError if num_total_sequences is 0
                    assert False, "get_dataloaders should raise ValueError for empty dataset"
                except ValueError as e:
                     print(f"  Correctly caught ValueError for empty dataset: {e}")

            print(f"--- SCENARIO {scenario_name} COMPLETED ---")

    # --- Run Test Scenarios ---
    run_test_scenario("Default Processing - Dummy Data")

    run_test_scenario("Force Reprocess - With Pre-existing Dummy Data", {
        "force_reprocess_data": True,
        "generate_dummy_data_if_empty": False # Turn off generation to use manually created ones
    }) # For this one, we'd ideally create files in temp_data_dir before DataProcessor init

    run_test_scenario("Test Reprocessing Logic - Missing Chunk") # Setup handled within the function

    run_test_scenario("Test Reprocessing Logic - Valid Cache") # Setup handled within the function

    run_test_scenario("Edge Case - Sequence Length Larger Than Chunks", {
        "sequence_length": 20 * 1024, # 20KB, larger than max_size_kb for dummy files
        "target_chunk_size_bytes_override": 5 * 1024 # Ensure chunks are small
    })

    run_test_scenario("Edge Case - No Text Files, No Dummy Data Gen", {
        "generate_dummy_data_if_empty": False,
        "num_files": 0 # Ensure no dummy files are generated by default either
    })

    run_test_scenario("Edge Case - Very Small Files, Small Chunks", {
        "num_files": 5, "min_size_kb": 0, "max_size_kb": 1, # Files might be only a few bytes
        "sequence_length": 5,
        "target_chunk_size_bytes_override": 512 # Tiny chunks
    })

    print("\n--- All dataset.py tests completed ---")
    # tempfile directories are automatically cleaned up on exiting the 'with' block
