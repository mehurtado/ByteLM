# utils.py

import glob
from pathlib import Path

def ensure_dir(directory_path: str | Path):
    """
    Ensures that a directory exists, creating it if necessary.

    Args:
        directory_path (str or Path): The path to the directory.
    """
    Path(directory_path).mkdir(parents=True, exist_ok=True)

def generate_dummy_data(data_dir: str | Path, config_dict: dict, num_files: int = 5, lines_per_file: int = 10000):
    """
    Generates dummy text files in the specified directory if it's empty and
    the configuration allows it.

    Args:
        data_dir (str or Path): The directory to generate dummy data in.
        config_dict (dict): Configuration dictionary, checked for "generate_dummy_data_if_empty".
        num_files (int): Number of dummy files to create.
        lines_per_file (int): Number of lines per dummy file.
    """
    data_dir_path = Path(data_dir)
    ensure_dir(data_dir_path) # Ensure the directory itself exists

    if not config_dict.get("generate_dummy_data_if_empty", False):
        print("Dummy data generation is turned OFF in the configuration.")
        return

    # Check if there are any .txt files in the directory
    existing_files = list(data_dir_path.glob('*.txt'))
    if not existing_files:
        print(f"Generating dummy data in {data_dir_path}...")
        for i in range(num_files):
            file_path = data_dir_path / f"dummy_data_{i}.txt"
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    for j in range(lines_per_file):
                        # Create somewhat varied lines to better simulate real text
                        line_content = (
                            f"This is line {j+1} of GrugV3 dummy file {i+1}. "
                            f"The quick brown fox jumps over the lazy dog. 0123456789. "
                            f"Special characters: áéíóúñüçßøå. "
                            f"Repeating sequence for length: {j%10} " * 3 # Ensure lines have some length
                        )
                        f.write(line_content.strip() + "\n")
                print(f"Generated {file_path}")
            except IOError as e:
                print(f"Error writing dummy file {file_path}: {e}")
        print("Dummy data generation complete.")
    else:
        print(f"Directory {data_dir_path} already contains .txt files ({len(existing_files)} found). Skipping dummy data generation.")

if __name__ == '__main__':
    # Example usage (for testing this module directly)
    print("Testing utils.py...")
    
    # Create a dummy config for testing
    dummy_config_for_utils_test = {
        "generate_dummy_data_if_empty": True,
        # Add other keys if your functions depend on them, though these two don't directly
    }
    
    test_data_dir = Path("./temp_test_data_dir_utils")
    
    print(f"\n1. Testing ensure_dir for {test_data_dir}")
    ensure_dir(test_data_dir)
    if test_data_dir.exists() and test_data_dir.is_dir():
        print(f"ensure_dir PASSED: Directory {test_data_dir} created or already exists.")
    else:
        print(f"ensure_dir FAILED: Directory {test_data_dir} was not created.")

    print(f"\n2. Testing generate_dummy_data in {test_data_dir}")
    generate_dummy_data(test_data_dir, dummy_config_for_utils_test, num_files=2, lines_per_file=5)
    
    # Verify files were created
    created_files = list(test_data_dir.glob('*.txt'))
    if len(created_files) == 2:
        print(f"generate_dummy_data PASSED: Found {len(created_files)} dummy files.")
        # Basic content check for the first file
        try:
            with open(created_files[0], 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if len(lines) == 5:
                    print(f"Content check for {created_files[0]} PASSED: Correct number of lines.")
                else:
                    print(f"Content check for {created_files[0]} FAILED: Expected 5 lines, got {len(lines)}.")
        except Exception as e:
            print(f"Error reading dummy file for verification: {e}")
            
    else:
        print(f"generate_dummy_data FAILED: Expected 2 dummy files, found {len(created_files)}.")

    print(f"\n3. Testing generate_dummy_data again (should skip as files exist)")
    generate_dummy_data(test_data_dir, dummy_config_for_utils_test, num_files=2, lines_per_file=5)


    # Clean up the test directory
    print(f"\nCleaning up test directory: {test_data_dir}")
    try:
        for item in test_data_dir.glob('*'):
            item.unlink() # Remove files
        test_data_dir.rmdir() # Remove directory
        print("Cleanup successful.")
    except Exception as e:
        print(f"Error during cleanup: {e}")
        
    print("\nutils.py testing finished.")
