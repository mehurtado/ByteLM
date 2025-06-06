import os
import glob
from collections import Counter
import chardet # For more robust encoding detection, if desired (optional)

# --- Configuration ---
# This should match the DATA_DIR in your main.py
DATA_DIR = "./dataset/USE_processed"
EXPECTED_ENCODING = 'utf-8' # The encoding your model/tokenizer expects
TOP_N_COMMON_BYTES = 20
TOP_N_LEAST_COMMON_BYTES = 20
LINE_LENGTH_SAMPLE_SIZE = 1000 # Number of lines to sample for average length calculation

def analyze_text_data(data_dir):
    """
    Analyzes text data in the specified directory, including identifying missing bytes.
    """
    print(f"--- Starting Data Analysis for Directory: {data_dir} ---")

    txt_files = glob.glob(os.path.join(data_dir, "*.txt"))

    if not txt_files:
        print(f"No .txt files found in directory: {data_dir}")
        print("Please ensure DATA_DIR is correctly set and contains .txt files.")
        return

    # --- Overall Statistics ---
    total_files = len(txt_files)
    total_characters = 0 # For byte-level, this is total bytes
    total_bytes_all_files = [] # For overall byte frequency
    empty_files_count = 0
    encoding_errors = {} # file_path: error_message
    file_sizes = []

    # --- Per-File Analysis & Aggregation ---
    print(f"\n--- Processing {total_files} files... ---")
    for i, filepath in enumerate(txt_files):
        if (i + 1) % 10 == 0 or (i+1) == total_files: # Print progress update
            print(f"Processing file {i+1}/{total_files}: {os.path.basename(filepath)}")
        try:
            with open(filepath, 'rb') as f_byte: # Read as bytes first
                raw_content_bytes = f_byte.read()

            file_size = len(raw_content_bytes)
            file_sizes.append(file_size)

            if file_size == 0:
                empty_files_count += 1
                continue

            total_characters += file_size
            total_bytes_all_files.extend(list(raw_content_bytes))

            try:
                _ = raw_content_bytes.decode(EXPECTED_ENCODING)
            except UnicodeDecodeError as e:
                encoding_errors[filepath] = str(e)
        except Exception as e:
            print(f"Error processing file {filepath}: {e}")
            encoding_errors[filepath] = f"File read error: {e}"

    print("\n--- Overall Dataset Statistics ---")
    print(f"Total .txt files found: {total_files}")
    if not file_sizes and empty_files_count == total_files:
        print("All files are empty or no files found to analyze further.")
        return
    if not file_sizes and total_files > 0 :
         print(f"Warning: All {empty_files_count} processed files were empty.")
         # Fall through to allow empty file count to be reported if some files were processed but all were empty.

    print(f"Total bytes (characters): {total_characters:,}")
    print(f"Empty files found: {empty_files_count}")

    if file_sizes: # Only calculate these if there were non-empty files
        avg_file_size = sum(file_sizes) / len(file_sizes)
        print(f"Average file size (non-empty): {avg_file_size:,.2f} bytes")
        print(f"Smallest file size (non-empty): {min(file_sizes):,} bytes")
        print(f"Largest file size (non-empty): {max(file_sizes):,} bytes")
    else:
        print("No non-empty files to report size statistics for.")


    if encoding_errors:
        print(f"\n--- Encoding Issues ({len(encoding_errors)} files) ---")
        print(f"Found {len(encoding_errors)} files that could not be decoded as {EXPECTED_ENCODING}:")
        for f_path, err_msg in list(encoding_errors.items())[:10]:
            print(f"  - {os.path.basename(f_path)}: {err_msg}")
        if len(encoding_errors) > 10:
            print(f"  ... and {len(encoding_errors) - 10} more.")

    if not total_bytes_all_files:
        print("No data from non-empty files to perform byte frequency or further analysis.")
        return

    # --- Byte Frequency Analysis ---
    print("\n--- Byte Frequency Analysis (Overall Dataset) ---")
    byte_counts = Counter(total_bytes_all_files)
    unique_bytes_present = len(byte_counts)
    print(f"Total unique bytes found: {unique_bytes_present} (out of 256 possible)")

    print(f"\nTop {TOP_N_COMMON_BYTES} Most Common Bytes:")
    for byte_val, count in byte_counts.most_common(TOP_N_COMMON_BYTES):
        char_repr = chr(byte_val) if 32 <= byte_val <= 126 else f"0x{byte_val:02x}"
        if byte_val == 10: char_repr = "\\n (Newline)"
        if byte_val == 13: char_repr = "\\r (Carriage Return)"
        if byte_val == 9: char_repr = "\\t (Tab)"
        print(f"  Byte {byte_val:3d} ('{char_repr}'): {count:,} times ({count/total_characters*100:.2f}%)")

    print(f"\nTop {TOP_N_LEAST_COMMON_BYTES} Least Common Bytes (among those present):")
    least_common = sorted(byte_counts.items(), key=lambda item: item[1])
    for byte_val, count in least_common[:TOP_N_LEAST_COMMON_BYTES]:
        char_repr = chr(byte_val) if 32 <= byte_val <= 126 else f"0x{byte_val:02x}"
        if byte_val == 10: char_repr = "\\n (Newline)"
        if byte_val == 13: char_repr = "\\r (Carriage Return)"
        if byte_val == 9: char_repr = "\\t (Tab)"
        print(f"  Byte {byte_val:3d} ('{char_repr}'): {count:,} times ({count/total_characters*100:.2f}%)")

    # --- Missing Byte Identification (NEWLY INTEGRATED) ---
    missing_bytes_list = []
    for byte_val_check in range(256):
        if byte_val_check not in byte_counts: # Check against keys of byte_counts
            missing_bytes_list.append(byte_val_check)

    if not missing_bytes_list:
        print("\n--- All 256 byte values (0-255) are present in the dataset! ---")
    else:
        print(f"\n--- Found {len(missing_bytes_list)} Missing Byte Values (0-255) ---")
        print("Int | Hex  | Char | Description")
        print("----|------|------|------------------------------------")
        for byte_val in missing_bytes_list:
            hex_val = f"0x{byte_val:02x}"
            char_repr = ""
            description = ""
            control_chars_map = {
                0: ("NUL", "Null"), 1: ("SOH", "Start of Heading"), 2: ("STX", "Start of Text"),
                3: ("ETX", "End of Text"), 4: ("EOT", "End of Transmission"), 5: ("ENQ", "Enquiry"),
                6: ("ACK", "Acknowledge"), 7: ("BEL", "Bell"), 8: ("BS", "Backspace"),
                9: ("HT", "Horizontal Tab"), 10: ("LF", "Line Feed (Newline)"), 11: ("VT", "Vertical Tab"),
                12: ("FF", "Form Feed"), 13: ("CR", "Carriage Return"), 14: ("SO", "Shift Out"),
                15: ("SI", "Shift In"), 16: ("DLE", "Data Link Escape"), 17: ("DC1", "Device Control 1 (XON)"),
                18: ("DC2", "Device Control 2"), 19: ("DC3", "Device Control 3 (XOFF)"), 20: ("DC4", "Device Control 4"),
                21: ("NAK", "Negative Acknowledge"), 22: ("SYN", "Synchronous Idle"), 23: ("ETB", "End of Transmission Block"),
                24: ("CAN", "Cancel"), 25: ("EM", "End of Medium"), 26: ("SUB", "Substitute"),
                27: ("ESC", "Escape"), 28: ("FS", "File Separator"), 29: ("GS", "Group Separator"),
                30: ("RS", "Record Separator"), 31: ("US", "Unit Separator")
            }
            if 0 <= byte_val <= 31:
                char_repr, description = control_chars_map.get(byte_val, (f"Ctrl-{chr(byte_val + 64)}", "Control Character"))
            elif 32 <= byte_val <= 126:
                char_repr = chr(byte_val)
                description = "Printable ASCII"
            elif byte_val == 127:
                char_repr = "DEL"
                description = "Delete"
            else:
                try:
                    char_repr = bytes([byte_val]).decode('latin-1') # Show Latin-1 representation
                except UnicodeDecodeError:
                    char_repr = "n/a"
                description = "Non-ASCII / Extended"
            print(f"{byte_val:<3} | {hex_val:<4} | {char_repr:<4} | {description}")

    # --- Character Type Analysis ---
    print("\n--- Character Type Analysis (Overall Dataset) ---")
    ascii_printable_count = 0
    non_ascii_count = 0
    control_char_count = 0
    whitespace_count = 0
    common_whitespace_bytes = {ord(' '), ord('\t'), ord('\n'), ord('\r'), ord('\f'), ord('\v')}

    for byte_val in total_bytes_all_files:
        if 32 <= byte_val <= 126: ascii_printable_count += 1
        if byte_val > 127: non_ascii_count += 1
        if (byte_val < 32 and byte_val not in common_whitespace_bytes) or byte_val == 127: control_char_count += 1
        if byte_val in common_whitespace_bytes: whitespace_count +=1

    print(f"ASCII Printable (32-126): {ascii_printable_count:,} ({ascii_printable_count/total_characters*100:.2f}%)")
    print(f"Non-ASCII (>127): {non_ascii_count:,} ({non_ascii_count/total_characters*100:.2f}%)")
    print(f"Other Control Characters (0-31 excl. common whitespace, 127): {control_char_count:,} ({control_char_count/total_characters*100:.2f}%)")
    print(f"Total Whitespace (space, \\t, \\n, \\r, \\f, \\v): {whitespace_count:,} ({whitespace_count/total_characters*100:.2f}%)")

    # --- Line Length Analysis (Sampled) ---
    print("\n--- Line Length Analysis (Sampled) ---")
    line_lengths = []
    total_lines_sampled = 0
    # Sample from a subset of files to speed up, prioritize larger files if possible or just first N
    files_for_line_sampling = sorted(txt_files, key=lambda x: os.path.getsize(x) if os.path.exists(x) else 0, reverse=True)
    files_for_line_sampling = files_for_line_sampling[:max(10, total_files // 10)]


    for filepath in files_for_line_sampling:
        if total_lines_sampled >= LINE_LENGTH_SAMPLE_SIZE: break
        try:
            with open(filepath, 'r', encoding=EXPECTED_ENCODING, errors='ignore') as f:
                for line in f:
                    if total_lines_sampled >= LINE_LENGTH_SAMPLE_SIZE: break
                    line_lengths.append(len(line.rstrip('\n\r')))
                    total_lines_sampled += 1
        except Exception:
            pass

    if line_lengths:
        avg_line_length = sum(line_lengths) / len(line_lengths)
        max_line_length = max(line_lengths)
        min_line_length = min(line_lengths)
        print(f"Analyzed {total_lines_sampled:,} lines from up to {len(files_for_line_sampling)} files (sampled).")
        print(f"Average line length: {avg_line_length:.2f} characters")
        print(f"Maximum line length: {max_line_length:,} characters")
        print(f"Minimum line length: {min_line_length:,} characters")
    else:
        print("Could not sample lines for length analysis.")

    print("\n--- Analysis Complete ---")

if __name__ == '__main__':
    analyze_text_data(DATA_DIR)
