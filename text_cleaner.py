# text_cleaner.py
import argparse
import re
import unicodedata
from pathlib import Path

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

# Pre-compile regex patterns for efficiency
# Matches most C0 and C1 control chars, excluding tab, LF, CR
# \x00-\x08: NUL to BS
# \x0B-\x0C: VT, FF
# \x0E-\x1F: SO to US
# \x7F: DEL
# \x80-\x9F: C1 control characters
CONTROL_CHARS_PATTERN = re.compile(
    r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F\x80-\x9F]"
)
# Matches multiple spaces or tabs
MULTI_SPACE_TAB_PATTERN = re.compile(r"[ \t]+")
# Matches sequences of 3 or more newlines
EXCESSIVE_NEWLINES_PATTERN = re.compile(r"\n{3,}")

def strip_html_content(text_content):
    """Removes HTML tags from text using BeautifulSoup."""
    if not BS4_AVAILABLE:
        print("Warning: BeautifulSoup4 library not found. HTML stripping will be skipped.")
        return text_content
    soup = BeautifulSoup(text_content, "lxml")
    # Get text, and use .join to handle multiple text elements gracefully
    stripped_text = "".join(soup.find_all(string=True))
    return stripped_text

def normalize_unicode_text(text_content, form='NFC'):
    """Normalizes Unicode text to a specified form (e.g., NFC, NFKC)."""
    return unicodedata.normalize(form, text_content)

def remove_control_characters(text_content):
    """Removes most control characters, preserving tab, newline, carriage return."""
    return CONTROL_CHARS_PATTERN.sub("", text_content)

def normalize_whitespace_and_lines(text_content, min_line_length=0):
    """
    Normalizes whitespace, newlines, and filters short/empty lines.
    - Converts \r\n and \r to \n
    - Replaces multiple spaces/tabs with a single space
    - Strips leading/trailing whitespace from each line
    - Reduces 3+ consecutive newlines to 2
    - Removes lines shorter than min_line_length (after stripping)
    - Removes lines that become empty after stripping, unless they contribute to a double newline.
    """
    # Normalize newline characters
    text_content = text_content.replace("\r\n", "\n").replace("\r", "\n")
    
    lines = text_content.split('\n')
    cleaned_lines = []
    for line in lines:
        # Replace multiple spaces/tabs with a single space
        line = MULTI_SPACE_TAB_PATTERN.sub(" ", line)
        # Strip leading/trailing whitespace
        line = line.strip()
        
        if len(line) >= min_line_length:
            cleaned_lines.append(line)
        elif line == "" and cleaned_lines and cleaned_lines[-1] != "": 
            # Allow an empty line if the previous line was not empty (for paragraph spacing)
            # This helps in the next step of reducing excessive newlines
            cleaned_lines.append(line) 
            
    # Join lines back, then handle excessive newlines
    processed_text = "\n".join(cleaned_lines)
    # Reduce 3 or more newlines to 2 newlines
    processed_text = EXCESSIVE_NEWLINES_PATTERN.sub("\n\n", processed_text)
    # Final strip to remove any leading/trailing newlines from the whole document if desired
    # For now, let's keep single leading/trailing newlines if they result from processing
    return processed_text


def clean_text_file(input_file_path, output_file_path,
                    strip_html=False, unicode_form='NFC', 
                    min_line_length=0):
    """Reads, cleans, and writes a single text file."""
    try:
        print(f"Processing: {input_file_path}")
        text = input_file_path.read_text(encoding='utf-8', errors='replace')

        if strip_html:
            text = strip_html_content(text)
        
        text = normalize_unicode_text(text, form=unicode_form)
        text = remove_control_characters(text)
        text = normalize_whitespace_and_lines(text, min_line_length=min_line_length)
        
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        output_file_path.write_text(text, encoding='utf-8')
        print(f"Cleaned content saved to: {output_file_path}")

    except Exception as e:
        print(f"Error processing file {input_file_path}: {e}")
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Clean text files in a directory.")
    parser.add_argument("input_dir", type=str, help="Directory containing input .txt files.")
    parser.add_argument("output_dir", type=str, help="Directory to save cleaned .txt files.")
    parser.add_argument("--strip_html", action="store_true", help="Remove HTML tags from text.")
    parser.add_argument("--unicode_form", type=str, default="NFC", choices=['NFC', 'NFKC', 'NFD', 'NFKD'],
                        help="Unicode normalization form (default: NFC).")
    parser.add_argument("--min_line_length", type=int, default=5,
                        help="Minimum character length for a line to be kept after cleaning (default: 5). Set to 0 to disable.")
    parser.add_argument("--file_extension", type=str, default=".txt",
                        help="File extension of text files to process (default: .txt).")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        print(f"Error: Input directory '{input_dir}' not found.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting text cleaning process...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Strip HTML: {args.strip_html}")
    print(f"Unicode Form: {args.unicode_form}")
    print(f"Min Line Length: {args.min_line_length}")
    print(f"File Extension: '{args.file_extension}'")
    print("-" * 30)

    file_count = 0
    for input_file_path in input_dir.rglob(f"*{args.file_extension}"): # rglob for recursive search
        if input_file_path.is_file():
            # Maintain subdirectory structure if any
            relative_path = input_file_path.relative_to(input_dir)
            output_file_path = output_dir / relative_path
            
            clean_text_file(
                input_file_path,
                output_file_path,
                strip_html=args.strip_html,
                unicode_form=args.unicode_form,
                min_line_length=args.min_line_length
            )
            file_count += 1
    
    if file_count == 0:
        print(f"No files with extension '{args.file_extension}' found in '{input_dir}'.")
    else:
        print(f"\nSuccessfully processed {file_count} file(s).")
    print("Cleaning process finished.")

if __name__ == "__main__":
    if not BS4_AVAILABLE:
        print("Note: BeautifulSoup4 library is not available. HTML stripping (--strip_html) will be skipped if enabled.")
        print("You can install it with: pip install beautifulsoup4 lxml")
    main()
