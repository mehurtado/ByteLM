# text_cleaner.py (Corrected Version)
import argparse
import json
import re
import unicodedata
import traceback
from pathlib import Path

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

# Regex patterns remain the same
CONTROL_CHARS_PATTERN = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F\x80-\x9F]")
MULTI_SPACE_TAB_PATTERN = re.compile(r"[ \t]+")
EXCESSIVE_NEWLINES_PATTERN = re.compile(r"\n{3,}")

# --- NEW HELPER FUNCTION ---
def extract_json_from_html_export(html_content: str) -> str:
    """
    Finds and extracts the JSON string assigned to 'var jsonData' from an HTML file.
    """
    try:
        # Find the start of the JSON data
        start_marker = "var jsonData = "
        start_index = html_content.find(start_marker)
        if start_index == -1:
            return "" # Marker not found

        # The JSON data (as a list) starts with '['
        json_start_index = html_content.find('[', start_index)
        if json_start_index == -1:
            return ""

        # Find the end of the JSON data, which is the last ']' followed by a ';'
        end_index = html_content.rfind('];')
        if end_index == -1:
            return "" # Closing marker not found

        # Extract the pure JSON string (the list itself)
        json_string = html_content[json_start_index : end_index + 1]
        return json_string
    except Exception as e:
        print(f"Error isolating JSON from HTML content: {e}")
        return ""


def strip_html_content(text_content):
    if not BS4_AVAILABLE:
        print("Warning: BeautifulSoup4 library not found. HTML stripping will be skipped.")
        return text_content
    soup = BeautifulSoup(text_content, "lxml")
    return "".join(soup.find_all(string=True))

def normalize_unicode_text(text_content, form='NFC'):
    return unicodedata.normalize(form, text_content)

def remove_control_characters(text_content):
    return CONTROL_CHARS_PATTERN.sub("", text_content)

def normalize_whitespace_and_lines(text_content, min_line_length=0):
    text_content = text_content.replace("\r\n", "\n").replace("\r", "\n")
    lines = text_content.split('\n')
    cleaned_lines = []

    for line in lines:
        line = MULTI_SPACE_TAB_PATTERN.sub(" ", line).strip()
        if len(line) >= min_line_length:
            cleaned_lines.append(line)
        elif line == "" and cleaned_lines and cleaned_lines[-1] != "":
            cleaned_lines.append(line)

    processed_text = "\n".join(cleaned_lines)
    return EXCESSIVE_NEWLINES_PATTERN.sub("\n\n", processed_text)

def clean_and_write_text(text, output_path,
                         strip_html=False, unicode_form='NFC',
                         min_line_length=0):
    try:
        if strip_html:
            text = strip_html_content(text)
        text = normalize_unicode_text(text, form=unicode_form)
        text = remove_control_characters(text)
        text = normalize_whitespace_and_lines(text, min_line_length=min_line_length)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding='utf-8')
        print(f"Cleaned content saved to: {output_path}")
    except Exception as e:
        print(f"Error cleaning and writing text to {output_path}: {e}")
        traceback.print_exc()

def extract_chatgpt_dialogue_from_json(json_data):
    """
    Parses a loaded JSON object (list of conversations) to extract dialogue.
    """
    all_messages = []
    try:
        for conversation in json_data:
            mapping = conversation.get("mapping", {})
            if not mapping:
                continue

            # Find the root of the conversation
            root_id = None
            for node_id, node in mapping.items():
                if node.get("parent") is None:
                    root_id = node_id
                    break
            
            if not root_id: continue

            # Traverse the conversation from the root
            current_id = root_id
            while current_id:
                node = mapping.get(current_id)
                if not node: break
                
                message_data = node.get("message")
                if message_data:
                    role = message_data.get("author", {}).get("role")
                    parts = message_data.get("content", {}).get("parts", [])
                    
                    if parts and isinstance(parts, list) and len(parts) > 0:
                        text = parts[0].strip()
                        # We only care about user and assistant messages with content
                        if text and role in {"user", "assistant"}:
                            all_messages.append(f"{role.capitalize()}: {text}")
                
                # Move to the first child
                children = node.get("children", [])
                current_id = children[0] if children else None

        return "\n\n".join(all_messages)
    except Exception as e:
        print(f"Failed to parse structured ChatGPT data: {e}")
        traceback.print_exc()
        return ""

# --- UPDATED MAIN PROCESSING FUNCTION ---
def clean_text_file(input_file_path, output_file_path,
                    strip_html=False, unicode_form='NFC',
                    min_line_length=0):
    try:
        print(f"Processing: {input_file_path}")
        raw_text = input_file_path.read_text(encoding='utf-8', errors='replace')
        processed_text = ""

        # This now specifically handles the HTML export format
        if "var jsonData = " in raw_text:
            print("Detected ChatGPT HTML export format. Extracting JSON...")
            json_string = extract_json_from_html_export(raw_text)
            if json_string:
                try:
                    json_data = json.loads(json_string)
                    print("JSON extracted successfully. Parsing for dialogue...")
                    processed_text = extract_chatgpt_dialogue_from_json(json_data)
                except json.JSONDecodeError as e:
                    print(f"Error decoding the extracted JSON: {e}")
                    processed_text = raw_text # Fallback to original if JSON is broken
            else:
                 processed_text = raw_text # Fallback if JSON can't be isolated
        else:
             processed_text = raw_text # Not the format we are looking for, treat as plain text

        # Write the fully processed text to the output file
        clean_and_write_text(processed_text, output_file_path,
                             strip_html=strip_html,
                             unicode_form=unicode_form,
                             min_line_length=min_line_length)

    except Exception as e:
        print(f"Error processing file {input_file_path}: {e}")
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Clean text files in a directory.")
    parser.add_argument("input_dir", type=str, help="Directory containing input .txt or .html files.")
    parser.add_argument("output_dir", type=str, help="Directory to save cleaned .txt files.")
    parser.add_argument("--strip_html", action="store_true", help="Remove HTML tags from text.")
    parser.add_argument("--unicode_form", type=str, default="NFC", choices=['NFC', 'NFKC', 'NFD', 'NFKD'],
                        help="Unicode normalization form (default: NFC).")
    parser.add_argument("--min_line_length", type=int, default=5,
                        help="Minimum character length for a line to be kept after cleaning (default: 5).")
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
    # Use rglob to find files recursively
    for input_file_path in input_dir.rglob(f"*{args.file_extension}"):
        if input_file_path.is_file():
            relative_path = input_file_path.relative_to(input_dir)
            # Ensure the output is always a .txt file
            output_file_path = (output_dir / relative_path).with_suffix('.txt')
            
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