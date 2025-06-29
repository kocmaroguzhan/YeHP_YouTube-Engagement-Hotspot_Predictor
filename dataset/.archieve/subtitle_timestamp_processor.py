import os
import re
import pandas as pd

# Define the root directory (where this script is placed)
root_dir = os.path.dirname(os.path.abspath(__file__))

def clean_line(text):
    # Remove lines that start with specific unwanted labels 
    # Added for TEDex talks
    if re.match(r'^\s*(Transcriber|Reviewer)\s*:\s*', text, re.IGNORECASE):
        return ''  # Skip the whole line
    text = re.sub(r'^-+\s*', '', text)                          # Remove leading dashes
    text = re.sub(r'(?<=\s)-\s+', '', text)
    text = re.sub(r'\([^)]*\)', '', text)                       # Remove (non-verbal cues)
    text = re.sub(r'\[[^]]*\]', '', text)                       # Remove [non-verbal cues]
    text = re.sub(r'\b[A-Z][\w\s]{0,29}:\s*(?=[A-Z])', '', text)  # Remove speaker names
    text = re.sub(r'\s+', ' ', text)                            # Collapse multiple spaces
    return text.strip()


def timestamp_to_seconds(ts):
    h, m, s = ts.split(':')
    s, ms = s.split('.')
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

def process_vtt_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    entries = []
    start_time, end_time, buffer = None, None, []
    collecting = False

    for line in lines:
        line = line.strip()

        if re.match(r'^\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}$', line):
            collecting = True

            if buffer and start_time is not None:
                sentence = ' '.join(buffer).strip()
                sentence = clean_line(sentence)
                if sentence:
                    entries.append((start_time, end_time, sentence))
                buffer = []

            raw_start, raw_end = line.split(' --> ')
            start_time = timestamp_to_seconds(raw_start)
            end_time = timestamp_to_seconds(raw_end)

        elif collecting and line and not line.isdigit():
            buffer.append(line)

    if collecting and buffer and start_time is not None:
        sentence = ' '.join(buffer).strip()
        sentence = clean_line(sentence)
        if sentence:
            entries.append((start_time, end_time, sentence))

    return pd.DataFrame(entries, columns=["start_time", "end_time", "text"])

# Walk through all subdirectories
for subfolder in os.listdir(root_dir):
    subfolder_path = os.path.join(root_dir, subfolder)
    if os.path.isdir(subfolder_path):
        for file in os.listdir(subfolder_path):
            if file.endswith(".vtt"):
                vtt_path = os.path.join(subfolder_path, file)
                df = process_vtt_file(vtt_path)
                output_csv_path = os.path.join(subfolder_path, file.replace(".vtt", "_cleaned.csv"))
                df.to_csv(output_csv_path, index=False)
                print(f"Processed and saved: {output_csv_path}")
