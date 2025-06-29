import os
import re
import pandas as pd
import spacy
import json
nlp = spacy.load("en_core_web_sm")




def clean_line(text):
    # Remove lines that start with specific unwanted labels 
    #This is added for TEDex talks spesifically
    if re.match(r'^\s*(Transcriber|Reviewer)\s*:\s*', text, re.IGNORECASE):
        return ''  # Skip line entirely

    text = re.sub(r'<\d{2}:\d{2}:\d{2}\.\d{3}><c>', '', text)  # Remove <timestamp><c> tags
    text = re.sub(r'</c>', '', text)                          # Remove </c> closing tags
    text = re.sub(r'^-+\s*', '', text)                        # Remove leading dashes
    text = re.sub(r'(?<=\s)-\s+', '', text)                   # Remove stray dashes
    text = re.sub(r'\([^)]*\)', '', text)                     # Remove (non-verbal cues)
    text = re.sub(r'\[[^]]*\]', '', text)                     # Remove [non-verbal cues]
    text = re.sub(r'\b[A-Z][\w\s]{0,29}:\s*(?=[A-Z])', '', text)  # Remove speaker names
    text = re.sub(r'\s+', ' ', text)                          # Collapse multiple spaces
    return text.strip()


def timestamp_to_seconds(ts):
    h, m, s = ts.split(':')
    s, ms = s.split('.')
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

def process_manual_written_vtt_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    entries = []
    start_time, end_time, buffer = None, None, []
    collecting = False

    for line in lines:
        line = line.strip()

        if re.match(r'^\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}', line):
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

def process_auto_generated_vtt_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    entries = []
    start_time, end_time, buffer = None, None, []
    collecting = False

    for line in lines:
        line = line.strip()

        if re.match(r'^\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}', line):
            collecting = True
            if buffer and start_time is not None:
                sentence = ' '.join(buffer).strip()
                sentence = clean_line(sentence)
                if sentence:
                    entries.append((start_time, end_time, sentence))
                buffer = []

            raw_start, raw_end = line.split(' --> ')
            start_time = timestamp_to_seconds(raw_start)
            end_time = timestamp_to_seconds(raw_end.split()[0])##remove unnecessary part of auto generated subtitles like 'align:start position:0%' 

        # ✅ Only add subtitle lines that include <timestamp><c> tags
        elif collecting and line and not line.isdigit() and (re.search(r'<\d{2}:\d{2}:\d{2}\.\d{3}><c>', line) or re.search(r'</c>', line)):
            buffer.append(line)

    if collecting and buffer and start_time is not None:
        sentence = ' '.join(buffer).strip()
        sentence = clean_line(sentence)
        if sentence:
            entries.append((start_time, end_time, sentence))

    return pd.DataFrame(entries, columns=["start_time", "end_time", "text"])

def word_tokenizer (text):
    doc = nlp(text)
    # Extract word-level tokens without punctuation
    tokens = [token.text for token in doc if not token.is_space]
    return tokens


def segment_subtitle_time_frames(df, time_frame_increment=10):
    results = []
    frame_start = 0
    frame_end = time_frame_increment
    max_duration = df["end_time"].max()
    carryover_words = []

    while frame_start < max_duration:
        window_words = carryover_words
        carryover_words = []

        for _, row in df.iterrows():
            line_start = row["start_time"]
            line_end = row["end_time"]
            text = row["text"]

            # Line is fully within time frame
            if line_start >= frame_start and line_end <= frame_end:
                window_words.extend(word_tokenizer(text))

            # Take the part of line which is inside time frame
            elif line_start >= frame_start and line_end > frame_end and line_start <=frame_end :
                tokens = word_tokenizer(text)
                # Define which tokens are punctuation using spaCy
                doc = nlp(text)
                is_punct_flags = [token.is_punct for token in doc if not token.is_space]
                char_tokens = [token for token, is_punct in zip(tokens, is_punct_flags) if not is_punct]
                total_chars = sum(len(token) for token in char_tokens)
                duration = line_end - line_start
                current_time = line_start

                for token, is_punct in zip(tokens, is_punct_flags):
                    ##Only add the punct is we are still in certain time frame
                    if is_punct and token_end<=frame_end:
                        window_words.append(token)  # Directly add punctuation
                        continue

                    # Interpolate time for non-punctuation tokens
                    token_duration = duration * (len(token) / total_chars)
                    token_start = current_time
                    token_end = token_start + token_duration

                    if frame_start <= token_start and frame_end >= token_end:
                        window_words.append(token)
                    else:
                        carryover_words.append(token)

                    current_time += token_duration

        if window_words:  # Only append if the list is not empty
            results.append((frame_start, frame_end, " ".join(window_words)))
        frame_start += time_frame_increment
        frame_end += time_frame_increment
    return pd.DataFrame(results, columns=["start_time", "end_time", "words"])



            
def main(time_frame_increment):
    root_folder = os.path.dirname(os.path.abspath(__file__))
    # Iterate through all subdirectories inside the root folder
    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)

        if os.path.isdir(subfolder_path):
            vtt_path = None
            subtitle_type = "none"

            for file in os.listdir(subfolder_path):
                if file.endswith("_metadata.json"):
                    metadata_path = os.path.join(subfolder_path, file)
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                        auto_captions = metadata.get("automatic_captions", {})
                        manual_subs = metadata.get("subtitles", {})
                        if "en" in manual_subs:
                            subtitle_type = "manual"
                        elif "en" in auto_captions:
                            subtitle_type = "auto"
                        else:
                            print(f"No available subtitle for: {subfolder_path}")
                            break  # No need to continue

            for file in os.listdir(subfolder_path):
                if file.endswith(".vtt") and "en" in file:  # Optional: restrict to English VTTs
                    vtt_path = os.path.join(subfolder_path, file)
                    break

            if subtitle_type == "none" or not vtt_path:
                continue

            if subtitle_type == "auto":
                df = process_auto_generated_vtt_file(vtt_path)
            elif subtitle_type == "manual":
                df = process_manual_written_vtt_file(vtt_path)

            # Save cleaned subtitles
            output_csv_path = os.path.join(subfolder_path, os.path.basename(vtt_path).replace(".vtt", "_cleaned.csv"))
            df.to_csv(output_csv_path, index=False)
            print(f"Processed and saved cleaned subtitle: {output_csv_path}")

            # Drop empty lines and convert time columns
            df.dropna(subset=['text'], inplace=True)
            df = df[df['text'].str.strip().astype(bool)]
            df['start_time'] = df['start_time'].astype(float)
            df['end_time'] = df['end_time'].astype(float)

            # Segment and save word-level time frames
            segmented_df = segment_subtitle_time_frames(df, time_frame_increment=time_frame_increment)
            output_path = os.path.join(subfolder_path, os.path.basename(vtt_path).replace(".vtt", "_time_frame_segmented_words.csv"))
            segmented_df.to_csv(output_path, index=False)
            print(f"Processed and saved: {output_path}")