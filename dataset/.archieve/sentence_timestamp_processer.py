import os
import pandas as pd
import spacy

# Define the root directory (where this script is placed)
root_folder = os.path.dirname(os.path.abspath(__file__))
# Load spaCy English model (make sure it's installed)
nlp = spacy.load("en_core_web_sm")

def interpolate_token_timestamps(df):
    """Assign start and end times to each token using character-length weighted interpolation."""
    token_infos = []

    for _, row in df.iterrows():
        start = row['start_time']
        end = row['end_time']
        text = str(row['text'])

        doc = nlp(text)
        tokens = [token.text for token in doc if not token.is_space and not token.is_punct]
        if not tokens:
            continue

        total_length = sum(len(token) for token in tokens)
        if total_length == 0:
            continue

        duration = end - start
        current_time = start

        for token in tokens:
            weight = len(token) / total_length
            token_duration = duration * weight
            token_start = current_time
            token_end = token_start + token_duration
            token_infos.append((token_start, token_end, token))
            current_time = token_end

    return token_infos

def merge_with_precise_timing(df):
    """Merge broken lines into full sentences with interpolated timing."""
    token_infos = interpolate_token_timestamps(df)
    all_text = ' '.join(str(t) for t in df['text'].tolist())
    doc = nlp(all_text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    sentence_data = []
    token_index = 0

    for sentence in sentences:
        sentence_doc = nlp(sentence)
        num_tokens = len([t for t in sentence_doc if not t.is_space and not t.is_punct])
        if num_tokens == 0:
            continue
        available_tokens = len(token_infos) - token_index
        actual_tokens = min(num_tokens, available_tokens)

        sentence_tokens = token_infos[token_index: (token_index + actual_tokens)]
        if not sentence_tokens:
            continue  # skip if no tokens matched
        sentence_tokens = token_infos[token_index: token_index + num_tokens]
        start_time = sentence_tokens[0][0]
        end_time = sentence_tokens[-1][1]
        sentence_data.append((start_time, end_time, sentence))
        token_index += actual_tokens

    return pd.DataFrame(sentence_data, columns=["start_time", "end_time", "text"])


# Iterate through all subdirectories inside the root folder
for subfolder in os.listdir(root_folder):
    subfolder_path = os.path.join(root_folder, subfolder)

    if os.path.isdir(subfolder_path):
        # Look for any *_cleaned.csv file in that subfolder
        for file in os.listdir(subfolder_path):
            if file.endswith("_cleaned.csv"):
                input_path = os.path.join(subfolder_path, file)

                # Load and clean CSV
                df = pd.read_csv(input_path)
                df.dropna(subset=['text'], inplace=True)
                df = df[df['text'].str.strip().astype(bool)]
                df['start_time'] = df['start_time'].astype(float)
                df['end_time'] = df['end_time'].astype(float)

                # Merge sentences using spaCy
                merged_df = merge_with_precise_timing(df)

                # Save result
                output_path = os.path.join(subfolder_path, file.replace("_cleaned.csv", "_sentence_vs_timestamp.csv"))
                merged_df.to_csv(output_path, index=False)
                print(f"Processed and saved: {output_path}")
