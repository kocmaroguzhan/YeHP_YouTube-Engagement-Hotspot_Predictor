import os
import torch
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from transformers import BertTokenizer, BertModel

nltk.download('punkt')

# Load model and tokenizer once globally
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def extract_cls_vectors(text):
    sentences = sent_tokenize(text)
    cls_vectors = []

    for sent in sentences:
        inputs = tokenizer(sent, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        cls_vec = outputs.last_hidden_state[:, 0, :]
        cls_vectors.append(cls_vec.squeeze(0).cpu().numpy())

    return np.stack(cls_vectors)

def process_all_txt_files(root_dir, output_dir="embeddings"):
    os.makedirs(output_dir, exist_ok=True)

    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(".txt"):
                txt_path = os.path.join(dirpath, file)

                with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                    full_text = f.read()

                try:
                    cls_array = extract_cls_vectors(full_text)
                except Exception as e:
                    print(f"❌ Failed to process {txt_path}: {e}")
                    continue

                # Use folder name as video ID
                video_id = os.path.basename(os.path.dirname(txt_path))
                output_base = os.path.join(output_dir, f"{video_id}_cls_embeddings")

                # Save outputs
                np.save(f"{output_base}.npy", cls_array)
                pd.DataFrame(cls_array).to_csv(f"{output_base}.csv", index=False)

                print(f"✅ Processed: {txt_path}")
                print(f"   → {output_base}.npy")
                print(f"   → {output_base}.csv")


root_txt_folder = "dataset"           # Contains folders with .txt files
process_all_txt_files(root_txt_folder)
