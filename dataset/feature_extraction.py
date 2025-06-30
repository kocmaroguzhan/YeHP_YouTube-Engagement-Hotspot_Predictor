import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize


# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
# Embedding Dimensions
dim_bert = 768
dim_sbert = 384
# Load models
bert_uncased = BertModel.from_pretrained("bert-base-uncased").to(device).eval()
tokenizer_uncased = BertTokenizer.from_pretrained("bert-base-uncased")

bert_cased = BertModel.from_pretrained("bert-base-cased").to(device).eval()
tokenizer_cased = BertTokenizer.from_pretrained("bert-base-cased")

sbert = SentenceTransformer("all-MiniLM-L6-v2")


def l2_normalize(np_array):
    return normalize(np_array, norm='l2', axis=1)

def masked_mean_pooling(last_hidden_state, attention_mask):
    # Expand mask to match last_hidden_state shape
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    
    # Zero out padded embeddings
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
    
    # Count non-padded tokens per sequence
    sum_mask = input_mask_expanded.sum(dim=1)
    
    # Avoid division by zero
    mean_embeddings = sum_embeddings / sum_mask.clamp(min=1e-9)
    
    return mean_embeddings

# BERT feature extractor
def get_bert_features(texts, tokenizer, model):
    inputs = tokenizer(texts, return_tensors='pt', truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    cls = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # [CLS] token

    # Use attention mask for accurate mean pooling
    mean = masked_mean_pooling(outputs.last_hidden_state, inputs["attention_mask"]).cpu().numpy()

    return cls, mean

def main(time_frame_increment):
    root_folder = os.path.dirname(os.path.abspath(__file__))

    for subfolder in tqdm(os.listdir(root_folder), desc="Processing folders"):
        subfolder_path = os.path.join(root_folder, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
        time_frame_subfolder = os.path.join(subfolder_path, f"time_frame_increment_{time_frame_increment}")
        if not os.path.isdir(time_frame_subfolder):
            print(f"⚠️ Missing time_frame_increment_{time_frame_increment} folder in {subfolder}, skipping...")
            continue

        heatmap_file, words_file = None, None
        for file in os.listdir(time_frame_subfolder):
            if file.endswith("_averaged_intensity_heatmap.csv"):
                heatmap_file = os.path.join(time_frame_subfolder, file)
            elif file.endswith("_time_frame_segmented_words.csv"):
                words_file = os.path.join(time_frame_subfolder, file)

        if not heatmap_file or not words_file:
            print(f"⚠️ Missing required CSV files in {subfolder}, skipping...")
            continue

        features_file = os.path.basename(words_file).replace(".csv", "_features_vectorlist.csv")
        csv_path = os.path.join(time_frame_subfolder, features_file)
         # ✅ Skip if features vector list already exists
        if os.path.exists(csv_path):
            print(f"⏩ Skipped: Feature vector file already exists for {subfolder}")
            continue
        df_heatmap = pd.read_csv(heatmap_file)
        df_words = pd.read_csv(words_file)

        word_dict = dict(zip(df_words["start_time"], df_words["words"].fillna("").astype(str)))
        start_time = df_heatmap["start_time"].tolist()
        end_time = df_heatmap["stop_time"].tolist()

        all_texts, start_times, end_times_collected = [], [], []
        u_cls_list, u_mean_list = [], []
        c_cls_list, c_mean_list = [], []
        sbert_list = []

        texts_to_encode = []
        time_to_encode = []
        time_to_encode_end = []

        for i, t in enumerate(start_time):
            if t in word_dict and word_dict[t].strip():
                texts_to_encode.append(word_dict[t])
                time_to_encode.append(t)
                time_to_encode_end.append(end_time[i])
            else:
                u_cls_list.append(np.zeros(dim_bert))
                u_mean_list.append(np.zeros(dim_bert))
                c_cls_list.append(np.zeros(dim_bert))
                c_mean_list.append(np.zeros(dim_bert))
                sbert_list.append(np.zeros(dim_sbert))
                all_texts.append("")
                start_times.append(t)
                end_times_collected.append(end_time[i])

        for i in range(0, len(texts_to_encode), batch_size):
            batch = texts_to_encode[i:i + batch_size]
            t_batch = time_to_encode[i:i + batch_size]
            t_end_batch = time_to_encode_end[i:i + batch_size]

            u_cls, u_mean = get_bert_features(batch, tokenizer_uncased, bert_uncased)
            c_cls, c_mean = get_bert_features(batch, tokenizer_cased, bert_cased)
            s_vec = l2_normalize(sbert.encode(batch))

            u_cls = l2_normalize(u_cls)
            u_mean = l2_normalize(u_mean)
            c_cls = l2_normalize(c_cls)
            c_mean = l2_normalize(c_mean)

            for j in range(len(batch)):
                all_texts.append(batch[j])
                start_times.append(t_batch[j])
                end_times_collected.append(t_end_batch[j])
                u_cls_list.append(u_cls[j])
                u_mean_list.append(u_mean[j])
                c_cls_list.append(c_cls[j])
                c_mean_list.append(c_mean[j])
                sbert_list.append(s_vec[j])

        df_final = pd.DataFrame({
            "text": all_texts,
            "start_time": start_times,
            "end_time": end_times_collected,
            "bert_uncased_cls": u_cls_list,
            "bert_uncased_mean": u_mean_list,
            "bert_cased_cls": c_cls_list,
            "bert_cased_mean": c_mean_list,
            "sbert": sbert_list
        }).sort_values("start_time")
        # Convert numpy arrays to standard stringified Python lists
        for col in ["bert_uncased_cls", "bert_uncased_mean", "bert_cased_cls", "bert_cased_mean", "sbert"]:
            df_final[col] = df_final[col].apply(lambda x: str(x.tolist()) if isinstance(x, np.ndarray) else str(list(x)))

        df_final.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"✅ Saved Feature Vector List CSV: {csv_path}")