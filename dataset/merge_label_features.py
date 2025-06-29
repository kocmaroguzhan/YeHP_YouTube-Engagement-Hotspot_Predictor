import os
import pandas as pd
import numpy as np
import ast

def main(context_window_size=5):
    # Define paths
    root_folder = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(root_folder, "labeled_data_folder")
    os.makedirs(output_folder, exist_ok=True)

    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)
        if not os.path.isdir(subfolder_path) or subfolder == "labeled_data_folder":
            continue

        vector_file, intensity_file = None, None

        for file in os.listdir(subfolder_path):
            if file.endswith("_features_vectorlist.csv"):
                vector_file = os.path.join(subfolder_path, file)
            elif file.endswith("_averaged_intensity_heatmap.csv"):
                intensity_file = os.path.join(subfolder_path, file)

        if not vector_file or not intensity_file:
            print(f"⚠️ Missing files in '{subfolder}', skipping...")
            continue

        try:
            embedding_df = pd.read_csv(vector_file)
            intensity_df = pd.read_csv(intensity_file)

            # Align by 'start_time'
            merged = pd.merge(embedding_df, intensity_df, on="start_time", how="inner")
            intensities = merged["avg_intensity"].values

            embedding_types = {
                "bert_uncased_cls": merged["bert_uncased_cls"].apply(ast.literal_eval).tolist(),
                "bert_uncased_mean": merged["bert_uncased_mean"].apply(ast.literal_eval).tolist(),
                "bert_cased_cls": merged["bert_cased_cls"].apply(ast.literal_eval).tolist(),
                "bert_cased_mean": merged["bert_cased_mean"].apply(ast.literal_eval).tolist(),
                "sbert": merged["sbert"].apply(ast.literal_eval).tolist()
            }

            # Containers for current file
            all_embeddings = {name: [] for name in embedding_types}
            all_labels = []

            for i in range(len(intensities) - context_window_size + 1):
                for name, vectors in embedding_types.items():
                    context = vectors[i:i + context_window_size]
                    all_embeddings[name].append(context)

                all_labels.append(intensities[i + context_window_size - 1])

            print(f"✅ Processed: {subfolder}")

        except Exception as e:
            print(f"❌ Error in '{subfolder}': {e}")
            continue

        # Save .npy files for each embedding
        for name, data in all_embeddings.items():
            try:
                x_array = np.array(data)
                filename = f"{subfolder}_{name}_merged.npy"
                np.save(os.path.join(output_folder, filename), x_array)
                print(f"✅ Saved {filename} with shape {x_array.shape}")
            except Exception as e:
                print(f"❌ Failed to save {subfolder}_{name}_merged.npy: {e}")

        # Save label array
        try:
            y_array = np.array(all_labels)
            label_filename = f"{subfolder}_labels.npy"
            np.save(os.path.join(output_folder, label_filename), y_array)
            print(f"✅ Saved {label_filename} with shape {y_array.shape}")
        except Exception as e:
            print(f"❌ Failed to save labels for {subfolder}: {e}")

