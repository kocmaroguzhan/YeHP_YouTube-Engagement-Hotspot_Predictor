import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import numpy as np
import os


class EngagementDataset(Dataset):
    def __init__(self, data_folder, embedding_type="bert_uncased_cls"):
        """
        data_folder: path to the folder that contains .npy files
        embedding_type: one of ['bert_uncased_cls', 'bert_uncased_mean', 'bert_cased_cls', 'bert_cased_mean', 'sbert']
        """
        self.data_folder = data_folder
        self.embedding_type = embedding_type

        self.embedding_files = []
        self.label_files = []

        for file in os.listdir(data_folder):
            if file.endswith(f"{embedding_type}_context_windows.npy"):
                label_file = file.replace(f"_{embedding_type}_context_windows.npy", "_labels.npy")

                label_path = os.path.join(data_folder, label_file)
                embedding_path = os.path.join(data_folder, file)

                if os.path.exists(label_path):
                    self.embedding_files.append(embedding_path)
                    self.label_files.append(label_path)
                else:
                    print(f"⚠️ Label file missing for: {file}")

        # Load all data
        self.embeddings = []
        self.labels = []
        self.sample_origins = []
        for emb_file, lab_file in zip(self.embedding_files, self.label_files):
            X = np.load(emb_file)
            y = np.load(lab_file)
            assert len(X) == len(y), f"Mismatched shapes: {emb_file} and {lab_file}"
            self.embeddings.append(X)
            self.labels.append(y)
            self.sample_origins.extend([(os.path.basename(emb_file), i) for i in range(len(X))])

        self.embeddings = np.concatenate(self.embeddings, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)

        print(f"✅ Loaded dataset with {len(self)} samples from '{data_folder}'")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.embeddings[idx], dtype=torch.float32)  # shape: (T, D)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)      # shape: ()
        origin_file, row_index = self.sample_origins[idx]
        #print(f"📂 Sample {idx} comes from '{origin_file}', row {row_index}")
        return x, y
# Positional Encoding Module
# ------------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=1000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, embedding_dim)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class AttentionPooling(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.attn = nn.Linear(embedding_dim, 1)

    def forward(self, x):  # x: (batch_size, time_frames, embedding_dim)
        scores = self.attn(x).squeeze(-1)           # (batch_size, time_frames)
        weights = torch.softmax(scores, dim=1)      # soft attention weights
        pooled = (x * weights.unsqueeze(-1)).sum(dim=1)  # weighted sum -> (batch_size, embedding_dim)
        return pooled
    
# ------------------------
# Transformer-based Regressor
# ------------------------
class TransformerRegressor(nn.Module):
    def __init__(self, embedding_dim=384, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.pos_encoder = SinusoidalPositionalEncoding(embedding_dim, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True  # (B, T, D) shape
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attn_pool = AttentionPooling(embedding_dim)
        self.regressor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.BatchNorm1d(embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, 1)
        )

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, time_frames, embedding_dim)
        """
        x = self.pos_encoder(x)          # Add positional info
        x = self.transformer(x)          # Transformer encoding
        pooled = self.attn_pool(x)     # Attention pooling
        out = self.regressor(pooled)     # Final regressor
        return out.squeeze(-1)           # (batch_size,)




time_frame_increment=20
context_window_size=4
dataset_folder = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "dataset",
    "labeled_data_folder",
    f"time_frame_increment_{time_frame_increment}_context_window_size_{context_window_size}"
)

# Create dataset
#embedding_type: one of ['bert_uncased_cls', 'bert_uncased_mean', 'bert_cased_cls', 'bert_cased_mean', 'sbert']
dataset = EngagementDataset(dataset_folder, embedding_type="bert_uncased_cls")
"""
print("🔍 Checking first 5 samples in the dataset...\n")
for i in range(5):
    x, y = dataset[i]
    print(f"Sample {i+1}:")
    print(f"  - Label (float): {y.item():.4f}")
    print(f"  - Embedding shape: {x.shape}")
    print(f"  - Embedding dtype: {x.dtype}")
    print(f"  - First 2 vectors:\n{x[:2].numpy()}\n")  # first 2 rows of the embedding
"""
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])






batch_size = 32  # or whatever you want

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#384 sbert ,768 others 
model = TransformerRegressor(embedding_dim=768).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 40  # set as needed

mean_intensity = dataset.labels.mean()
baseline_mse = np.mean((dataset.labels - mean_intensity) ** 2)
print(f"Baseline MSE (predicting mean always): {baseline_mse:.4f}")



for epoch in range(num_epochs):
    model.train()
    total_train_loss  = 0

    for batch_embeddings, batch_labels in train_loader:
        batch_embeddings = batch_embeddings.to(device)  # (B, T, D)
        batch_labels = batch_labels.to(device)          # (B,)

        preds = model(batch_embeddings)
        loss = criterion(preds, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    print(f"Epoch {epoch+1} | Loss: {avg_train_loss:.4f}")
    # -------- VALIDATION --------
    model.eval()
    total_val_loss = 0

    with torch.no_grad():
        for batch_embeddings, batch_labels in val_loader:
            batch_embeddings = batch_embeddings.to(device)
            batch_labels = batch_labels.to(device)

            preds = model(batch_embeddings)
            loss = criterion(preds, batch_labels)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")


