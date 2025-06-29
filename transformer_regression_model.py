import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset
import numpy as np


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

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerRegressor(embedding_dim=768).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
num_epochs = 10  # set as needed

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch_embeddings, batch_labels in dataloader:
        batch_embeddings = batch_embeddings.to(device)  # (B, T, D)
        batch_labels = batch_labels.to(device)          # (B,)

        preds = model(batch_embeddings)
        loss = criterion(preds, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")
