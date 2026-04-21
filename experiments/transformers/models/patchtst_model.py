import torch
import torch.nn as nn
import math

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class PatchTST(nn.Module):
    def __init__(self, n_features, seq_len=192, patch_len=16, stride=8, d_model=128, n_heads=4, n_layers=4, dropout=0.1):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.n_features = n_features
        self.channel_independence = True
        
        # Calculate number of patches
        # Formula: floor((seq_len - patch_len) / stride) + 1
        self.num_patches = (seq_len - patch_len) // stride + 1
        
        # Print parameters for traceability
        print(f"[PatchTST] Initializing with patch_len={patch_len}, stride={stride}, channel_independence=True, num_patches={self.num_patches}")
        
        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.output_layer = nn.Linear(d_model, 1)
        
    def forward(self, x):
        # x: (Batch, Seq_Len, Features)
        B, L, F = x.shape
        
        # Channel-Independent Processing: treat features as separate series
        # Reshape: (Batch, Seq_Len, Features) -> (Batch * Features, Seq_Len)
        x = x.permute(0, 2, 1).reshape(B * F, L)
        
        # Patching: (Batch * Features, Seq_Len) -> (Batch * Features, Num_Patches, Patch_Len)
        # Using unfolding to create patches
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        
        # Embedding: (Batch * Features, Num_Patches, Patch_Len) -> (Batch * Features, Num_Patches, D_Model)
        x = self.patch_embedding(x)
        x = self.pos_encoder(x)
        
        # Encoder: (Batch * Features, Num_Patches, D_Model)
        x = self.transformer_encoder(x)
        
        # Mean pooling across patches (Canonical aggregation)
        # x: (Batch * Features, D_Model)
        x = torch.mean(x, dim=1)
        
        # Output: (Batch * Features, 1)
        x = self.output_layer(x)
        
        # Reshape back to Batch and average across features to get 1 scalar per sample
        # x: (Batch, Features)
        x = x.reshape(B, F)
        
        # Final aggregation across features (Simple mean)
        x = torch.mean(x, dim=1)
        
        return x
