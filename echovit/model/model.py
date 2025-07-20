import torch
import torch.nn as nn
import math
import pandas as pd
# Tubelet Embedding
class TubeletEmbedding(nn.Module):
    def __init__(self, embed_dim, patch_size):
        super(TubeletEmbedding, self).__init__()
        self.projection = nn.Conv3d(
            in_channels = 3,
            out_channels= embed_dim,
            kernel_size = patch_size,
            stride      = patch_size
        )

    def forward(self, x):
        x               = self.projection(x)
        return x.flatten(2).transpose(1, 2)

# Positional Encoder
class PositionalEncoder(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super(PositionalEncoder, self).__init__()
        self.embed_dim      = embed_dim
        self.num_patches    = num_patches

    def forward(self, x):
        batch_size, num_patches, embed_dim = x.size()
        position = torch.arange(0, num_patches, dtype=torch.float32, device=x.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32, device=x.device) * 
                             -(torch.log(torch.tensor(10000.0)) / embed_dim))

        pos_encoding = torch.zeros((num_patches, embed_dim), device=x.device)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)

        pos_encoding = pos_encoding.unsqueeze(0)
        return x + pos_encoding

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super(TransformerBlock, self).__init__()
        self.attn  = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff    = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim)
        )

    def forward(self, x):
        attn_output, _  = self.attn(x, x, x)
        x               = self.norm1(x + attn_output)
        ff_output       = self.ff(x)
        x               = self.norm2(x + ff_output)
        return x
    
    
# Video Vision Transformer
class ViViT(nn.Module):
    def __init__(self, input_shape, patch_size, embed_dim, num_heads, num_layers):
        super(ViViT, self).__init__()
        num_patches = (
            (input_shape[1] // patch_size[0]) * 
            (input_shape[2] // patch_size[1]) * 
            (input_shape[3] // patch_size[2])
        )
        self.embedding      = TubeletEmbedding(embed_dim, patch_size)
        self.pos_encoder    = PositionalEncoder(num_patches, embed_dim)
        self.transformer    = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, embed_dim * 4)
            for _ in range(num_layers)
        ])
        self.regressor = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),  # İlk katman: Giriş boyutunu yarıya indirir
            nn.GELU(),  # Aktivasyon fonksiyonu
            nn.Dropout(p=0.3),  # Dropout, aşırı öğrenmeyi önlemek için

            nn.Linear(embed_dim // 2, embed_dim // 4),  # İkinci katman: Boyutu tekrar küçültür
            nn.GELU(),
            nn.Dropout(p=0.3),

            nn.Linear(embed_dim // 4, embed_dim // 8),  # Üçüncü katman: Daha küçük bir boyut
            nn.GELU(),

            nn.Linear(embed_dim // 8, 1)  # Çıkış katmanı: Tek bir değer döndürür
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        for layer in self.transformer:
            x = layer(x)
        x = x.mean(dim=1)
        return self.regressor(x).squeeze()