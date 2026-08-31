import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class MultimodalStockTransformer(nn.Module):
    def __init__(
        self,
        price_features: int = 6,
        rag_dim: int = 384,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.price_proj = nn.Linear(price_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.price_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        self.rag_proj = nn.Sequential(
            nn.Linear(rag_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, price_seq: torch.Tensor, rag_embeddings: torch.Tensor):

        p = self.pos_encoder(self.price_proj(price_seq))
        price_enc = self.price_transformer(p)  # [Batch, 60, d_model]

        text_enc = self.rag_proj(rag_embeddings) # [Batch, 3, d_model]
        
        attn_out, attn_weights = self.cross_attention(
            query=price_enc,
            key=text_enc,
            value=text_enc
        )
        
        fused = self.layer_norm(price_enc + attn_out) 
        pooled = fused.mean(dim=1)                   
        
        logits = self.classifier(pooled)              
        return logits, attn_weights