"""
model.py
Architecture: ModernBERT encoder → GCN layers → Span classifier + VA regressor

ModernBERT (answerdotai/ModernBERT-base)
    ↓  [B, seq, H]
GCN (2 layers, residual)
    ↓  [B, seq, H]
Span classifier head  → token-level BIO logits  [B, seq, 5]
VA regressor head     → mean-pooled → [valence, arousal]  [B, 2]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from data_loader import NUM_SPAN_LABELS


# ── Graph Convolutional Layer ────────────────────────────────────────────────

class GCNLayer(nn.Module):
    """
    Single GCN layer:  H' = ReLU( D^{-1} A H W )
    Adjacency A is passed pre-normalised from the data loader.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear  = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(out_dim)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        h   : [B, seq, in_dim]
        adj : [B, seq, seq]  (row-normalised)
        """
        support = self.linear(h)                    # [B, seq, out_dim]
        agg     = torch.bmm(adj, support)           # [B, seq, out_dim]
        out     = F.relu(agg)
        out     = self.dropout(out)
        out     = self.norm(out)
        return out


# ── Full ASTE model ──────────────────────────────────────────────────────────

class ASTEModel(nn.Module):

    def __init__(
        self,
        encoder_name: str  = "answerdotai/ModernBERT-base",
        gcn_layers: int    = 2,
        gcn_dropout: float = 0.1,
        span_dropout: float = 0.2,
        va_dropout: float   = 0.2,
        freeze_encoder_layers: int = 0,   # 0 = fine-tune all
    ):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────────────────────
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_size  = self.encoder.config.hidden_size   # 768 for base

        # Optionally freeze early transformer layers
        if freeze_encoder_layers > 0:
            for i, layer in enumerate(self.encoder.encoder.layers):
                if i < freeze_encoder_layers:
                    for p in layer.parameters():
                        p.requires_grad = False

        # ── GCN ──────────────────────────────────────────────────────────────
        self.gcn_layers = nn.ModuleList()
        in_dim = hidden_size
        for _ in range(gcn_layers):
            self.gcn_layers.append(GCNLayer(in_dim, hidden_size, dropout=gcn_dropout))
            in_dim = hidden_size

        # Residual projection (if dims differ; here they're equal → identity)
        self.gcn_proj = nn.Linear(hidden_size, hidden_size)

        # ── Span classifier head ──────────────────────────────────────────────
        self.span_dropout  = nn.Dropout(span_dropout)
        self.span_classifier = nn.Linear(hidden_size, NUM_SPAN_LABELS)

        # ── VA regression head ────────────────────────────────────────────────
        self.va_dropout  = nn.Dropout(va_dropout)
        self.va_regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(va_dropout),
            nn.Linear(hidden_size // 2, 2),     # → [valence, arousal]
            nn.Sigmoid(),                        # map to [0, 1]; scale to [1, 9] at inference
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        adj:            torch.Tensor,
    ):
        """
        Returns
        -------
        span_logits : [B, seq, NUM_SPAN_LABELS]
        va_pred     : [B, 2]   values in [1, 9]
        """
        # Encoder
        enc_out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        h = enc_out.last_hidden_state   # [B, seq, H]

        # GCN with residual
        gcn_in = h
        for layer in self.gcn_layers:
            gcn_out = layer(gcn_in, adj)
            gcn_in  = gcn_out + self.gcn_proj(gcn_in)   # residual

        h_gcn = gcn_in  # [B, seq, H]

        # Span logits (token level)
        span_logits = self.span_classifier(self.span_dropout(h_gcn))  # [B, seq, 5]

        # VA prediction (mean pool over non-padding tokens)
        mask_expanded = attention_mask.unsqueeze(-1).float()           # [B, seq, 1]
        pooled = (h_gcn * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        va_raw = self.va_regressor(self.va_dropout(pooled))            # [B, 2] in [0,1]
        va_pred = va_raw * 8.0 + 1.0                                   # scale to [1, 9]

        return span_logits, va_pred


# ── Loss function ─────────────────────────────────────────────────────────────

class ASTELoss(nn.Module):
    """
    Combined loss:
      L = λ_span * CE(span, weighted) + λ_va * MSE(VA)

    Class weights upweight B/I span labels vs O to fight token imbalance
    (~81% of tokens are O, causing the model to predict O for everything).
    """

    def __init__(self, lambda_span: float = 1.0, lambda_va: float = 0.3,
                 span_class_weights: list = None, device: str = "cpu"):
        super().__init__()
        self.lambda_span = lambda_span
        self.lambda_va   = lambda_va

        # Default weights: O=1.0, B-ASP=5.0, I-ASP=4.0, B-OPN=5.0, I-OPN=4.0
        if span_class_weights is None:
            span_class_weights = [1.0, 5.0, 4.0, 5.0, 4.0]
        weights = torch.tensor(span_class_weights, dtype=torch.float)
        self.ce_loss  = nn.CrossEntropyLoss(ignore_index=-100, weight=weights)
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        span_logits: torch.Tensor,   # [B, seq, 5]
        va_pred:     torch.Tensor,   # [B, 2]
        span_labels: torch.Tensor,   # [B, seq]
        valence:     torch.Tensor,   # [B]
        arousal:     torch.Tensor,   # [B]
    ):
        # Span classification loss
        B, S, C = span_logits.shape
        span_loss = self.ce_loss(
            span_logits.view(B * S, C),
            span_labels.view(B * S),
        )

        # VA regression loss
        va_target = torch.stack([valence, arousal], dim=1)  # [B, 2]
        va_loss   = self.mse_loss(va_pred, va_target)

        total = self.lambda_span * span_loss + self.lambda_va * va_loss
        return total, span_loss, va_loss


if __name__ == "__main__":
    model = ASTEModel()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    B, S = 2, 128
    ids   = torch.randint(0, 1000, (B, S))
    mask  = torch.ones(B, S, dtype=torch.long)
    adj   = torch.eye(S).unsqueeze(0).expand(B, -1, -1)
    sl, va = model(ids, mask, adj)
    print(f"span_logits: {sl.shape}, va_pred: {va.shape}")
