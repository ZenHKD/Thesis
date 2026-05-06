"""
MODULE: Number Head — RTI-augmented Regression for Distance & Count

Position: After Decoder, parallel to LM Head
Input:  1. hidden state at <|num|> token [B_num, 1024] (decoder context)
        2. cross-attended RTI features [N_masks, 512] per sample (spatial info)
Output: continuous scalar predictions [B_num] (non-negative)

Used for:
    - distance tasks: predicts distance in meters (float)
    - count tasks:    predicts count (float, rounded to int at inference)

Changes:
    - h_num (query) cross-attends to RTI features (keys/values)
    - Provides direct spatial pathway independent of CoT
    - Both decoder context AND visual RTI info contribute to prediction

Architecture:
    Query projection:  h_num [1024] -> Linear(1024, 256) -> query [256]
    KV projection:     cross_rti [N_masks, 512] -> Linear(512, 256) -> keys [N_masks, 256]
    Cross-attention:   query attends to keys -> attended [256]
    Regression MLP:    concat(query, attended) [512] -> Linear(512, 256) -> GELU -> Linear(256, 1) -> softplus

    Standard attention (not flash): query is single token, keys = 2-12 masks.

Params: ~0.53M
    query_proj:  LN(1024) + Linear(1024, 256)  = 2*1024 + 1024*256 + 256 = 264,448
    kv_proj:     LN(512) + Linear(512, 256)     = 2*512 + 512*256 + 256   = 132,352
    cross_attn:  MHA(256, 4 heads)              = 4*256*256 + 256         = 262,400
    regression:  LN(512) + Linear(512, 256) + Linear(256, 1)              = 132,609
    Total:       ~791,809 ≈ 0.79M
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NumberHead(nn.Module):
    """RTI-augmented regression head for distance and count tasks.

    Uses cross-attention: h_num (query) attends to cross-attended RTI (keys)
    to extract task-relevant spatial information directly from mask features,
    independent of any text generation (CoT-free).

    Standard attention used (not flash) — query is 1 token, keys = 2-12 masks.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,       # decoder hidden dim
        mask_feat_dim: int = 512,     # MaskCrossAttention output dim
        proj_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj_dim = proj_dim

        # Project decoder hidden state -> query
        self.query_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, proj_dim),
        )

        # Project cross-attended RTI -> keys/values
        self.kv_proj = nn.Sequential(
            nn.LayerNorm(mask_feat_dim),
            nn.Linear(mask_feat_dim, proj_dim),
        )

        # Cross-attention: h_num attends to mask features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Final regression: concat(query, attended) -> scalar
        self.regression = nn.Sequential(
            nn.LayerNorm(proj_dim * 2),
            nn.Linear(proj_dim * 2, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, 1),
        )

    def forward(
        self,
        h_num: torch.Tensor,
        cross_rti_list: list,
    ) -> torch.Tensor:
        """
        Args:
            h_num:          [B_num, 1024] — hidden states at <|num|> positions.
                            B_num = number of numeric samples in batch.
            cross_rti_list: list of [N_masks_b, 512] tensors, one per sample.
                            len(cross_rti_list) == B_num.

        Returns:
            [B_num] — predicted values (non-negative via softplus)
        """
        B_num = h_num.shape[0]
        if B_num == 0:
            return torch.zeros(0, device=h_num.device, dtype=h_num.dtype)

        preds = []
        for i in range(B_num):
            # Query: decoder hidden -> [1, 1, proj_dim]
            q = self.query_proj(h_num[i:i+1]).unsqueeze(0)  # [1, 1, proj_dim]

            # Keys/Values: cross-attended RTI -> [1, N_masks, proj_dim]
            rti_i = cross_rti_list[i]  # [N_masks, mask_feat_dim]
            kv = self.kv_proj(rti_i).unsqueeze(0)  # [1, N_masks, proj_dim]

            # Cross-attention: query attends to mask features
            attended, _ = self.cross_attn(q, kv, kv)  # [1, 1, proj_dim]

            # Concat query + attended -> regression
            q_flat = q.squeeze(0)             # [1, proj_dim]
            att_flat = attended.squeeze(0)     # [1, proj_dim]
            combined = torch.cat([q_flat, att_flat], dim=-1)  # [1, 2*proj_dim]

            pred = self.regression(combined).squeeze(-1)  # [1]
            preds.append(pred)

        return F.softplus(torch.cat(preds))  # [B_num]
