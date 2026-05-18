"""
MODULE: Count Head — Mask-Centric Tri-Source Regression

Position: After Decoder, parallel to LM Head
Input:  1. hidden state at <|count|> token [B_count, 1024]
        2. Tri-source mask features: rgb, dep, gdep lists of [N_masks, 1024]
Output: continuous scalar predictions [B_count] (positive, rounded at inference)

Architecture:
    1. Sigmoid Filtering (Sum Pooling): Replaces softmax attention to preserve object cardinality.
    2. Regression MLP: concat(q, att_rgb, att_dep, att_gdep) [4096]
       → LN → Linear(4096, 512) → GELU → Dropout → Linear(512, 1)

Params: ~2.10M (private regression MLP only)
    regression: LN(4096) + Linear(4096, 512) + Linear(512, 1) = ~2.1M
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CountHead(nn.Module):
    """Mask-centric Tri-Source regression head for counting."""

    def __init__(
        self,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scale = math.sqrt(hidden_dim)

        # Count Regression MLP (shallower than DistanceHead)
        # Input: concat(q, att_rgb, att_dep, att_gdep) [4096]
        self.regression = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
        )

    def forward(
        self,
        h_token: torch.Tensor,         # [B_count, 1024]
        rgb_list: list,                 # list of [N_masks, 1024]
        dep_list: list,                 # list of [N_masks, 1024]
        gdep_list: list,                # list of [N_masks, 1024]
    ) -> torch.Tensor:
        """
        Args:
            h_token:         [B_count, 1024]
            rgb_list:        list of [N_masks_b, 1024]
            dep_list:        list of [N_masks_b, 1024]
            gdep_list:       list of [N_masks_b, 1024]

        Returns:
            [B_count] — predicted counts (positive via softplus)
        """
        B_count = h_token.shape[0]
        if B_count == 0:
            return torch.zeros(0, device=h_token.device, dtype=h_token.dtype)

        preds = []
        for i in range(B_count):
            q = h_token[i]  # [1024]

            rgb  = rgb_list[i]   # [N_masks, 1024]
            dep  = dep_list[i]   # [N_masks, 1024]
            gdep = gdep_list[i]  # [N_masks, 1024]

            # Tri-Source Attention @ full 1024-dim
            # Use Sigmoid to filter objects and SUM their features (preserves cardinality)
            score_rgb = (q @ rgb.T) / self.scale
            att_rgb = torch.sigmoid(score_rgb) @ rgb    # [1024]

            score_dep = (q @ dep.T) / self.scale
            att_dep = torch.sigmoid(score_dep) @ dep    # [1024]

            score_gdep = (q @ gdep.T) / self.scale
            att_gdep = torch.sigmoid(score_gdep) @ gdep  # [1024]

            # Concat → shallower regression (no n_masks in features)
            combined = torch.cat([q, att_rgb, att_dep, att_gdep], dim=-1)  # [4096]
            pred = self.regression(combined.unsqueeze(0)).squeeze(-1).squeeze(-1)
            preds.append(pred)

        return F.softplus(torch.stack(preds))  # [B_count]
