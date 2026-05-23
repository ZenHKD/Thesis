"""
MODULE: Distance Head v2 — Concat Fusion Deep Regression

Position: After Decoder, parallel to LM Head
Input:  1. hidden state at <|dist|> token [B_dist, 1024]
        2. Tri-source mask features: rgb, dep, gdep lists of [N_masks, 1024]
Output: continuous scalar predictions [B_dist] (non-negative, meters)

Architecture:
    1. Softmax Attention: Focus on the most relevant mask(s) per source.
    2. Concat Fusion: cat(q, att_rgb, att_dep, att_gdep) [4096]
       → Preserves all signals independently (no information collapse).
    3. Deep Regression MLP: LN(4096) → Linear(4096, 1024) → GELU → Dropout
       → Linear(1024, 256) → GELU → Dropout → Linear(256, 1) → softplus
       Deeper MLP for careful, progressive dimensionality reduction —
       distance regression is highly sensitive to small feature variations.

Changes from v1:
    - Removed sigmoid gates (3M params saved — MLP handles filtering)
    - Removed per-source projections (3M params saved)
    - Residual Addition → Concatenation (no q domination)
    - relu → softplus (smooth gradient, no dead zone)
    - Deeper regression MLP (4096→1024→256→1) for fine-grained learning

Params: ~4.5M
    regression: LN(4096) + Linear(4096,1024) + Linear(1024,256) + Linear(256,1)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DistanceHeadV2(nn.Module):
    """Concat fusion deep regression head for distance prediction.

    Key differences from v1:
        - Concat fusion [4096] instead of residual addition [1024]
        - No sigmoid gates (MLP learns its own filtering)
        - Deeper MLP (4096→1024→256→1) for sensitive regression
        - Softplus output (smooth, no dead gradient zone)
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scale = math.sqrt(hidden_dim)

        # Deep Regression MLP: progressive reduction for precise regression
        self.regression = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        h_token: torch.Tensor,         # [B_dist, 1024]
        rgb_list: list,                 # list of [N_masks, 1024]
        dep_list: list,                 # list of [N_masks, 1024]
        gdep_list: list,                # list of [N_masks, 1024]
    ) -> torch.Tensor:
        """
        Args:
            h_token:   [B_dist, 1024] — hidden states at <|dist|> positions.
            rgb_list:  list of [N_masks_b, 1024] tensors.
            dep_list:  list of [N_masks_b, 1024] tensors.
            gdep_list: list of [N_masks_b, 1024] tensors.

        Returns:
            [B_dist] — predicted distances (non-negative via softplus)
        """
        B_dist = h_token.shape[0]
        if B_dist == 0:
            return torch.zeros(0, device=h_token.device, dtype=h_token.dtype)

        preds = []
        for i in range(B_dist):
            q = h_token[i]  # [1024]

            rgb  = rgb_list[i]   # [N_masks, 1024]
            dep  = dep_list[i]   # [N_masks, 1024]
            gdep = gdep_list[i]  # [N_masks, 1024]

            # Softmax Attention: focus on the most relevant mask(s)
            score_rgb = (q @ rgb.T) / self.scale
            att_rgb = torch.softmax(score_rgb, dim=-1) @ rgb    # [1024]

            score_dep = (q @ dep.T) / self.scale
            att_dep = torch.softmax(score_dep, dim=-1) @ dep    # [1024]

            score_gdep = (q @ gdep.T) / self.scale
            att_gdep = torch.softmax(score_gdep, dim=-1) @ gdep  # [1024]

            # Concat Fusion: all 4 signals preserved independently
            combined = torch.cat([q, att_rgb, att_dep, att_gdep], dim=-1)  # [4096]

            pred = self.regression(combined.unsqueeze(0)).squeeze(-1).squeeze(-1)
            preds.append(pred)

        return F.softplus(torch.stack(preds))  # [B_dist]
