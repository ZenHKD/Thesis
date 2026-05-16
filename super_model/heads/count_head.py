"""
MODULE: Count Head — Mask-Centric Tri-Source Regression (uses SharedVisualFuser)

Position: After Decoder, parallel to LM Head
Input:  1. hidden state at <|count|> token [B_count, 1024]
        2. Tri-source mask features: rgb, dep, gdep lists of [N_masks, 1024]
        3. Visual tokens [N_vis, 1024] (via shared fuser)
Output: continuous scalar predictions [B_count] (positive, rounded at inference)

Design differences from DistanceHead:
    - Down-weighted scene context (×scene_gate) — counting is mask-centric
    - N_masks as query bias: q *= (1 + N_masks/10), more masks → stronger query
    - Shallower MLP: 4096 → 512 → 1
    - Softplus output for smooth positive values

Params: ~2.10M (private regression MLP + scene_gate)
    regression: LN(4096) + Linear(4096, 512) + Linear(512, 1)
    scene_gate: 1 parameter
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CountHead(nn.Module):
    """Mask-centric Tri-Source regression head for counting.

    Uses external SharedVisualFuser for scene context extraction.
    N_masks modulates query strength: q *= (1 + N_masks / 10).
    More masks → stronger attention query → model aware of density.
    """

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

        # Learnable gate for scene context weighting
        self.scene_gate = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        h_token: torch.Tensor,         # [B_count, 1024]
        rgb_list: list,                 # list of [N_masks, 1024]
        dep_list: list,                 # list of [N_masks, 1024]
        gdep_list: list,                # list of [N_masks, 1024]
        q_list: list,                   # list of [1, 1024] — pre-fused queries
        scene_ctx_list: list,           # list of [1, 1024] — scene contexts (for gating)
    ) -> torch.Tensor:
        """
        Args:
            h_token:         [B_count, 1024]
            rgb_list:        list of [N_masks_b, 1024]
            dep_list:        list of [N_masks_b, 1024]
            gdep_list:       list of [N_masks_b, 1024]
            q_list:          list of [1, 1024] — fused queries from SharedVisualFuser
            scene_ctx_list:  list of [1, 1024] — scene contexts for gating

        Returns:
            [B_count] — predicted counts (positive via softplus)
        """
        B_count = h_token.shape[0]
        if B_count == 0:
            return torch.zeros(0, device=h_token.device, dtype=h_token.dtype)

        preds = []
        for i in range(B_count):
            q = q_list[i].squeeze(0)  # [1024]

            rgb  = rgb_list[i]   # [N_masks, 1024]
            dep  = dep_list[i]   # [N_masks, 1024]
            gdep = gdep_list[i]  # [N_masks, 1024]

            # N_masks as query bias: more masks → stronger query
            n_masks = rgb.shape[0]
            q = q * (1.0 + n_masks / 10.0)

            # Tri-Source Attention @ full 1024-dim
            score_rgb = (q @ rgb.T) / self.scale
            att_rgb = torch.softmax(score_rgb, dim=-1) @ rgb    # [1024]

            score_dep = (q @ dep.T) / self.scale
            att_dep = torch.softmax(score_dep, dim=-1) @ dep    # [1024]

            score_gdep = (q @ gdep.T) / self.scale
            att_gdep = torch.softmax(score_gdep, dim=-1) @ gdep  # [1024]

            # Concat → shallower regression (no n_masks in features)
            combined = torch.cat([q, att_rgb, att_dep, att_gdep], dim=-1)  # [4096]
            pred = self.regression(combined.unsqueeze(0)).squeeze(-1).squeeze(-1)
            preds.append(pred)

        return F.softplus(torch.stack(preds))  # [B_count]
