"""
MODULE: Distance Head — Tri-Source Cross-Attention Regression (uses SharedVisualFuser)

Position: After Decoder, parallel to LM Head
Input:  1. hidden state at <|dist|> token [B_dist, 1024]
        2. Tri-source mask features: rgb, dep, gdep lists of [N_masks, 1024]
        3. Visual tokens [N_vis, 1024] (via shared fuser)
Output: continuous scalar predictions [B_dist] (non-negative, meters)

Architecture:
    1. SharedVisualFuser: h_dist + vis_tokens → q [1024]
    2. Tri-Source Attention:
       att_rgb  = softmax(q · rgb^T  / √1024) · rgb   → [1024]
       att_dep  = softmax(q · dep^T  / √1024) · dep   → [1024]
       att_gdep = softmax(q · gdep^T / √1024) · gdep  → [1024]
    3. Regression MLP:
       concat(q, att_rgb, att_dep, att_gdep) [4096]
       → LN → Linear(4096, 1024) → GELU → Dropout → Linear(1024, 1) → relu

Params: ~4.20M (private regression MLP only)
    regression: LN(4096) + Linear(4096,1024) + Linear(1024,1) = ~4,204,545
"""

import math
import torch
import torch.nn as nn


class DistanceHead(nn.Module):
    """Tri-Source cross-attention regression head for distance prediction.

    Uses external SharedVisualFuser for scene context extraction.
    Private params: regression MLP only.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scale = math.sqrt(hidden_dim)

        # Regression MLP: concat(q, att_rgb, att_dep, att_gdep) → scalar
        self.regression = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        h_token: torch.Tensor,         # [B_dist, 1024]
        rgb_list: list,                 # list of [N_masks, 1024]
        dep_list: list,                 # list of [N_masks, 1024]
        gdep_list: list,                # list of [N_masks, 1024]
        q_list: list,                   # list of [1, 1024] — pre-fused queries from SharedVisualFuser
    ) -> torch.Tensor:
        """
        Args:
            h_token:   [B_dist, 1024] — hidden states at <|dist|> positions.
            rgb_list:  list of [N_masks_b, 1024] tensors.
            dep_list:  list of [N_masks_b, 1024] tensors.
            gdep_list: list of [N_masks_b, 1024] tensors.
            q_list:    list of [1, 1024] — fused queries from SharedVisualFuser.

        Returns:
            [B_dist] — predicted distances (non-negative via relu)
        """
        B_dist = h_token.shape[0]
        if B_dist == 0:
            return torch.zeros(0, device=h_token.device, dtype=h_token.dtype)

        preds = []
        for i in range(B_dist):
            q = q_list[i].squeeze(0)  # [1024]

            rgb  = rgb_list[i]   # [N_masks, 1024]
            dep  = dep_list[i]   # [N_masks, 1024]
            gdep = gdep_list[i]  # [N_masks, 1024]

            # Tri-Source Attention @ full 1024-dim
            score_rgb = (q @ rgb.T) / self.scale
            att_rgb = torch.softmax(score_rgb, dim=-1) @ rgb    # [1024]

            score_dep = (q @ dep.T) / self.scale
            att_dep = torch.softmax(score_dep, dim=-1) @ dep    # [1024]

            score_gdep = (q @ gdep.T) / self.scale
            att_gdep = torch.softmax(score_gdep, dim=-1) @ gdep  # [1024]

            # Concat all sources + query → regression
            combined = torch.cat([q, att_rgb, att_dep, att_gdep], dim=-1)  # [4096]
            pred = self.regression(combined.unsqueeze(0)).squeeze(-1).squeeze(-1)
            preds.append(pred)

        return torch.relu(torch.stack(preds))  # [B_dist]
