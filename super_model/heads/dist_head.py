"""
MODULE: Distance Head — Tri-Source Cross-Attention Regression (uses SharedVisualFuser)

Position: After Decoder, parallel to LM Head
Input:  1. hidden state at <|dist|> token [B_dist, 1024]
        2. Tri-source mask features: rgb, dep, gdep lists of [N_masks, 1024]
        3. Visual tokens [N_vis, 1024] (via shared fuser)
Output: continuous scalar predictions [B_dist] (non-negative, meters)

Architecture:
    1. Query-Guided Gating: Uses q to generate a sigmoid mask to filter out visual noise (e.g., color) from att_rgb, att_dep, att_gdep.
    2. Transformer-style Residual Fusion: clean visual features are projected and added (res) to q, rather than concatenated.
    3. Regression MLP: q_fused [1024] → Linear(1024, 512) → GELU → Dropout → Linear(512, 1) → relu

Params: ~6.5M 
    gates & projs: 6 * 1M = ~6.2M
    regression: 1024*512 + 512*1 = ~0.5M
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

        # Query-guided gates to filter out visual noise (e.g. color/texture)
        self.gate_rgb  = nn.Linear(hidden_dim, hidden_dim)
        self.gate_dep  = nn.Linear(hidden_dim, hidden_dim)
        self.gate_gdep = nn.Linear(hidden_dim, hidden_dim)

        # Projections to align visual features for residual addition
        self.proj_rgb  = nn.Linear(hidden_dim, hidden_dim)
        self.proj_dep  = nn.Linear(hidden_dim, hidden_dim)
        self.proj_gdep = nn.Linear(hidden_dim, hidden_dim)

        # Fusion normalization
        self.fusion_norm = nn.LayerNorm(hidden_dim)

        # Lightweight Regression MLP: q_fused -> scalar
        self.regression = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
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
            [B_dist] — predicted distances (non-negative via relu)
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

            # Tri-Source Attention & Filtering
            score_rgb = (q @ rgb.T) / self.scale
            att_rgb = torch.softmax(score_rgb, dim=-1) @ rgb    # [1024]
            clean_rgb = att_rgb * torch.sigmoid(self.gate_rgb(q)) # Filter noise
            res_rgb = self.proj_rgb(clean_rgb)

            score_dep = (q @ dep.T) / self.scale
            att_dep = torch.softmax(score_dep, dim=-1) @ dep    # [1024]
            clean_dep = att_dep * torch.sigmoid(self.gate_dep(q))
            res_dep = self.proj_dep(clean_dep)

            score_gdep = (q @ gdep.T) / self.scale
            att_gdep = torch.softmax(score_gdep, dim=-1) @ gdep  # [1024]
            clean_gdep = att_gdep * torch.sigmoid(self.gate_gdep(q))
            res_gdep = self.proj_gdep(clean_gdep)

            # Residual Addition Fusion (Transformer-style)
            q_fused = self.fusion_norm(q + res_rgb + res_dep + res_gdep)

            pred = self.regression(q_fused.unsqueeze(0)).squeeze(-1).squeeze(-1)
            preds.append(pred)

        return torch.relu(torch.stack(preds))  # [B_dist]
