"""
MODULE: SharedVisualFuser — Shared Scene Cross-Attention + Fuse Projection

This module is shared across all 4 task heads (MCQ, LR, Distance, Count).
All 4 heads perform the same scene context extraction:
    1. Cross-attend h_token over 160 ViT visual tokens → scene_ctx
    2. Fuse h_token + scene_ctx → enriched query q

By sharing these layers, we:
    - Save ~9.44M params (3 redundant copies eliminated)
    - Reduce 3 redundant cross-attention ops per forward pass
    - Ensure consistent scene understanding across all tasks

Params: ~3.15M (shared by all 4 heads)
    scene_q_proj: Linear(1024, 1024, bias=False)  = 1,048,576
    fuse_proj:    Linear(2048, 1024) + GELU        = 2,098,176
    Total:        3,146,752 ≈ 3.15M
"""

import math
import torch
import torch.nn as nn


class SharedVisualFuser(nn.Module):
    """Shared scene cross-attention + fusion for all 4 task heads.

    Input:
        h_token:    [1024] or [1, 1024] — hidden state at any special token
        vis_tokens: [N_vis, 1024] — ViT visual tokens (160 tokens post-merger)

    Output:
        q: [1, 1024] — enriched query fusing decoder context + scene context
        scene_ctx: [1, 1024] — scene context (for heads that need it separately)
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scale = math.sqrt(hidden_dim)

        # Scene context: cross-attention query projection
        self.scene_q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Fuse decoder context + scene context → scoring query
        self.fuse_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h_token: torch.Tensor,            # [1024] or [1, 1024]
        vis_tokens: torch.Tensor = None,   # [N_vis, 1024]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            q:         [1, 1024] — fused query (decoder + scene)
            scene_ctx: [1, 1024] — scene context vector
        """
        if h_token.dim() == 1:
            h_token = h_token.unsqueeze(0)  # [1, 1024]

        # Scene context via visual cross-attention
        if vis_tokens is not None and vis_tokens.shape[0] > 0:
            q_scene = self.scene_q_proj(h_token)  # [1, 1024]
            attn_scores = (q_scene @ vis_tokens.T) / self.scale  # [1, N_vis]
            scene_ctx = torch.softmax(attn_scores, dim=-1) @ vis_tokens  # [1, 1024]
        else:
            scene_ctx = torch.zeros_like(h_token)

        # Fuse decoder context + scene context
        fused = torch.cat([h_token, scene_ctx], dim=-1)  # [1, 2048]
        q = self.fuse_proj(fused)  # [1, 1024]
        q = self.dropout(q)

        return q, scene_ctx
