"""
MODULE: SharedVisualFuser — Dual-Stream Scene Cross-Attention + Fuse Projection

This module is shared across all 4 task heads (MCQ, LR, Distance, Count).
All 4 heads perform the same scene context extraction:
    1. Concat RGB + Depth visual tokens: [160, 1024] || [160, 1024] → [160, 2048]
    2. Project to unified space: Linear(2048, 1024) → [160, 1024]
    3. Cross-attend h_token over fused visual tokens → scene_ctx
    4. Fuse h_token + scene_ctx → enriched query q

By sharing these layers, we:
    - Save ~9.44M params (3 redundant copies eliminated)
    - Reduce 3 redundant cross-attention ops per forward pass
    - Ensure consistent scene understanding across all tasks
    - Provide joint RGB+Depth visual context without information mixing

Params: ~5.24M (shared by all 4 heads)
    dual_proj:    Linear(2048, 1024) + GELU        = 2,098,176
    scene_q_proj: Linear(1024, 1024, bias=False)   = 1,048,576
    fuse_proj:    Linear(2048, 1024) + GELU        = 2,098,176
    Total:        5,244,928 ≈ 5.24M
"""

import math
import torch
import torch.nn as nn


class SharedVisualFuser(nn.Module):
    """Shared dual-stream scene cross-attention + fusion for all 4 task heads.

    Input:
        h_token:         [1024] or [1, 1024] — hidden state at any special token
        vis_rgb_tokens:  [N_vis, 1024] — RGB ViT visual tokens (160 tokens post-merger)
        vis_dep_tokens:  [N_vis, 1024] — Depth ViT visual tokens (160 tokens post-merger)

    Output:
        q: [1, 1024] — enriched query fusing decoder context + dual-stream scene context
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

        # Dual-stream projection: concat [RGB; Depth] → unified visual space
        self.dual_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )

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
        h_token: torch.Tensor,                 # [1024] or [1, 1024]
        vis_rgb_tokens: torch.Tensor = None,    # [N_vis, 1024]
        vis_dep_tokens: torch.Tensor = None,    # [N_vis, 1024]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            q:         [1, 1024] — fused query (decoder + dual-stream scene)
            scene_ctx: [1, 1024] — scene context vector
        """
        if h_token.dim() == 1:
            h_token = h_token.unsqueeze(0)  # [1, 1024]

        # Dual-stream scene context via visual cross-attention
        has_rgb = vis_rgb_tokens is not None and vis_rgb_tokens.shape[0] > 0
        has_dep = vis_dep_tokens is not None and vis_dep_tokens.shape[0] > 0

        if has_rgb and has_dep:
            # Concat along feature dim: [N_vis, 2048]
            dual_vis = torch.cat([vis_rgb_tokens, vis_dep_tokens], dim=-1)
            # Project to unified space: [N_vis, 1024]
            vis_tokens = self.dual_proj(dual_vis)
        elif has_rgb:
            # Fallback: only RGB available (pad depth with zeros)
            dual_vis = torch.cat([vis_rgb_tokens, torch.zeros_like(vis_rgb_tokens)], dim=-1)
            vis_tokens = self.dual_proj(dual_vis)
        elif has_dep:
            # Fallback: only Depth available (pad RGB with zeros)
            dual_vis = torch.cat([torch.zeros_like(vis_dep_tokens), vis_dep_tokens], dim=-1)
            vis_tokens = self.dual_proj(dual_vis)
        else:
            vis_tokens = None

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
