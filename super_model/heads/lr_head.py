"""
MODULE: Left-Right Head — Binary Tri-Source + Visual Context Scoring

Position: After Decoder, parallel to LM Head
Input:  1. hidden state at <|lr|> token [1024] (decoder context / query)
        2. Per-mask RTI features (exactly 2 masks for left/right):
           - rgb_masks:  [2, 1024] (appearance features)
           - dep_masks:  [2, 1024] (depth/spatial features)
           - gdep_masks: [2, 1024] (global depth — surrounding context)
        3. Visual tokens from ViT merger: [N_vis, 1024] (160 tokens)
Output: logits over 2 masks [2] — argmax 0 = left, 1 = right

Architecture (Binary Tri-Source + Visual Context Scoring):
    Identical structure to MCQHead but specialized for binary classification.
    Always expects exactly 2 masks (left object, right object).

    1. Scene Context via Visual Cross-Attention:
       q_scene = h_lr → Linear(1024, 1024)
       scene_ctx = softmax(q_scene · vis_tokens^T / √1024) · vis_tokens  → [1024]

    2. Fused Query:
       q = Linear(concat(h_lr, scene_ctx)) → [1024]  (2048 → 1024)

    3. Tri-Source Scoring:
       score_rgb  = q · rgb_i  / √1024
       score_dep  = q · dep_i  / √1024
       score_gdep = q · gdep_i / √1024
       final_score_i = score_rgb + score_dep + score_gdep

Params: ~3.15M (same structure as MCQHead)
"""

import math
import torch
import torch.nn as nn


class LeftRightHead(nn.Module):
    """Binary Tri-Source + Visual Context scoring head for Left/Right classification.

    Specialized for exactly 2 masks. h_lr (query) first attends to ViT
    visual tokens, then scores the 2 masks via tri-source dot-products.
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
        rgb_masks: torch.Tensor,       # [2, 1024]
        dep_masks: torch.Tensor,       # [2, 1024]
        gdep_masks: torch.Tensor,      # [2, 1024]
        h_token: torch.Tensor,         # [1024] or [1, 1024] — hidden state at <|lr|>
        vis_tokens: torch.Tensor = None,  # [N_vis, 1024] — ViT visual tokens
    ) -> torch.Tensor:
        """
        Args:
            rgb_masks:  [2, 1024] — RGB features per mask (left, right).
            dep_masks:  [2, 1024] — Depth features per mask.
            gdep_masks: [2, 1024] — Global depth features per mask.
            h_token:    [1024] or [1, 1024] — hidden state at <|lr|> position.
            vis_tokens: [N_vis, 1024] — ViT visual tokens (post-merger).

        Returns:
            [2] — raw logits for [left, right].
        """
        if rgb_masks.shape[0] == 0:
            return torch.zeros(0, device=rgb_masks.device, dtype=rgb_masks.dtype)

        if h_token.dim() == 1:
            h_token = h_token.unsqueeze(0)  # [1, 1024]

        # Step 1: Scene context via visual cross-attention
        if vis_tokens is not None and vis_tokens.shape[0] > 0:
            q_scene = self.scene_q_proj(h_token)  # [1, 1024]
            attn_scores = (q_scene @ vis_tokens.T) / self.scale  # [1, N_vis]
            scene_ctx = torch.softmax(attn_scores, dim=-1) @ vis_tokens  # [1, 1024]
        else:
            scene_ctx = torch.zeros_like(h_token)

        # Step 2: Fuse decoder context + scene context
        fused = torch.cat([h_token, scene_ctx], dim=-1)  # [1, 2048]
        q = self.fuse_proj(fused)  # [1, 1024]
        q = self.dropout(q)

        # Step 3: Tri-Source Scoring @ full 1024-dim
        score_rgb  = (q @ rgb_masks.T).squeeze(0) / self.scale   # [2]
        score_dep  = (q @ dep_masks.T).squeeze(0) / self.scale   # [2]
        score_gdep = (q @ gdep_masks.T).squeeze(0) / self.scale  # [2]

        return score_rgb + score_dep + score_gdep  # [2]
