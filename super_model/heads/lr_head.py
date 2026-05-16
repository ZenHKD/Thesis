"""
MODULE: Left-Right Head — Binary Tri-Source Scoring (uses SharedVisualFuser)

Position: After Decoder, parallel to LM Head
Input:  1. hidden state at <|lr|> token [1024]
        2. Tri-source mask features (exactly 2 masks): rgb, dep, gdep [2, 1024] each
        3. Visual tokens [N_vis, 1024] (via shared fuser)
Output: logits over 2 masks [2] — argmax 0 = left, 1 = right

Architecture:
    1. SharedVisualFuser: h_lr + vis_tokens → q [1024]
    2. Tri-Source Scoring:
       final_score_i = (q · rgb_i + q · dep_i + q · gdep_i) / √1024

Params: ~0 (only uses shared fuser, no private params)
"""

import math
import torch
import torch.nn as nn


class LeftRightHead(nn.Module):
    """Binary Tri-Source scoring head for Left/Right classification.

    Uses external SharedVisualFuser for scene context extraction.
    Identical logic to MCQHead but specialized for exactly 2 masks.
    """

    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scale = math.sqrt(hidden_dim)

    def forward(
        self,
        rgb_masks: torch.Tensor,       # [2, 1024]
        dep_masks: torch.Tensor,       # [2, 1024]
        gdep_masks: torch.Tensor,      # [2, 1024]
        q: torch.Tensor,               # [1, 1024] — pre-fused query from SharedVisualFuser
    ) -> torch.Tensor:
        """
        Args:
            rgb_masks:  [2, 1024] — left, right
            dep_masks:  [2, 1024]
            gdep_masks: [2, 1024]
            q:          [1, 1024] — fused query (from SharedVisualFuser)

        Returns:
            [2] — raw logits for [left, right].
        """
        if rgb_masks.shape[0] == 0:
            return torch.zeros(0, device=rgb_masks.device, dtype=rgb_masks.dtype)

        # Tri-Source Scoring @ full 1024-dim
        score_rgb  = (q @ rgb_masks.T).squeeze(0) / self.scale   # [2]
        score_dep  = (q @ dep_masks.T).squeeze(0) / self.scale   # [2]
        score_gdep = (q @ gdep_masks.T).squeeze(0) / self.scale  # [2]

        return score_rgb + score_dep + score_gdep  # [2]
