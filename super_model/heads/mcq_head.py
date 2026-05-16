"""
MODULE: MCQ Head — Tri-Source Scoring (uses SharedVisualFuser)

Position: After Decoder, parallel to LM Head
Input:  1. hidden state at <|mcq|> token [1024]
        2. Tri-source mask features: rgb, dep, gdep [N_masks, 1024] each
        3. Visual tokens [N_vis, 1024] (via shared fuser)
Output: logits over N_masks [N_masks]

Architecture:
    1. SharedVisualFuser: h_mcq + vis_tokens → q [1024]
    2. Tri-Source Scoring (no key projection):
       final_score_i = (q · rgb_i + q · dep_i + q · gdep_i) / √1024

Params: ~0 (only uses shared fuser, no private params)
"""

import math
import torch
import torch.nn as nn


class MCQHead(nn.Module):
    """Tri-Source scoring head for MCQ classification.

    Uses external SharedVisualFuser for scene context extraction.
    Only contains task-specific scoring logic (no private params).
    """

    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scale = math.sqrt(hidden_dim)

    def forward(
        self,
        rgb_masks: torch.Tensor,       # [N_masks, 1024]
        dep_masks: torch.Tensor,       # [N_masks, 1024]
        gdep_masks: torch.Tensor,      # [N_masks, 1024]
        q: torch.Tensor,               # [1, 1024] — pre-fused query from SharedVisualFuser
    ) -> torch.Tensor:
        """
        Args:
            rgb_masks:  [N_masks, 1024]
            dep_masks:  [N_masks, 1024]
            gdep_masks: [N_masks, 1024]
            q:          [1, 1024] — fused query (from SharedVisualFuser)

        Returns:
            [N_masks] — raw logits for each mask.
        """
        if rgb_masks.shape[0] == 0:
            return torch.zeros(0, device=rgb_masks.device, dtype=rgb_masks.dtype)

        # Tri-Source Scoring @ full 1024-dim (no key projection)
        score_rgb  = (q @ rgb_masks.T).squeeze(0) / self.scale   # [N_masks]
        score_dep  = (q @ dep_masks.T).squeeze(0) / self.scale   # [N_masks]
        score_gdep = (q @ gdep_masks.T).squeeze(0) / self.scale  # [N_masks]

        return score_rgb + score_dep + score_gdep  # [N_masks]
