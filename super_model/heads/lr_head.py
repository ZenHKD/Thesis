"""
MODULE: Left-Right Head — Binary Tri-Source Scoring

Position: After Decoder, parallel to LM Head
Input:  1. hidden state at <|lr|> token [1024]
        2. Tri-source mask features (exactly 2 masks): rgb, dep, gdep [2, 1024] each
Output: logits over 2 masks [2] — argmax 0 = left, 1 = right

Architecture:
    1. Direct LLM Query: Uses the raw h_token from Qwen as the query, bypassing any visual fusers.
    2. Tri-Source Scoring:
       final_score_i = (h_token · rgb_i + h_token · dep_i + h_token · gdep_i) / √1024

Params: 0 (pure dot product scoring)
"""

import math
import torch
import torch.nn as nn


class LeftRightHead(nn.Module):
    """Binary Tri-Source scoring head for Left/Right classification.

    Uses raw h_token directly from LLM as the semantic query.
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
        h_token: torch.Tensor,         # [1024] — direct from LLM
    ) -> torch.Tensor:
        """
        Args:
            rgb_masks:  [2, 1024] — left, right
            dep_masks:  [2, 1024]
            gdep_masks: [2, 1024]
            h_token:    [1024] — LLM query vector

        Returns:
            [2] — raw logits for [left, right].
        """
        if rgb_masks.shape[0] == 0:
            return torch.zeros(0, device=rgb_masks.device, dtype=rgb_masks.dtype)

        if h_token.dim() == 1:
            h_token = h_token.unsqueeze(0) # [1, 1024]

        # Tri-Source Scoring @ full 1024-dim
        score_rgb  = (h_token @ rgb_masks.T).squeeze(0) / self.scale   # [2]
        score_dep  = (h_token @ dep_masks.T).squeeze(0) / self.scale   # [2]
        score_gdep = (h_token @ gdep_masks.T).squeeze(0) / self.scale  # [2]

        return score_rgb + score_dep + score_gdep  # [2]
