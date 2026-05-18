"""
MODULE: MCQ Head — Tri-Source Scoring

Position: After Decoder, parallel to LM Head
Input:  1. hidden state at <|mcq|> token [1024]
        2. Tri-source mask features: rgb, dep, gdep [N_masks, 1024] each
Output: logits over N_masks [N_masks]

Architecture:
    1. Direct LLM Query: Uses the raw h_token from Qwen as the query, bypassing any visual fusers.
    2. Tri-Source Scoring (no key projection):
       final_score_i = (h_token · rgb_i + h_token · dep_i + h_token · gdep_i) / √1024

Params: 0 (pure dot product scoring)
"""

import math
import torch
import torch.nn as nn


class MCQHead(nn.Module):
    """Tri-Source scoring head for MCQ classification.

    Uses raw h_token directly from LLM as the semantic query.
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
        h_token: torch.Tensor,         # [1024] — direct from LLM
    ) -> torch.Tensor:
        """
        Args:
            rgb_masks:  [N_masks, 1024]
            dep_masks:  [N_masks, 1024]
            gdep_masks: [N_masks, 1024]
            h_token:    [1024] — LLM query vector

        Returns:
            [N_masks] — raw logits for each mask.
        """
        if rgb_masks.shape[0] == 0:
            return torch.zeros(0, device=rgb_masks.device, dtype=rgb_masks.dtype)

        if h_token.dim() == 1:
            h_token = h_token.unsqueeze(0) # [1, 1024]

        # Tri-Source Scoring @ full 1024-dim (no key projection)
        score_rgb  = (h_token @ rgb_masks.T).squeeze(0) / self.scale   # [N_masks]
        score_dep  = (h_token @ dep_masks.T).squeeze(0) / self.scale   # [N_masks]
        score_gdep = (h_token @ gdep_masks.T).squeeze(0) / self.scale  # [N_masks]

        return score_rgb + score_dep + score_gdep  # [N_masks]
