"""
MODULE: Category Head — Classification for MCQ & Left/Right Tasks

Position: After Decoder, parallel to LM Head and NumberHead
Input:  1. hidden state at <|cat|> token [1024] (query/context from decoder)
        2. cross-attended RTI features [N_masks, 512] (pairwise-aware mask keys)
Output: logits over N_masks [N_masks] — argmax = selected mask index

Used for:
    - mcq tasks:        score N candidate masks → argmax = region index answer
    - left_right tasks: score 2 candidate masks → determine spatial relationship

Changes:
    - Keys are now cross-attended RTI (512-dim, pairwise-aware)
      instead of raw RTI concat (3072-dim, individual features only)
    - Each mask key "knows" about all other masks → enables relational reasoning
      e.g., "which pallet is nearest to transporter" now possible

Architecture (Bilinear Attention):
    Query projection:  h_cat  [1024] -> Linear(1024, 256)  -> query [256]
    Key projection:    cross_rti [N_masks, 512] -> Linear(512, 256) -> key [N_masks, 256]
    Score:             dot(query, key) / sqrt(256) for each mask

Params: ~0.40M
    query_proj: LayerNorm(1024) + Linear(1024, 256) = 2*1024 + 1024*256 + 256 = 264,448
    key_proj:   LayerNorm(512) + Linear(512, 256)   = 2*512 + 512*256 + 256   = 132,352
    Total:      ~396,800 ≈ 0.40M
"""

import math
import torch
import torch.nn as nn


class CategoryHead(nn.Module):
    """Bilinear attention scorer for MCQ / Left-Right tasks.

    Uses a bilinear attention mechanism:

        score_i = dot(W_q @ h_cat, W_k @ cross_rti_i) / sqrt(d)

    h_cat (at <|cat|> position) has seen the ENTIRE sequence including the
    question, so it carries all the semantic context needed.
    cross_rti keys are pairwise-aware: each mask "knows" about all other masks
    via MaskCrossAttention, enabling relational reasoning.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,       # decoder hidden dim
        mask_feat_dim: int = 512,     # MaskCrossAttention output dim
        proj_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.proj_dim = proj_dim
        self.scale = math.sqrt(proj_dim)

        # Project h_cat (full context) into query space
        self.query_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, proj_dim),
        )

        # Project cross-attended RTI features into key space
        self.key_proj = nn.Sequential(
            nn.LayerNorm(mask_feat_dim),
            nn.Linear(mask_feat_dim, proj_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, h_masks: torch.Tensor, h_cat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_masks: [N_masks, 512] — cross-attended RTI features per mask.
            h_cat:   [1024] or [1, 1024] — hidden state at <|cat|> position (query).

        Returns:
            [N_masks] — raw logits (scaled dot-product scores) for each mask.
        """
        if h_masks.shape[0] == 0:
            return torch.zeros(0, device=h_masks.device, dtype=h_masks.dtype)

        if h_cat.dim() == 1:
            h_cat = h_cat.unsqueeze(0)  # [1, 1024]

        # Query: [1, 1024] -> [1, 256]
        query = self.query_proj(h_cat)       # [1, proj_dim]
        query = self.dropout(query)

        # Keys: [N_masks, 512] -> [N_masks, 256]
        keys = self.key_proj(h_masks)        # [N_masks, proj_dim]
        keys = self.dropout(keys)

        # Scaled dot-product: [1, 256] @ [256, N_masks] -> [1, N_masks] -> [N_masks]
        scores = torch.matmul(query, keys.T).squeeze(0) / self.scale  # [N_masks]

        return scores
