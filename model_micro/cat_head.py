"""
MODULE: Category Head — Bilinear Attention Scoring for MCQ & Left/Right

Position: After Decoder, parallel to LM Head and NumberHead
Input:  1. hidden state at <|cat|> token [1024] (decoder context / query)
        2. raw RTI features [N_masks, 3072] (per-mask spatial features)
Output: logits over N_masks [N_masks] — argmax = selected mask index

Used for:
    - mcq tasks:        score N candidate masks → argmax = region index answer
    - left_right tasks: score 2 candidate masks → determine spatial relationship

Architecture (Bilinear Attention → Scoring):
    Both query (h_cat) and keys (raw RTI) are independently projected into a
    shared latent space, then interact via scaled dot-product to produce
    per-mask selection scores.

    1. Query projection:  h_cat [1024] → Linear(1024, 256) → query [256]
    2. Key projection:    raw_rti [N_masks, 3072] → Linear(3072, 256) → keys [N_masks, 256]
    3. Scoring:           dot(query, key_i) / sqrt(256) → scores [N_masks]

    h_cat from decoder already encodes full context (question + spatial info).
    Raw RTI keys provide direct per-mask spatial features (no inter-mask mixing),
    preserving each mask's distinct identity for accurate classification.

Params: ~1.06M
    query_proj: LayerNorm(1024) + Linear(1024, 256) = 264,448
    key_proj:   LayerNorm(3072) + Linear(3072, 256)  = 792,064
    Total:      ~1,056,512 ≈ 1.06M
"""

import math
import torch
import torch.nn as nn


class CategoryHead(nn.Module):
    """Bilinear Attention head for MCQ / Left-Right classification.

    Pattern: h_cat (query) × raw_rti (keys) → per-mask scores

    h_cat from decoder already encodes full context (question + spatial info).
    Raw RTI provides direct per-mask features without inter-mask mixing,
    preserving each mask's distinct spatial identity.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,       # decoder hidden dim
        mask_feat_dim: int = 3072,    # raw RTI concat dim (rgb+dep+geo)
        proj_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.proj_dim = proj_dim
        self.scale = math.sqrt(proj_dim)

        # Project h_cat (full context) → query
        self.query_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, proj_dim),
        )

        # Project raw RTI features → keys
        self.key_proj = nn.Sequential(
            nn.LayerNorm(mask_feat_dim),
            nn.Linear(mask_feat_dim, proj_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, h_masks: torch.Tensor, h_cat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_masks: [N_masks, 3072] — raw RTI features per mask.
            h_cat:   [1024] or [1, 1024] — hidden state at <|cat|> position (query).

        Returns:
            [N_masks] — raw logits (scaled dot-product scores) for each mask.
        """
        if h_masks.shape[0] == 0:
            return torch.zeros(0, device=h_masks.device, dtype=h_masks.dtype)

        if h_cat.dim() == 1:
            h_cat = h_cat.unsqueeze(0)  # [1, 1024]

        # Query: [1, 1024] → [1, 256]
        query = self.query_proj(h_cat)       # [1, proj_dim]
        query = self.dropout(query)

        # Keys: [N_masks, 3072] → [N_masks, 256]
        keys = self.key_proj(h_masks)        # [N_masks, proj_dim]
        keys = self.dropout(keys)

        # Scoring: dot(query, key_i) / sqrt(d) → [N_masks]
        scores = torch.matmul(query, keys.T).squeeze(0) / self.scale  # [N_masks]

        return scores
