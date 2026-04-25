"""
MODULE: Category Head — Classification for MCQ & Left/Right Tasks

Position: After Decoder, parallel to LM Head and NumberHead
Input:  hidden states at <|cat|> token position [1024] (query/context)
        + hidden states at ALL mask positions [N_masks, 3072] (candidates, 3 RTI tokens concat)
Output: logits over N_masks [N_masks] — argmax = selected mask index

Used for:
    - mcq tasks:        score N candidate masks → argmax = region index answer
    - left_right tasks: score 2 candidate masks → determine spatial relationship

Both tasks use CrossEntropy loss during training.

Architecture (Bilinear Attention):
    Query projection:  h_cat  [1024] -> Linear(1024, 256)  -> query [256]
    Key projection:    h_mask [3072] -> Linear(3072, 256)   -> key   [256]
    Score:             dot(query, key) / sqrt(256) for each mask
    
    The query (h_cat) has full causal context (question + reasoning).
    The keys (h_masks) concatenate all 3 RTI token hidden states:
        [region_rgb | region_depth | region_geo] = [1024 + 1024 + 1024] = 3072
    This preserves distinct RGB, depth, and geometry information per mask.

Params: ~1.05M
    query_proj: LayerNorm(1024) + Linear(1024, 256) = 2*1024 + 1024*256 + 256 = 264,448
    key_proj:   LayerNorm(3072) + Linear(3072, 256) = 2*3072 + 3072*256 + 256 = 792,832
    Total:      ~1,057,280 ≈ 1.06M
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CategoryHead(nn.Module):
    """Bilinear attention scorer for MCQ / Left-Right tasks.

    Uses a bilinear attention mechanism:
    
        score_i = dot(W_q @ h_cat, W_k @ h_mask_i) / sqrt(d)
    
    h_cat (at <|cat|> position) has seen the ENTIRE sequence including the
    question and reasoning, so it carries all the semantic context needed.
    h_masks concatenate all 3 RTI token hidden states per mask, preserving
    distinct RGB, depth, and geometry information for discrimination.
    """

    def __init__(self, hidden_dim: int = 1024, mask_dim: int = 3072,
                 proj_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.proj_dim = proj_dim
        self.scale = math.sqrt(proj_dim)
        
        # Project h_cat (full context) into query space
        self.query_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, proj_dim),
        )
        
        # Project each h_mask (3 RTI tokens concat) into key space
        self.key_proj = nn.Sequential(
            nn.LayerNorm(mask_dim),
            nn.Linear(mask_dim, proj_dim),
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_masks: torch.Tensor, h_cat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_masks: [N_masks, 3072] — concatenated hidden states of 3 RTI tokens per mask.
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
        
        # Keys: [N_masks, 3072] -> [N_masks, 256]  
        keys = self.key_proj(h_masks)        # [N_masks, proj_dim]
        keys = self.dropout(keys)
        
        # Scaled dot-product: [1, 256] @ [256, N_masks] -> [1, N_masks] -> [N_masks]
        scores = torch.matmul(query, keys.T).squeeze(0) / self.scale  # [N_masks]
        
        return scores
