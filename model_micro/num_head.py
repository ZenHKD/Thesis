"""
MODULE: Number Head — Multi-Head Cross-Attention Regression for Distance & Count

Position: After Decoder, parallel to LM Head
Input:  1. hidden state at <|num|> token [B_num, 1024] (decoder context)
        2. raw RTI features [N_masks, 3072] per sample (concat of rgb/dep/geo)
Output: continuous scalar predictions [B_num] (non-negative)

Used for:
    - distance tasks: predicts distance in meters (float)
    - count tasks:    predicts count (float, rounded to int at inference)

Architecture (Multi-Head Cross-Attention → Regression):
    Both query (h_num) and keys (raw RTI) are independently projected into a
    shared latent space, then interact via multi-head cross-attention.
    The attended output is concatenated with the query and fed into a
    regression MLP to produce a scalar prediction.

    1. Query projection:  h_num [1024] → Linear(1024, 256) → query [256]
    2. Key/Value proj:    raw_rti [N_masks, 3072] → Linear(3072, 256) → kv [N_masks, 256]
    3. Cross-attention:   MHA(query, kv, kv) → attended [256]
    4. Regression MLP:    concat(query, attended) [512] → Linear → GELU → Linear → softplus → scalar

    h_num carries full context (question + all masks + image) from 24-layer decoder.
    Raw RTI keys provide direct per-mask spatial features (no inter-mask mixing).
    Standard attention (not flash): query is single token, keys = 2-12 masks.

Params: ~1.45M
    query_proj:  LN(1024) + Linear(1024, 256)  = 264,448
    kv_proj:     LN(3072) + Linear(3072, 256)   = 792,064
    cross_attn:  MHA(256, 4 heads)              = 262,400
    regression:  LN(512) + Linear(512, 256) + Linear(256, 1) = 132,609
    Total:       ~1,451,521 ≈ 1.45M
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NumberHead(nn.Module):
    """Multi-Head Cross-Attention head for distance/count regression.

    Pattern: h_num (query) attends to raw_rti (keys/values) → attended context → scalar

    h_num from decoder already encodes full context (question + spatial info).
    Raw RTI provides direct per-mask features without inter-mask mixing,
    preserving each mask's distinct spatial identity.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,       # decoder hidden dim
        mask_feat_dim: int = 3072,    # raw RTI concat dim (rgb+dep+geo)
        proj_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj_dim = proj_dim

        # Project decoder hidden state → query
        self.query_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, proj_dim),
        )

        # Project raw RTI concat → keys/values
        self.kv_proj = nn.Sequential(
            nn.LayerNorm(mask_feat_dim),
            nn.Linear(mask_feat_dim, proj_dim),
        )

        # Cross-attention: h_num attends to mask features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Regression: concat(query, attended, gated_sum) → scalar
        self.regression = nn.Sequential(
            nn.LayerNorm(proj_dim * 3),
            nn.Linear(proj_dim * 3, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, 1),
        )

    def forward(
        self,
        h_num: torch.Tensor,
        rti_list: list,
    ) -> torch.Tensor:
        """
        Args:
            h_num:    [B_num, 1024] — hidden states at <|num|> positions.
                      B_num = number of numeric samples in batch.
            rti_list: list of [N_masks_b, 3072] tensors, one per sample.
                      len(rti_list) == B_num.

        Returns:
            [B_num] — predicted values (non-negative via softplus)
        """
        B_num = h_num.shape[0]
        if B_num == 0:
            return torch.zeros(0, device=h_num.device, dtype=h_num.dtype)

        preds = []
        for i in range(B_num):
            # Query: decoder hidden → [1, 1, proj_dim]
            q = self.query_proj(h_num[i:i+1]).unsqueeze(0)  # [1, 1, proj_dim]

            # Keys/Values: raw RTI concat → [1, N_masks, proj_dim]
            rti_i = rti_list[i]  # [N_masks, mask_feat_dim]
            kv = self.kv_proj(rti_i).unsqueeze(0)  # [1, N_masks, proj_dim]

            # 1. Softmax Cross-attention: Good for averaging features (Distance task)
            attended, _ = self.cross_attn(q, kv, kv)  # [1, 1, proj_dim]

            # 2. Sigmoid Gated Sum: Good for preserving magnitude (Count task)
            q_flat = q.squeeze(0)              # [1, proj_dim]
            kv_flat = kv.squeeze(0)            # [N_masks, proj_dim]
            
            # Calculate independent relevance score for each mask (no sum=1 constraint)
            relevance = torch.matmul(q_flat, kv_flat.transpose(0, 1)) / (self.proj_dim ** 0.5)
            gate = torch.sigmoid(relevance)    # [1, N_masks] (Values 0.0 -> 1.0 for each mask independently)
            
            # Aggregate relevant masks via summation (Preserves magnitude for counting)
            gated_sum = torch.matmul(gate, kv_flat) # [1, proj_dim]

            # 3. Concat query + attended (Softmax) + gated_sum (Sigmoid) → regression
            att_flat = attended.squeeze(0)     # [1, proj_dim]
            combined = torch.cat([q_flat, att_flat, gated_sum], dim=-1)  # [1, 3*proj_dim]

            pred = self.regression(combined).squeeze(-1)  # [1]
            preds.append(pred)

        return F.softplus(torch.cat(preds))  # [B_num]
