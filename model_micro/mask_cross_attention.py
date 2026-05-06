"""
MODULE: Mask Cross-Attention — Pairwise Relationship Encoding

Position: Between RTI output and Dual Heads (Number Head + Category Head)
Input:  RTI tokens per mask [N_masks, 3072] (concat of rgb/dep/geo projections)
Output: Context-aware mask features [N_masks, hidden_dim] (each mask "sees" all others)

Purpose:
    Individual RTI tokens encode per-mask features (depth, position, color).
    But tasks like "which pallet is nearest to transporter" require PAIRWISE
    relationships between masks (relative positions, relative depths).
    
    This module adds self-attention among masks so each mask's representation
    encodes its relationship to ALL other masks in the scene.

Architecture:
    1. Input projection:  Linear(3072, hidden_dim) + LayerNorm
    2. Self-attention:    standard scaled dot-product (N_masks = 2-12, too small for flash attention)
    3. FFN:              Linear(hidden_dim, hidden_dim*2) -> GELU -> Linear -> residual
    4. Output LayerNorm

Params (~3.7M with hidden_dim=512, num_heads=4):
    input_proj:   3072 × 512 + 512 + 2×512  = ~1.58M
    self_attn:    4 × (512 × 128) × 3 + 512  = ~0.79M  (Q,K,V + out_proj)
    ffn:          512 × 1024 + 1024 × 512     = ~1.05M
    norms:        2 × 2 × 512                 = ~2K
    Total:        ~3.42M
"""

import torch
import torch.nn as nn

class MaskCrossAttention(nn.Module):
    """Self-attention among mask representations for pairwise reasoning.

    Each mask attends to all other masks in the scene, encoding:
    - Relative positions (centroid differences)
    - Relative depths (which is closer/farther)
    - Spatial layout (left-of, right-of, between)

    Uses standard scaled dot-product attention (NOT flash attention)
    because N_masks = 2-12 is too small for flash to be beneficial.
    """

    def __init__(
        self,
        input_dim: int = 3072,     # concat of [rgb(1024) + dep(1024) + geo(1024)]
        hidden_dim: int = 512,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Project concatenated RTI features to hidden dim
        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
        )

        # Pre-norm self-attention among masks
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,   # input shape: [batch, seq, dim]
        )

        # Pre-norm FFN with expansion factor 2
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, rti_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rti_tokens: [N_masks, 3072] — concatenated RTI tokens per mask.
                        N_masks is variable (2-12 typically).

        Returns:
            [N_masks, hidden_dim] — pairwise-aware mask features.
            Each mask's output encodes its relationship to all other masks.
        """
        if rti_tokens.shape[0] == 0:
            return torch.zeros(
                0, self.hidden_dim,
                device=rti_tokens.device, dtype=rti_tokens.dtype,
            )

        # Project: [N_masks, 3072] -> [N_masks, hidden_dim]
        x = self.input_proj(rti_tokens)

        # Add batch dim for MHA: [1, N_masks, hidden_dim]
        x = x.unsqueeze(0)

        # Self-attention (pre-norm): each mask attends to all masks
        residual = x
        x_norm = self.attn_norm(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = residual + attn_out

        # FFN (pre-norm)
        residual = x
        x_norm = self.ffn_norm(x)
        x = residual + self.ffn(x_norm)

        # Remove batch dim: [1, N_masks, hidden_dim] -> [N_masks, hidden_dim]
        return x.squeeze(0)
