"""
MODULE: Geometry Self-Attention (GSA) - DFormerv2 Full_GSA (CVPR 2025)
Source: https://arxiv.org/abs/2504.04701
        https://github.com/VCIP-RGBD/DFormer

Position: After Qwen Vision Encoder (merger), before Concat Fusion
Input:  visual_tokens [B, N, 1024] + depth_map [B, H, W]
Output: geometry-aware visual tokens [B, N, 1024]

Key components (faithful to DFormerv2):
    GeoPriorGen   - RoPE + exponential depth/pos decay  (shared per block)
    Full_GSA      - Full NxN attention + lepe (DWConv2d) + geometry prior
    FFN - FC1 -> GELU -> DWConv -> FC2  (with subconv)
    GeometrySelfAttention - 2 x (cnn_pos_encode + norm + Full_GSA + FFN)

Differences from original DFormerv2:
    - Input/output: [B, N, D] flat tokens (not [B, H, W, D] spatial)
      -> reshape internally to [B, H, W, D] using (h_patches, w_patches)
    - Depth patchified via adaptive_avg_pool2d (any resolution, any HxW)
    - No DropPath, no layer scale (simplify for this task)
    - GeoPriorGen is NOT shared -- each block has its own (matches DFormerv2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def angle_transform(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to x"""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return (x * cos) - (torch.stack([-x2, x1], dim=-1).flatten(-2) * sin)


class DWConv2d(nn.Module):
    """Depth-wise Conv2d operating on [B, H, W, C] tensors """

    def __init__(self, dim: int, kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, H, W, C] -> [B, C, W, H] -> [B, H, W, C]"""
        x = x.permute(0, 3, 1, 2)      # [B, C, H, W]
        x = self.dwconv(x)             # [B, C, H, W]
        return x.permute(0, 2, 3, 1)   # [B, H, W, C]


# -----------------------------------------------------------------------------
# GeoPriorGen - Geometry Prior Generator
# -----------------------------------------------------------------------------

class GeoPriorGen(nn.Module):
    """Generate geometry prior from depth map.

    Combines:
    + Spatial decay : Manhattan distance between path positions
    + Depth decay   : |d_i - d_j| between patchified depth values

    Both use learnable per-head exponential decay rates.
    RoPE angles for Q/K rotational encoding.

    Params: weight [2, 1, 1, 1] (learnable blend)
            angle  [D/heads]    (buffer, RoPE)
            decay  [num_heads]  (buffer, decay rates)
    """

    def __init__(self, embed_dim: int, num_heads: int, initial_value: float = 2.0, heads_range: float = 4.0):
        super().__init__()
        # RoPE angles
        angle = 1.0 / (10_000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.register_buffer("angle", angle)

        # Per-head exponential decay rates (non-learnable)
        decay = torch.log(
            1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads)
        )
        self.register_buffer("decay", decay)

        # Learnable blend weight: [0] = spatial, [1] = depth
        self.weight = nn.Parameter(torch.ones(2, 1, 1, 1), requires_grad=True)

    def _rope(self, H: int, W: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute RoPE sin/cos for HxW grid."""
        index = torch.arange(H * W, device=self.angle.device)
        sin = torch.sin(index[:, None] * self.angle[None, :])           # [N, D/heads]
        sin = sin.reshape(H, W, -1)                                     # [H, W, D/heads]
        cos = torch.cos(index[:, None] * self.angle[None, :])
        cos = cos.reshape(H, W, -1)
        return sin, cos

    def _pos_decay(self, H: int, W: int) -> torch.Tensor:
        """Manhattan distance decay: [num_heads, N, N]."""
        index_h = torch.arange(H, device=self.decay.device).float()
        index_w = torch.arange(W, device=self.decay.device).float()
        gh, gw = torch.meshgrid(index_h, index_w, indexing='ij')
        grid = torch.stack([gh.reshape(-1), gw.reshape(-1)], dim=-1)    # [N, 2]
        diff = (grid[:, None, :] - grid[None, :, :]).abs().sum(-1)      # [N, N]
        return diff * self.decay[:, None, None]                         # [heads, N, N]

    def _depth_decay(self, depth_grid: torch.Tensor) -> torch.Tensor:
        """Depth distance decay. depth_grid: [B, 1, H, W] -> [B, heads, N, N]."""
        B, _, H, W = depth_grid.shape
        d = depth_grid.reshape(B, H * W, 1)                             # [B, N, 1]
        diff = (d[:, :, None, :] - d[:, None, :, :]).abs().squeeze(-1)  # [B, N, N]
        return diff.unsqueeze(1) * self.decay[None, :, None, None]      # [B, heads, N, N]

    def forward(self, HW: Tuple[int, int], depth_map: torch.Tensor) -> tuple:
        """
        Args:
            HW: (h_patches, w_patches)
            depth_map: [B, 1,  H_orig, W_orig] or [B, H, W] raw depth
        Returns:
            ((sin, cos), mask) for Full_GSA
            sin/cos: [H, W, D/heads]
            mask:    [B, heads, N, N]
        """
        H, W = HW
        # Patchify depth to (H, W) via adaptive pool
        if depth_map.dim() == 3:
            depth_map = depth_map.unsqueeze(1)                          # [B, 1, H_d, W_d]
        depth_grid = F.adaptive_avg_pool2d(depth_map.float(), (H, W))   # [B, 1, H, W]
        depth_grid = depth_grid.to(depth_map.dtype)

        sin, cos = self._rope(H, W)
        pos_mask   = self._pos_decay(H, W)                              # [heads, N, N]
        depth_mask = self._depth_decay(depth_grid)                      # [B, heads, N, N]

        # Blend: weight[0]*spatial + weight[1]*depth
        mask = self.weight[0] * pos_mask.unsqueeze(0) + self.weight[1] * depth_mask  # [B, heads, N, N]

        return (sin, cos), mask


# -----------------------------------------------------------------------------
# Full_GSA - Full NxN Geometry Self-Attention
# -----------------------------------------------------------------------------

class Full_GSA(nn.Module):
    """Full NxN attention + geometry prior + RoPE + lepe.

    Faithful to DFormerv2 Full_GSA. Input: [B, H, W, C].
    Per block: (embed_dim=1024, num_heads=8)
    """

    def __init__(self, embed_dim: int, num_heads: int, value_factor: int = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.key_dim   = embed_dim // num_heads
        self.scaling   = self.key_dim ** -0.5

        self.q_proj  = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj  = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj  = nn.Linear(embed_dim, embed_dim * value_factor, bias=True)
        self.lepe    = DWConv2d(embed_dim * value_factor, 5, 1, 2)
        self.out_proj = nn.Linear(embed_dim * value_factor, embed_dim, bias=True)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight,  gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight,  gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight,  gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x: torch.Tensor, rel_pos: tuple) -> torch.Tensor:
        """
        Args:
            x:       [B, H, W, C]
            rel_pos: ((sin, cos), mask)  from GeoPriorGen
        Returns:
            [B, H, W, C]
        """
        B, H, W, _ = x.shape
        (sin, cos), mask = rel_pos                                      # mask: [B, heads, N, N]
        N = H * W

        q = self.q_proj(x)                                              # [B, H, W, D]
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)                                             # [B, H, W, D] -- local position enhancement

        k = k * self.scaling

        # Reshape for multi-head: [B, heads, H, W, key_dim]
        q = q.view(B, H, W, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)
        k = k.view(B, H, W, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)

        # Apply RoPE: sin/cos are [H, W, key_dim]
        sin_ = sin.unsqueeze(0).unsqueeze(0)                            # [1, 1, H, W, key_dim]
        cos_ = cos.unsqueeze(0).unsqueeze(0)
        q = angle_transform(q, sin_, cos_)
        k = angle_transform(k, sin_, cos_)

        # Flatten spatial: [B, heads, N, key_dim]
        q = q.flatten(2, 3)
        k = k.flatten(2, 3)
        v_r = v.view(B, H, W, self.num_heads, -1).permute(0, 3, 1, 2, 4).flatten(2, 3)  # [B, heads, N, vdim]

        # Full NxN attention + geometry prior
        attn = q @ k.transpose(-1, -2)                                  # [B, heads, N, N]
        attn = attn + mask                                              # add geometry decay bias
        attn = F.softmax(attn, dim=-1).to(v_r.dtype)
        out  = attn @ v_r                                               # [B, heads, N, vdim]

        # Restore spatial shape
        out = out.transpose(1, 2).reshape(B, H, W, -1)                  # [B, H, W, D]
        out = out + lepe
        return self.out_proj(out)


# -----------------------------------------------------------------------------
# FFN - Feed-Forward Network with DWConv subconv
# -----------------------------------------------------------------------------

class FFN(nn.Module):
    """FFN: FC1 -> GELU -> DWConv -> FC2. Input: [B, H, W, C]."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1    = nn.Linear(embed_dim, ffn_dim)
        self.fc2    = nn.Linear(ffn_dim, embed_dim)
        self.dwconv = DWConv2d(ffn_dim, 3, 1, 1)                        # subconv for local context
        self.drop   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = x + self.dwconv(x)                                          # residual local context
        x = self.fc2(x)
        x = self.drop(x)
        return x

# ----------------------------------------------------------------------------
# GSA - Geometry-Aware Self-Attention
# ----------------------------------------------------------------------------

class GSA(nn.Module):
    """DFormerv2 Full_GSA adapted to SpatialVLM pipeline.

    Wraps N x (GeoPriorGen + cnn_pos_encode + Full_GSA + FFN)
    with the same external interface as the original gsa.py.

    Input:  visual_tokens [B, N, 1024]  (flat, post-merger)
            depth_map     [B, H, W]     (raw sensor depth)
            h_patches, w_patches        (from image_grid_thw after merger)
    Output: geometry-aware tokens [B, N, 1024]
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        num_heads:  int = 8,
        ffn_dim:    int = 2048,
        dropout:    float = 0.1,
        patch_size: int = 32,      
        num_blocks: int = 2,
        init_value: float = 2.0,
        heads_range: float = 4.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleDict({
                'geo_prior':      GeoPriorGen(hidden_dim, num_heads, init_value, heads_range),
                'cnn_pos_encode': DWConv2d(hidden_dim, 3, 1, 1),
                'norm1':          nn.LayerNorm(hidden_dim),
                'attn':           Full_GSA(hidden_dim, num_heads),
                'norm2':          nn.LayerNorm(hidden_dim),
                'ffn':            FFN(hidden_dim, ffn_dim, dropout),
            }))

    def forward(
        self,
        visual_tokens: torch.Tensor,
        depth_map: torch.Tensor,
        h_patches: int | None = None,
        w_patches: int | None = None,
    ) -> torch.Tensor:
        """
        Args:
            visual_tokens: [B, N, 1024] -- flat tokens from Qwen merger
            depth_map:     [B, H_d, W_d] -- raw depth (any resolution)
            h_patches:     patch grid height (from image_grid_thw)
            w_patches:     patch grid width
        Returns:
            [B, N, 1024] geometry-aware tokens
        """
        B, N, D = visual_tokens.shape

        if h_patches is None or w_patches is None:
            h_patches = w_patches = int(math.isqrt(N))
            assert h_patches * w_patches == N, (
                f"N={N} is not a perfect square. Provide explicit h_patches, w_patches."
            )

        # Reshape tokens to spatial [B, H, W, D] for DFormerv2 blocks
        x = visual_tokens.view(B, h_patches, w_patches, D)

        for blk in self.blocks:
            # 1. CNN positional encoding (local spatial context)
            x = x + blk['cnn_pos_encode'](x)

            # 2. Geometry prior from depth map
            geo_prior = blk['geo_prior']((h_patches, w_patches), depth_map)

            # 3. Full_GSA with residual
            x = x + blk['attn'](blk['norm1'](x), geo_prior)

            # 4. FFN with residual
            x = x + blk['ffn'](blk['norm2'](x))

        # Restore flat shape
        return x.view(B, N, D)