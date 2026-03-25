"""
MODULE: Region-Level Token Injection (RTI)

Position: After GSA, before Backbone
Input:
    visual_tokens  [B, N, 1024]   post-GSA depth-aware tokens
    depth_map      [B, H, W]      raw sensor (FullHD 1920x1080)
    rle_list       list[dict]     RLE entries from JSON "rle" field
    image_grid_thw Tensor[1, 3]   [t, h, w] from Qwen processor

Each <mask> in the question -> 2 tokens:
    mask_rgb   = DB-style soft Gated Attention Pool -> [B, 1024]
    mask_depth = [mean_d, std_d, cx_soft, cy_soft, r0..r23] -> Linear(28->1024) -> [B, 1024]

Injection result:
    Before: [..., embed(<mask>),         ...]   (L tokens)
    After:  [..., mask_rgb, mask_depth,  ...]   (L+1 per mask)

mask_rgb: DB-style Soft Gated Attention Pool:
    coverage_i = adaptive_avg_pool(binary_mask)[i]
    soft_i     = sigmoid(K x (coverage_i - theta))
    score_i    = Linear(token_i)
    weight_i   = softmax(score_i + log(soft_i))   <- log-domain masking
    mask_rgb   = sum weight_i x token_i

mask_depth: 28-dim vector: [mean_d, std_d, cx_soft, cy_soft, r0..r23]
    cx_soft = sum_i(col_i x soft_i) / sum_i(soft_i)   -- soft-weighted centroid
    r0..r23 = 24-ray radial depth profile via F.grid_sample (bilinear, differentiable)
    All 28 dims have non-zero gradient -> Linear(28->1024) + LayerNorm

RLE format (COCO):
    {"size": [H, W], "counts": "<LEB128 ASCII string>"}
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
import pycocotools.mask as mask_utils


# ----------------------------------------------------------------------------
# Helper
# ----------------------------------------------------------------------------

def _soft_mask_from_coverage(
    coverage: torch.Tensor,                         # [h_vis, w_vis] in [0, 1]
    k: float = 50.0,                                # amplification factor (steepness)
    theta: float = 0.3,                             # threshold            (lower than 0.5 -> include boundary patches)
) -> torch.Tensor:
    """DB-style differentiable soft mask (DBNet++ style).

    Approximates hard threshold (>= theta) with steep sigmoid:
        soft = sigmoid(k x (coverage - theta))

    Benefits over hard binarization:
        - Boundary patches (coverage < theta) still get small weight
        - log(soft) in attention avoids empty-mask special case
        - Gradient path exists through coverage values
    """
    return torch.sigmoid(k * (coverage - theta))    # [h_vis, w_vis] in (0, 1)


def _radial_depth_profile(
    depth_map:  torch.Tensor,                       # [B, H, W]
    soft2d:     torch.Tensor,                       # [h_vis, w_vis] on dev  (soft mask weights)
    cx_soft:    torch.Tensor,                       # scalar in [0,1] normalized to w_vis
    cy_soft:    torch.Tensor,                       # scalar in [0,1] normalized to h_vis
    n_rays:     int = 24,                           # radial profile rays (kx15 deg, from soft centroid)
    n_samples:  int = 20,                           # bilinear samples per ray (F.grid_sample)
) -> torch.Tensor:                                  # [B, n_rays]
    """Differentiable 24-ray radial depth profile.

    For each ray k at angle theta_k = kx15 deg (k=0..23):
        Sample n_samples points along ray from centroid via F.grid_sample (bilinear).
        Weight by soft_mask value at each sample point.
        r_k = sum_t (soft_weight_t x depth_t) / sum soft_weight_t

    Advantages over histogram for this dataset:
        - Distinguishes overlapping masks (same hist, different spatial profile)
        - Captures asymmetric depth gradient from oblique camera angle
        - cx/cy encode position; r0..r23 encode directional depth variation

    All 24 outputs are differentiable via:
        depth_map values (F.grid_sample gradient) + soft_mask (sigmoid gradient)
    """
    B, H, W = depth_map.shape
    h_vis, w_vis = soft2d.shape
    dev  = depth_map.device
    dt   = depth_map.dtype

    # Centroid in pixel coordinates (depth_map resolution)
    cx_pix = cx_soft * (W - 1)                                                  # scalar tensor (preserves grad)
    cy_pix = cy_soft * (H - 1)

    # Max ray length: half shortest dim (stay within image)
    max_r = min(H, W) * 0.45

    # 24 ray directions
    angles = torch.linspace(0, 2 * math.pi, n_rays + 1, device=dev)[:-1]       # [R]
    cos_a  = torch.cos(angles)                                                 # [R]
    sin_a  = torch.sin(angles)                                                 # [R]

    # Sample distances: linspace [0, max_r] along each ray
    t_vals = torch.linspace(0.0, max_r, n_samples, device=dev)                 # [S]

    # Sample coordinates in pixel space: [R, S]
    xs_pix = cx_pix + cos_a.unsqueeze(1) * t_vals.unsqueeze(0)                 # [R, S]
    ys_pix = cy_pix + sin_a.unsqueeze(1) * t_vals.unsqueeze(0)                 # [R, S]

    # Sample depth_map at all [R*S] points via F.grid_sample 
    # grid_sample expects grid in [-1,1]:  x=(col/W)*2-1, y=(row/H)*2-1
    xs_g = (xs_pix / (W - 1)) * 2.0 - 1.0                                      # [R, S]
    ys_g = (ys_pix / (H - 1)) * 2.0 - 1.0

    # Build grid [1, R*S, 1, 2], broadcast over batch with expand
    grid = torch.stack([xs_g, ys_g], dim=-1).reshape(1, n_rays * n_samples, 1, 2)
    grid = grid.expand(B, -1, -1, -1)                                          # [B, R*S, 1, 2]

    depth_in  = depth_map.unsqueeze(1).float()                                  # [B, 1, H, W]
    d_samples = F.grid_sample(
        depth_in, grid, mode='bilinear', padding_mode='border', align_corners=True
    )                                                                           # [B, 1, R*S, 1]
    d_samples = d_samples.squeeze(1).squeeze(-1).reshape(B, n_rays, n_samples)  # [B,R,S]

    # Sample soft_mask at same points (patch resolution)
    xs_p   = (xs_pix / (W - 1)) * (w_vis - 1)                                   # [R, S] in patch columns
    ys_p   = (ys_pix / (H - 1)) * (h_vis - 1)                                   # [R, S] in patch rows
    xs_pg  = (xs_p / (w_vis - 1)) * 2.0 - 1.0                                   # normalize to [-1,1]
    ys_pg  = (ys_p / (h_vis - 1)) * 2.0 - 1.0
    grid_p = torch.stack([xs_pg, ys_pg], dim=-1).reshape(1, n_rays * n_samples, 1, 2)

    soft_in = soft2d.float().unsqueeze(0).unsqueeze(0)                          # [1,1,h,w]
    sw = F.grid_sample(
        soft_in, grid_p, mode='bilinear', padding_mode='zeros', align_corners=True
    )                                                                           # [1,1,R*S,1]
    sw = sw.squeeze(0).squeeze(0).squeeze(-1).reshape(n_rays, n_samples)        # [R, S]
    sw = sw.clamp(min=1e-6)                                                     # [R, S]

    # Soft-weighted depth per ray
    w_norm  = sw / sw.sum(dim=1, keepdim=True)                                  # [R, S]
    # [B, R, S] * [1, R, S] -> [B, R]
    profile = (d_samples * w_norm.unsqueeze(0).to(dt)).sum(dim=-1)              # [B, R]
    return profile


# ----------------------------------------------------------------------------
# RTE - Region Token Extractor
# ----------------------------------------------------------------------------

class RTE(nn.Module):
    """Extract (mask_rgb, mask_depth) token pairs from RLE annotations.

    Learnable:
        rgb_gate    Linear(1024, 1, bias=False)            
        depth_proj  Linear(28, 1024, bias=True) + LayerNorm

    mask_rgb: DB-style Soft Gated Attention Pool:
        coverage_i = adaptive_avg_pool(binary_mask)[i]
        soft_i     = sigmoid(K x (coverage_i - theta))
        score_i    = Linear(token_i)
        weight_i   = softmax(score_i + log(soft_i))   
        mask_rgb   = sum weight_i x token_i

    mask_depth: 28-dim vector: [mean_d, std_d, cx_soft, cy_soft, r0..r23]
        cx_soft, cy_soft: soft-weighted centroid (differentiable from DB soft mask)
        r0..r23: 24-ray radial depth profile via F.grid_sample (differentiable)
                 Distinguishes overlapping masks & complex shapes from oblique camera.
        All 28 dims have non-zero gradient -> projected via Linear(28->1024) + LayerNorm
    """

    def __init__(
        self, 
        hidden_dim:      int = 1024, 
        depth_stats_dim: int = 28,       # [mean_d, std_d, cx_soft, cy_soft, r0..r23] -> 4 + 24 = 28
    ):
        super().__init__()
        self.rgb_gate = nn.Linear(hidden_dim, 1, bias=False)
        self.depth_proj = nn.Sequential(
            nn.Linear(depth_stats_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    # Private helpers

    def _rle_to_soft_mask(
        self,
        rle:   dict,
        h_vis: int,
        w_vis: int,
        device: torch.device = None,
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """RLE -> binary mask + DB-style soft coverage mask.

        Returns:
            binary: np.ndarray [H, W] bool   - for pixel-level depth stats
            soft2d: torch.Tensor [h_vis, w_vis] float in (0,1)  - for attention weights
        """
        binary   = mask_utils.decode(rle).astype(bool)          # [H, W] bool
        t        = torch.from_numpy(binary.astype(np.float32))  # [H, W] float
        if device is not None:
            t = t.to(device)
        coverage = F.adaptive_avg_pool2d(
            t.unsqueeze(0).unsqueeze(0), (h_vis, w_vis)         # [1,1,h_vis,w_vis]
        ).squeeze()                                             # [h_vis, w_vis]
        soft2d   = _soft_mask_from_coverage(coverage)           # [h_vis, w_vis]
        return binary, soft2d                     
    
    def _rgb_token(
        self, 
        visual_tokens: torch.Tensor,
        soft2d:        torch.Tensor,   
    ) -> torch.Tensor:
        """DB-style Soft Gated Attention Pool over all N visual tokens.

        score_i    = Linear(token_i)           [B, N]
        log_soft_i = log(soft_i)               [N]   broadcast
        weights    = softmax(score + log_soft)  [B, N]
        mask_rgb   = sum weights_i x token_i     [B, 1024]
        """
        dev   = visual_tokens.device
        dtype = visual_tokens.dtype

        # Flatten soft2d -> [N] and move to device/dtype
        soft_flat = soft2d.reshape(-1).to(device=dev, dtype=dtype)         # [N]
        log_soft  = torch.log(soft_flat.clamp(min=1e-8))                   # [N]

        scores  = self.rgb_gate(visual_tokens).squeeze(-1)                 # [B, N]
        weights = torch.softmax(scores + log_soft.unsqueeze(0), dim=-1)    # [B, N]
        mask_rgb = (weights.unsqueeze(-1) * visual_tokens).sum(dim=1)      # [B, 1024]
        return mask_rgb

    def _depth_token(
        self,
        depth_map:    torch.Tensor,   # [B, H, W]
        binary_mask:  np.ndarray,     # [H, W] (pixel-level mask)
        soft2d:       torch.Tensor,   # [h_vis, w_vis] (for cx/cy & radial profile)
        h_vis:        int,
        w_vis:        int,
    ) -> torch.Tensor:                # [B, 1024]
        """Compute 28-dim depth stats -> Linear(28->1024) -> [B, 1024].

        Stats: [mean_d, std_d, cx_soft, cy_soft, r0..r23]
            - mean_d, std_d:   pixel-level masked depth statistics.
            - cx_soft, cy_soft: soft-weighted centroid (differentiable, normalized [0,1]).
            - r0..r23:          24-ray radial depth profile via F.grid_sample.
        """
        B, H, W = depth_map.shape
        dev     = depth_map.device
        dtype   = next(self.parameters()).dtype

        bool_mask = torch.from_numpy(binary_mask).to(device=dev)             # [H, W]

        # Handle empty mask
        if bool_mask.sum() == 0:
            stats = torch.zeros(B, 28, device=dev, dtype=dtype)
            return self.depth_proj(stats)

        # Soft-weighted centroid (differentiable)
        soft2d_dev = soft2d.to(dev)
        soft_sum   = soft2d_dev.sum() + 1e-8
        grid_x     = torch.arange(w_vis, device=dev, dtype=soft2d_dev.dtype) # [w_vis]
        grid_y     = torch.arange(h_vis, device=dev, dtype=soft2d_dev.dtype) # [h_vis]
        cx_soft    = (soft2d_dev.sum(0) * grid_x).sum() / (soft_sum * w_vis) # scalar [0,1]
        cy_soft    = (soft2d_dev.sum(1) * grid_y).sum() / (soft_sum * h_vis) # scalar [0,1]

        # Basic depth stats (pixel-level masked values)
        vals       = depth_map[:, bool_mask].float()                         # [B, M]  (M = number of masked pixels)
        mean_d     = vals.mean(dim=1)                                        # [B]
        std_d      = vals.std(dim=1).nan_to_num(0.0).clamp(min=0.0)          # [B]

        # 24-ray radial depth profile
        profile = _radial_depth_profile(
            depth_map, soft2d_dev, cx_soft, cy_soft
        )                                                                    # [B, 24]

        # Concatenate all 28 stats
        cx_b = cx_soft.expand(B).unsqueeze(1).to(mean_d.dtype)               # [B, 1]
        cy_b = cy_soft.expand(B).unsqueeze(1).to(mean_d.dtype)               # [B, 1]
        stats = torch.cat([
            mean_d.unsqueeze(1),                                             # [B, 1]
            std_d.unsqueeze(1),                                              # [B, 1]
            cx_b,                                                            # [B, 1]
            cy_b,                                                            # [B, 1]
            profile.to(mean_d.dtype),                                        # [B, 24]
        ], dim=1)                                                            # [B, 28]

        return self.depth_proj(stats.to(dtype))                              # [B, 1024]

    def forward(
        self,
        visual_tokens:  torch.Tensor,                                        # [B, N, 1024]
        depth_map:      torch.Tensor,                                        # [B, H, W]
        rle_list:       List[dict],                              
        image_grid_thw: torch.Tensor,                                        # [num_images, 3] 
    )-> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Returns [(mask_rgb, mask_depth), ...] one tuple per RLE mask.

        mask_rgb   -> DB-style soft gated pool over all N visual tokens  [B, 1024]
        mask_depth -> [mean_d, std_d, cx_soft, cy_soft, r0..r23] projected [B, 1024]
        """
        _, h_patches, w_patches = [int(x) for x in image_grid_thw[0].tolist()]
        h_vis, w_vis = h_patches // 2, w_patches // 2

        result = []
        dev = visual_tokens.device
        for rle in rle_list:
            binary, soft2d = self._rle_to_soft_mask(rle, h_vis, w_vis, device=dev)
            rgb = self._rgb_token(visual_tokens, soft2d)                     # [B, 1024]
            dep = self._depth_token(depth_map, binary, soft2d, h_vis, w_vis) # [B, 1024]
            result.append((rgb, dep))
        return result

    def inject_into_text_embeds(
        self,
        text_embeds:    torch.Tensor,                                        # [B, L, 1024]
        mask_positions: List[int],                                           # sorted <mask> start indices
        region_tokens:  List[Tuple[torch.Tensor, torch.Tensor]],
        mask_token_len: int = 1,                                             # number of tokens per <mask> (Qwen: 3)
    ) -> torch.Tensor:                                                       # [B, L', 1024]
        """Replace each <mask> token sequence with [mask_rgb, mask_depth]."""
        B, L, D = text_embeds.shape
        segments, prev = [], 0
        for pos, (rgb, dep) in zip(mask_positions, region_tokens):
            if pos > prev:
                segments.append(text_embeds[:, prev:pos, :])
            segments.append(rgb.unsqueeze(1))                                # [B, 1, D]
            segments.append(dep.unsqueeze(1))                                # [B, 1, D]
            prev = pos + mask_token_len                                      # skip all tokens of <mask>
        if prev < L:
            segments.append(text_embeds[:, prev:, :])
        return torch.cat(segments, dim=1)
    