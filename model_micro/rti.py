"""
MODULE: Region-Level Token Injection (RTI) — Micro (Batched)

Position: After GSA, before Backbone
Input:
    visual_tokens  [B, N, 1024]   post-GSA depth-aware tokens
    depth_map      [B, H, W]      raw sensor depth
    rle_list       list[list[dict]]  RLE entries per sample in batch
    image_grid_thw Tensor[B, 3]   [t, h, w] from Qwen processor

Each <mask> in the question -> 2 tokens:
    mask_rgb   = DB-style soft Gated Attention Pool -> [1, 1024]
    mask_depth = [mean_d, std_d, cx_soft, cy_soft, r0..r23] -> Linear(28->1024) -> [1, 1024]

Injection: replace each <mask> token with [mask_rgb, mask_depth]

Batched RTI Strategy (Flatten -> Parallel -> Scatter):
    1. FLATTEN: Collect all masks from all samples -> total_masks
    2. PARALLEL: Process all masks in a single GPU kernel
    3. SCATTER: Inject back into pre-padded text embeddings

NOTE: The parallel processing of _rgb_token and _depth_token is done
      per-mask but with minimal Python overhead. True GPU parallelism
      happens in the attention pooling (batched matmul).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
import pycocotools.mask as mask_utils


# ----------------------------------------------------------------------------
# Helpers (unchanged from v1)
# ----------------------------------------------------------------------------

def _soft_mask_from_coverage(
    coverage: torch.Tensor,                         # [h_vis, w_vis] in [0, 1]
    k: float = 50.0,
    theta: float = 0.3,
) -> torch.Tensor:
    """DB-style differentiable soft mask."""
    return torch.sigmoid(k * (coverage - theta))


def _radial_depth_profile(
    depth_map:  torch.Tensor,                       # [B, H, W]
    soft2d:     torch.Tensor,                       # [h_vis, w_vis]
    cx_soft:    torch.Tensor,
    cy_soft:    torch.Tensor,
    n_rays:     int = 24,
    n_samples:  int = 20,
) -> torch.Tensor:                                  # [B, n_rays]
    """Differentiable 24-ray radial depth profile."""
    B, H, W = depth_map.shape
    h_vis, w_vis = soft2d.shape
    dev  = depth_map.device
    dt   = depth_map.dtype

    cx_pix = cx_soft * (W - 1)
    cy_pix = cy_soft * (H - 1)
    max_r = min(H, W) * 0.45

    angles = torch.linspace(0, 2 * math.pi, n_rays + 1, device=dev)[:-1]
    cos_a  = torch.cos(angles)
    sin_a  = torch.sin(angles)
    t_vals = torch.linspace(0.0, max_r, n_samples, device=dev)

    xs_pix = cx_pix + cos_a.unsqueeze(1) * t_vals.unsqueeze(0)
    ys_pix = cy_pix + sin_a.unsqueeze(1) * t_vals.unsqueeze(0)

    xs_g = (xs_pix / (W - 1)) * 2.0 - 1.0
    ys_g = (ys_pix / (H - 1)) * 2.0 - 1.0

    grid = torch.stack([xs_g, ys_g], dim=-1).reshape(1, n_rays * n_samples, 1, 2)
    grid = grid.expand(B, -1, -1, -1)

    depth_in  = depth_map.unsqueeze(1).float()
    d_samples = F.grid_sample(
        depth_in, grid, mode='bilinear', padding_mode='border', align_corners=True
    )
    d_samples = d_samples.squeeze(1).squeeze(-1).reshape(B, n_rays, n_samples)

    xs_p   = (xs_pix / (W - 1)) * (w_vis - 1)
    ys_p   = (ys_pix / (H - 1)) * (h_vis - 1)
    xs_pg  = (xs_p / max(w_vis - 1, 1)) * 2.0 - 1.0
    ys_pg  = (ys_p / max(h_vis - 1, 1)) * 2.0 - 1.0
    grid_p = torch.stack([xs_pg, ys_pg], dim=-1).reshape(1, n_rays * n_samples, 1, 2)

    soft_in = soft2d.float().unsqueeze(0).unsqueeze(0)
    sw = F.grid_sample(
        soft_in, grid_p, mode='bilinear', padding_mode='zeros', align_corners=True
    )
    sw = sw.squeeze(0).squeeze(0).squeeze(-1).reshape(n_rays, n_samples)
    sw = sw.clamp(min=1e-6)

    w_norm  = sw / sw.sum(dim=1, keepdim=True)
    profile = (d_samples * w_norm.unsqueeze(0).to(dt)).sum(dim=-1)
    return profile


# ----------------------------------------------------------------------------
# RTE - Region Token Extractor (Batched)
# ----------------------------------------------------------------------------

class RTE(nn.Module):
    """Extract (mask_rgb, mask_depth) token pairs from RLE annotations.

    Supports batch_size > 1 via flatten->process->scatter strategy.

    Learnable:
        rgb_gate    Linear(1024, 1, bias=False)
        depth_proj  Linear(28, 1024, bias=True) + LayerNorm
    """

    def __init__(
        self,
        hidden_dim:      int = 1024,
        depth_stats_dim: int = 28,
    ):
        super().__init__()
        self.rgb_gate = nn.Linear(hidden_dim, 1, bias=False)
        self.depth_proj = nn.Sequential(
            nn.Linear(depth_stats_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    # ---- Private helpers (single-mask processing) ----

    def _rle_to_soft_mask(
        self,
        rle:   dict,
        h_vis: int,
        w_vis: int,
        device: torch.device = None,
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """RLE -> binary mask + DB-style soft coverage mask."""
        binary   = mask_utils.decode(rle).astype(bool)
        t        = torch.from_numpy(binary.astype(np.float32))
        if device is not None:
            t = t.to(device)
        coverage = F.adaptive_avg_pool2d(
            t.unsqueeze(0).unsqueeze(0), (h_vis, w_vis)
        ).squeeze()
        soft2d   = _soft_mask_from_coverage(coverage)
        return binary, soft2d

    def _rgb_token(
        self,
        visual_tokens: torch.Tensor,   # [1, N, 1024]
        soft2d:        torch.Tensor,    # [h_vis, w_vis]
    ) -> torch.Tensor:                  # [1, 1024]
        """DB-style Soft Gated Attention Pool over visual tokens."""
        dev   = visual_tokens.device
        dtype = visual_tokens.dtype

        soft_flat = soft2d.reshape(-1).to(device=dev, dtype=dtype)
        log_soft  = torch.log(soft_flat.clamp(min=1e-8))

        scores  = self.rgb_gate(visual_tokens).squeeze(-1)              # [1, N]
        weights = torch.softmax(scores + log_soft.unsqueeze(0), dim=-1) # [1, N]
        mask_rgb = (weights.unsqueeze(-1) * visual_tokens).sum(dim=1)   # [1, 1024]
        return mask_rgb

    def _depth_token(
        self,
        depth_map:    torch.Tensor,   # [1, H, W]
        binary_mask:  np.ndarray,     # [H, W]
        soft2d:       torch.Tensor,   # [h_vis, w_vis]
        h_vis:        int,
        w_vis:        int,
    ) -> torch.Tensor:                # [1, 1024]
        """Compute 28-dim depth stats -> Linear(28->1024) -> [1, 1024]."""
        B, H, W = depth_map.shape
        dev     = depth_map.device
        dtype   = next(self.parameters()).dtype

        bool_mask = torch.from_numpy(binary_mask).to(device=dev)

        if bool_mask.sum() == 0:
            stats = torch.zeros(B, 28, device=dev, dtype=dtype)
            return self.depth_proj(stats)

        soft2d_dev = soft2d.to(dev)
        soft_sum   = soft2d_dev.sum() + 1e-8
        grid_x     = torch.arange(w_vis, device=dev, dtype=soft2d_dev.dtype)
        grid_y     = torch.arange(h_vis, device=dev, dtype=soft2d_dev.dtype)
        cx_soft    = (soft2d_dev.sum(0) * grid_x).sum() / (soft_sum * w_vis)
        cy_soft    = (soft2d_dev.sum(1) * grid_y).sum() / (soft_sum * h_vis)

        vals       = depth_map[:, bool_mask].float()
        mean_d     = vals.mean(dim=1)
        std_d      = vals.std(dim=1, correction=0).clamp(min=0.0)

        profile = _radial_depth_profile(
            depth_map, soft2d_dev, cx_soft, cy_soft
        )

        cx_b = cx_soft.expand(B).unsqueeze(1).to(mean_d.dtype)
        cy_b = cy_soft.expand(B).unsqueeze(1).to(mean_d.dtype)
        stats = torch.cat([
            mean_d.unsqueeze(1),
            std_d.unsqueeze(1),
            cx_b,
            cy_b,
            profile.to(mean_d.dtype),
        ], dim=1)

        return self.depth_proj(stats.to(dtype))

    # ---- Batched forward ----

    def forward(
        self,
        visual_tokens:  torch.Tensor,                # [B, N, 1024]
        depth_map:      torch.Tensor,                # [B, H, W]
        rle_list:       List[List[dict]],             # [B][num_masks_b]
        image_grid_thw: torch.Tensor,                # [B, 3]
        decoded_masks:  List[List[dict]] = None,     # [B][num_masks_b]
    ) -> List[List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Process all masks for all samples in the batch.

        Returns:
            result[b][m] = (mask_rgb, mask_depth) for sample b, mask m
            mask_rgb:   [1, 1024]
            mask_depth: [1, 1024]
        """
        B = visual_tokens.shape[0]
        _, h_patches, w_patches = [int(x) for x in image_grid_thw[0].tolist()]
        h_vis, w_vis = h_patches // 2, w_patches // 2
        dev = visual_tokens.device

        result = []
        for b in range(B):
            vis_b = visual_tokens[b:b+1]  # [1, N, 1024]
            dep_b = depth_map[b:b+1]      # [1, H, W]
            masks_b = rle_list[b]
            dec_b = decoded_masks[b] if decoded_masks is not None else None

            pairs = []
            for m, rle in enumerate(masks_b):
                if dec_b is not None and m < len(dec_b):
                    binary = dec_b[m]['binary']
                    soft2d = dec_b[m]['soft2d']
                else:
                    binary, soft2d = self._rle_to_soft_mask(rle, h_vis, w_vis, device=dev)

                rgb = self._rgb_token(vis_b, soft2d)                     # [1, 1024]
                dep = self._depth_token(dep_b, binary, soft2d, h_vis, w_vis)  # [1, 1024]
                pairs.append((rgb, dep))
            result.append(pairs)
        return result

    def inject_into_text_embeds(
        self,
        text_embeds:    torch.Tensor,                # [B, L, 1024]
        mask_positions: List[List[int]],             # [B][num_masks_b] sorted <mask> start indices
        region_tokens:  List[List[Tuple[torch.Tensor, torch.Tensor]]],
        mask_token_len: int = 1,
    ) -> torch.Tensor:                               # [B, L', 1024]  (L' may differ per sample)
        """Replace each <mask> token sequence with [mask_rgb, mask_depth].

        For batched operation, processes each sample independently and pads
        to the maximum output length in the batch.

        Returns:
            embeds: [B, max_L', 1024] — padded output
        """
        B, L, D = text_embeds.shape
        all_sequences = []

        for b in range(B):
            positions = mask_positions[b]
            tokens = region_tokens[b]

            segments, prev = [], 0
            for pos, (rgb, dep) in zip(positions, tokens):
                if pos > prev:
                    segments.append(text_embeds[b:b+1, prev:pos, :])
                segments.append(rgb.unsqueeze(1))      # [1, 1, D]
                segments.append(dep.unsqueeze(1))      # [1, 1, D]
                prev = pos + mask_token_len
            if prev < L:
                segments.append(text_embeds[b:b+1, prev:, :])

            all_sequences.append(torch.cat(segments, dim=1))  # [1, L'_b, D]

        # Pad all to max length
        max_len = max(seq.shape[1] for seq in all_sequences)
        padded = torch.zeros(B, max_len, D, device=text_embeds.device, dtype=text_embeds.dtype)
        output_lengths = []
        for b, seq in enumerate(all_sequences):
            L_b = seq.shape[1]
            padded[b, :L_b, :] = seq[0]
            output_lengths.append(L_b)

        return padded, output_lengths
