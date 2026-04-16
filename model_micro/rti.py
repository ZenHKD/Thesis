"""
MODULE: Region-Level Token Injection (RTI)

Position: Before Concat Fusion (independent of Vision Encoder)
Input:
    rgb_images     [B, 3, H, W]      raw RGB (0-1 range)
    depth_maps     [B, H, W]         raw sensor depth
    rle_list       list[list[dict]]   RLE entries per sample in batch
    image_grid_thw Tensor[B, 3]      [t, h, w] for soft mask grid sizing

Each <mask> in the question -> 3 learned tokens:
    mask_rgb   = RGB appearance stats    -> Linear(12->1024) + LN -> [1, 1024]
    mask_depth = depth stats + radial    -> Linear(28->1024) + LN -> [1, 1024]
    mask_geo   = spatial context         -> Linear(16->1024) + LN -> [1, 1024]

RTI is completely independent of the Vision Encoder.
Two parallel streams feed the decoder:
    Stream 1: RGB -> Vision Encoder -> visual_tokens (scene-level semantics)
    Stream 2: RGB + Depth + RLE -> RTI -> per-region tokens (region-level descriptors)

Batched Strategy (Flatten -> Parallel Project -> Scatter):
    1. FLATTEN: Extract feature vectors for each mask (GPU tensor ops)
    2. PARALLEL: Stack all features, project in single batched matmul
    3. SCATTER: Inject into text embeddings at <mask> positions

Injection: 3->3 in-place replacement (sequence length UNCHANGED)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
import pycocotools.mask as mask_utils


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _soft_mask_from_coverage(
    coverage: torch.Tensor,                         # [h, w] in [0, 1]
    k: float = 50.0,
    theta: float = 0.3,
) -> torch.Tensor:
    """DB-style differentiable soft mask."""
    return torch.sigmoid(k * (coverage - theta))


def _radial_depth_profile(
    depth_map:  torch.Tensor,                       # [1, H, W]
    soft2d:     torch.Tensor,                       # [h_soft, w_soft]
    cx_soft:    torch.Tensor,
    cy_soft:    torch.Tensor,
    n_rays:     int = 24,
    n_samples:  int = 20,
) -> torch.Tensor:                                  # [1, n_rays]
    """Differentiable 24-ray radial depth profile."""
    B, H, W = depth_map.shape
    h_soft, w_soft = soft2d.shape
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

    xs_p   = (xs_pix / (W - 1)) * (w_soft - 1)
    ys_p   = (ys_pix / (H - 1)) * (h_soft - 1)
    xs_pg  = (xs_p / max(w_soft - 1, 1)) * 2.0 - 1.0
    ys_pg  = (ys_p / max(h_soft - 1, 1)) * 2.0 - 1.0
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
# RTE - Region Token Extractor (3 learned tokens per <mask>)
# ----------------------------------------------------------------------------

class RTE(nn.Module):
    """Extract 3 learned region tokens per <mask> from raw inputs.

    Independent of Vision Encoder. Processes:
        - Raw RGB image + mask -> appearance features -> mask_rgb
        - Raw depth map + mask -> depth profile features -> mask_depth
        - Raw depth map + mask -> spatial context -> mask_geo

    Learnable:
        rgb_proj    Linear(12, 1024) + LayerNorm
        depth_proj  Linear(28, 1024) + LayerNorm
        geo_proj    Linear(16, 1024) + LayerNorm
    """

    RGB_FEAT_DIM = 12
    DEPTH_FEAT_DIM = 28
    GEO_FEAT_DIM = 16

    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        self.rgb_proj = nn.Sequential(
            nn.Linear(self.RGB_FEAT_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.depth_proj = nn.Sequential(
            nn.Linear(self.DEPTH_FEAT_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.geo_proj = nn.Sequential(
            nn.Linear(self.GEO_FEAT_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    # ---- Private: RLE decoding ----

    def _rle_to_soft_mask(
        self,
        rle:    dict,
        h_soft: int,
        w_soft: int,
        device: torch.device = None,
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """RLE -> binary mask + DB-style soft coverage mask."""
        binary   = mask_utils.decode(rle).astype(bool)
        t        = torch.from_numpy(binary.astype(np.float32))
        if device is not None:
            t = t.to(device)
        coverage = F.adaptive_avg_pool2d(
            t.unsqueeze(0).unsqueeze(0), (h_soft, w_soft)
        ).squeeze()

        # Relative Coverage Normalization
        coverage = coverage / (coverage.max() + 1e-8)

        soft2d   = _soft_mask_from_coverage(coverage)
        return binary, soft2d

    # ---- Private: Feature extraction ----

    def _rgb_features(
        self,
        rgb_image:   torch.Tensor,      # [3, H, W]
        gray_image:  torch.Tensor,      # [H, W]
        mask_t:      torch.Tensor,      # [H, W] bool
    ) -> torch.Tensor:                   # [12]
        """Extract RGB appearance features from masked region.

        Features (12-dim):
            mean R, G, B        (3)  — average color
            std R, G, B         (3)  — color variation
            mean luminance      (1)  — brightness
            std luminance       (1)  — texture energy
            min luminance       (1)
            max luminance       (1)
            color contrast      (1)  — std across channel means
            saturation          (1)  — (max_ch - min_ch) / (max_ch + eps)
        """
        dev   = rgb_image.device
        dtype = next(self.parameters()).dtype

        if mask_t.sum() == 0:
            return torch.zeros(self.RGB_FEAT_DIM, device=dev, dtype=dtype)

        # Vectorized channel stats [6]
        vals = rgb_image[:, mask_t]  # [3, K]
        ch_means = vals.mean(dim=1)  # [3]
        ch_stds  = vals.std(dim=1, correction=0).clamp(min=0)  # [3]

        # Vectorized luminance stats [4]
        gray_vals = gray_image[mask_t]  # [K]
        lum_mean = gray_vals.mean()
        lum_std  = gray_vals.std(correction=0).clamp(min=0)
        lum_min  = gray_vals.min()
        lum_max  = gray_vals.max()

        # Derived [2]
        color_contrast = ch_means.std(correction=0)
        saturation = (ch_means.max() - ch_means.min()) / (ch_means.max() + 1e-8)

        features = torch.cat([
            ch_means, ch_stds,
            torch.stack([lum_mean, lum_std, lum_min, lum_max, color_contrast, saturation])
        ])
        return features.to(dtype)  # [12]

    def _depth_features(
        self,
        depth_map:    torch.Tensor,      # [1, H, W]
        mask_t:       torch.Tensor,       # [H, W] bool
        soft2d_dev:   torch.Tensor,       # [h_soft, w_soft] ON DEVICE
        h_soft:       int,
        w_soft:       int,
    ) -> torch.Tensor:                   # [28]
        """Compute 28-dim depth statistics for masked region.

        Features (28-dim):
            mean_d              (1)  — average depth
            std_d               (1)  — depth variation
            centroid cx, cy     (2)  — soft-mask weighted
            24 radial depth rays (24) — from centroid outward
        """
        B, H, W = depth_map.shape
        dev     = depth_map.device
        dtype   = next(self.parameters()).dtype

        if mask_t.sum() == 0:
            return torch.zeros(self.DEPTH_FEAT_DIM, device=dev, dtype=dtype)

        soft_sum   = soft2d_dev.sum() + 1e-8
        grid_x     = torch.arange(w_soft, device=dev, dtype=soft2d_dev.dtype)
        grid_y     = torch.arange(h_soft, device=dev, dtype=soft2d_dev.dtype)
        cx_soft    = (soft2d_dev.sum(0) * grid_x).sum() / (soft_sum * w_soft)
        cy_soft    = (soft2d_dev.sum(1) * grid_y).sum() / (soft_sum * h_soft)

        vals       = depth_map[:, mask_t].float()
        mean_d     = vals.mean(dim=1)                    # [1]
        std_d      = vals.std(dim=1, correction=0).clamp(min=0.0)  # [1]

        profile = _radial_depth_profile(
            depth_map, soft2d_dev, cx_soft, cy_soft
        )                                                 # [1, 24]

        cx_b = cx_soft.unsqueeze(0).to(mean_d.dtype)     # [1]
        cy_b = cy_soft.unsqueeze(0).to(mean_d.dtype)     # [1]
        stats = torch.cat([
            mean_d,                                       # [1]
            std_d,                                        # [1]
            cx_b,                                         # [1]
            cy_b,                                         # [1]
            profile.squeeze(0).to(mean_d.dtype),          # [24]
        ], dim=0)                                         # [28]

        return stats.to(dtype)

    def _geo_features(
        self,
        depth_f:      torch.Tensor,      # [H, W] (single sample, float32)
        mask_t:       torch.Tensor,       # [H, W] bool
        global_mean:  torch.Tensor,
        global_std:   torch.Tensor,
        global_max:   torch.Tensor,
    ) -> torch.Tensor:                   # [16]
        """Compute spatial context features for masked region.

        Features (16-dim):
            global_mean_d       (1)  — scene average depth
            global_std_d        (1)  — scene depth variation
            relative_depth      (1)  — region_mean / global_mean
            depth_percentile    (1)  — fraction of pixels shallower than region
            area_ratio          (1)  — mask_pixels / total_pixels
            centroid_x          (1)  — normalized 0-1
            centroid_y          (1)  — normalized 0-1
            bbox_x0             (1)  — normalized bbox left
            bbox_y0             (1)  — normalized bbox top
            bbox_w              (1)  — normalized bbox width
            bbox_h              (1)  — normalized bbox height
            min_depth_norm      (1)  — region min / (global max + eps)
            max_depth_norm      (1)  — region max / (global max + eps)
            aspect_ratio        (1)  — bbox_w / (bbox_h + eps)
            dist_to_center      (1)  — centroid distance to image center
            depth_range_norm    (1)  — (region_max - region_min) / (global_max + eps)
        """
        dev   = depth_f.device
        dtype = next(self.parameters()).dtype
        H, W  = depth_f.shape

        if mask_t.sum() == 0:
            return torch.zeros(self.GEO_FEAT_DIM, device=dev, dtype=dtype)

        # Region depth stats
        region_vals = depth_f[mask_t]
        region_mean = region_vals.mean()
        region_min  = region_vals.min()
        region_max  = region_vals.max()

        relative_depth   = region_mean / (global_mean + 1e-8)
        depth_percentile = (depth_f < region_mean).float().mean()
        area_ratio       = mask_t.float().sum() / (H * W)

        # Centroid (from binary mask)
        ys, xs = torch.where(mask_t)
        centroid_x = xs.float().mean() / max(W - 1, 1)
        centroid_y = ys.float().mean() / max(H - 1, 1)

        # Bounding box (normalized)
        bbox_x0 = xs.min().float() / max(W - 1, 1)
        bbox_y0 = ys.min().float() / max(H - 1, 1)
        bbox_w  = (xs.max() - xs.min()).float() / max(W - 1, 1)
        bbox_h  = (ys.max() - ys.min()).float() / max(H - 1, 1)

        min_depth_norm   = region_min / global_max
        max_depth_norm   = region_max / global_max
        aspect_ratio     = bbox_w / (bbox_h + 1e-8)
        dist_to_center   = ((centroid_x - 0.5) ** 2 + (centroid_y - 0.5) ** 2).sqrt()
        depth_range_norm = (region_max - region_min) / global_max

        features = torch.stack([
            global_mean, global_std, relative_depth, depth_percentile,
            area_ratio, centroid_x, centroid_y,
            bbox_x0, bbox_y0, bbox_w, bbox_h,
            min_depth_norm, max_depth_norm, aspect_ratio,
            dist_to_center, depth_range_norm,
        ])
        return features.to(dtype)  # [16]

    # ---- Batched forward ----

    def forward(
        self,
        rgb_images:     torch.Tensor,                  # [B, 3, H, W]
        depth_maps:     torch.Tensor,                  # [B, H, W]
        rle_list:       List[List[dict]],               # [B][num_masks_b]
        image_grid_thw: torch.Tensor,                  # [B, 3] — for soft mask grid sizing
        decoded_masks:  List[List[dict]] = None,        # [B][num_masks_b] pre-decoded
    ) -> List[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """Process all masks for all samples in batch.

        Batched Strategy (Flatten -> Parallel Project -> Scatter):
            1. Extract feature vectors per mask (GPU tensor ops)
            2. Stack all features, project in single batched matmul per projector
            3. Scatter results back to per-sample structure

        Returns:
            result[b][m] = (mask_rgb, mask_depth, mask_geo) for sample b, mask m
            Each token: [1, 1024]
        """
        B = depth_maps.shape[0]
        dev = depth_maps.device

        # Soft mask grid size (decoupled from visual tokens — just a smoothing param)
        _, h_patches, w_patches = [int(x) for x in image_grid_thw[0].tolist()]
        h_soft = max(h_patches // 2, 1)
        w_soft = max(w_patches // 2, 1)

        # ---- Phase 1: FLATTEN — extract features per mask ----
        all_rgb_feats = []
        all_dep_feats = []
        all_geo_feats = []
        mask_counts = []
        
        gray_images = rgb_images.mean(dim=1)  # [B, H, W]
        depth_fs    = depth_maps.float()      # [B, H, W]
        g_means     = depth_fs.mean(dim=(1, 2))
        g_stds      = depth_fs.view(B, -1).std(dim=1, correction=0).clamp(min=0)
        g_maxs      = depth_fs.view(B, -1).max(dim=1).values.clamp(min=1e-8)

        for b in range(B):
            masks_b = rle_list[b]
            dec_b = decoded_masks[b] if decoded_masks is not None else None

            for m, rle in enumerate(masks_b):
                # Get binary + soft masks (pre-decoded in dataloader when available)
                if dec_b is not None and m < len(dec_b):
                    binary = dec_b[m]['binary']
                    soft2d = dec_b[m]['soft2d']
                    soft2d_dev = soft2d.to(dev) if torch.is_tensor(soft2d) else torch.from_numpy(soft2d).to(dev)
                else:
                    binary, soft2d = self._rle_to_soft_mask(rle, h_soft, w_soft, device=dev)
                    soft2d_dev = soft2d

                mask_t = torch.from_numpy(binary).to(device=dev, dtype=torch.bool) if isinstance(binary, np.ndarray) else binary.to(device=dev, dtype=torch.bool)

                # RGB features from raw image [12]
                rgb_feat = self._rgb_features(rgb_images[b], gray_images[b], mask_t)
                all_rgb_feats.append(rgb_feat)

                # Depth features [28]
                dep_feat = self._depth_features(
                    depth_maps[b:b+1], mask_t, soft2d_dev, h_soft, w_soft
                )
                all_dep_feats.append(dep_feat)

                # Geo features [16]
                geo_feat = self._geo_features(depth_fs[b], mask_t, g_means[b], g_stds[b], g_maxs[b])
                all_geo_feats.append(geo_feat)

            mask_counts.append(len(masks_b))

        # ---- Phase 2: PARALLEL — batch project all masks at once ----
        if len(all_rgb_feats) == 0:
            return [[] for _ in range(B)]

        # Stack: [total_masks, feat_dim] — single GPU kernel per projection
        rgb_stack = torch.stack(all_rgb_feats)       # [M_total, 12]
        dep_stack = torch.stack(all_dep_feats)       # [M_total, 28]
        geo_stack = torch.stack(all_geo_feats)       # [M_total, 16]

        rgb_tokens = self.rgb_proj(rgb_stack)        # [M_total, 1024]
        dep_tokens = self.depth_proj(dep_stack)      # [M_total, 1024]
        geo_tokens = self.geo_proj(geo_stack)        # [M_total, 1024]

        # ---- Phase 3: SCATTER — distribute back to per-sample structure ----
        result = []
        idx = 0
        for b in range(B):
            triples = []
            for m in range(mask_counts[b]):
                triples.append((
                    rgb_tokens[idx:idx+1],    # [1, 1024]
                    dep_tokens[idx:idx+1],    # [1, 1024]
                    geo_tokens[idx:idx+1],    # [1, 1024]
                ))
                idx += 1
            result.append(triples)
        return result

    def inject_into_text_embeds(
        self,
        text_embeds:    torch.Tensor,                # [B, L, 1024]
        mask_positions: List[List[int]],             # [B][num_masks_b] sorted <mask> start indices
        region_tokens:  List[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
        mask_token_len: int = 3,                     # <mask> = 3 tokens: < mask >
    ) -> torch.Tensor:                               # [B, L, 1024]
        """Replace each <mask> (3 tokens) with [mask_rgb, mask_depth, mask_geo].

        3 -> 3 replacement: sequence length UNCHANGED.

        Args:
            text_embeds:    [B, L, D] text token embeddings
            mask_positions: [B][num_masks] positions of <mask> starts
            region_tokens:  [B][num_masks] (mask_rgb, mask_depth, mask_geo) triples
            mask_token_len: number of tokens per <mask> (always 3: < mask >)

        Returns:
            embeds: [B, L, D] — same shape as input
        """
        B, L, D = text_embeds.shape

        for b in range(B):
            positions = mask_positions[b]
            tokens = region_tokens[b]

            for pos, (rgb, dep, geo) in zip(positions, tokens):
                if pos + mask_token_len <= L:
                    text_embeds[b, pos,     :] = rgb.squeeze(0)      # mask_rgb   [1024]
                    text_embeds[b, pos + 1, :] = dep.squeeze(0)      # mask_depth [1024]
                    text_embeds[b, pos + 2, :] = geo.squeeze(0)      # mask_geo   [1024]

        return text_embeds
