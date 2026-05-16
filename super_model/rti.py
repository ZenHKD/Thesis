"""
MODULE: Region-Level Token Injection (RTI) — Super Model

Architecture:
- Tri-Source U-Net (RGB + Depth + Global Depth) with shared ViT semantics at bottleneck.
  Multi-scale mask pooling: concat features from 4 decoder levels
  (512 + 256 + 128 + 128 = 1024 real dims) → Linear(1024, 1024) alignment.
- Token 1 (mask_rgb):     RGB U-Net + object mask → object appearance features
- Token 2 (mask_depth):   Depth U-Net + object mask → object depth features
- Token 3 (global_depth): Depth U-Net + REVERSED mask → surrounding spatial context
  The global_depth token shares the same depth U-Net encoder but uses a separate
  decoder, and pools with the inverted mask to capture the spatial relationship
  between the object and its environment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import pycocotools.mask as mask_utils
import numpy as np

class UNetEncoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.e1 = nn.Sequential(nn.Conv2d(in_channels, 64, 3, stride=2, padding=1), nn.GELU()) # /2
        self.e2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU())          # /4
        self.e3 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.GELU())         # /8
        self.e4 = nn.Sequential(nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.GELU())        # /16

    def forward(self, x):
        f1 = self.e1(x)
        f2 = self.e2(f1)
        f3 = self.e3(f2)
        f4 = self.e4(f3)
        return f1, f2, f3, f4

class UNetDecoder(nn.Module):
    """Multi-scale UNet decoder with channels: 512 → 256 → 128 → 128.

    Returns intermediate feature maps for multi-scale mask pooling:
      conv3: [B, 512, H/8,  W/8 ]  → pool → [512]
      conv2: [B, 256, H/4,  W/4 ]  → pool → [256]
      conv1: [B, 128, H/2,  W/2 ]  → pool → [128]
      conv0: [B, 128, H,    W   ]  → pool → [128]
      ─────────────────────────────────────────────
      Concat: [1024] → Linear(1024, 1024) + LN → [1024] token

    100% real information — zero inflation.
    """

    def __init__(self, vit_dim=768, hidden_dim=1024):
        super().__init__()
        # Bottleneck: fuse encoder features + ViT semantics
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512 + vit_dim, 1024, 1),
            nn.GELU()
        )

        # Decoder levels with increased channels: 512 → 256 → 128 → 128
        self.up3 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv3 = nn.Sequential(nn.Conv2d(512 + 256, 512, 3, padding=1), nn.GELU())

        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv2 = nn.Sequential(nn.Conv2d(256 + 128, 256, 3, padding=1), nn.GELU())

        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv1 = nn.Sequential(nn.Conv2d(128 + 64, 128, 3, padding=1), nn.GELU())

        self.up0 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.conv0 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.GELU())

        # Final projection: concat of all scales (512+256+128+128=1024) → aligned 1024
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, f1, f2, f3, f4, vit_feat):
        """Returns all intermediate feature maps for multi-scale pooling.

        Returns:
            feat3: [B, 512, H/8,  W/8 ]
            feat2: [B, 256, H/4,  W/4 ]
            feat1: [B, 128, H/2,  W/2 ]
            feat0: [B, 128, H,    W   ]
        """
        # f4: [B, 512, H/16, W/16]
        # vit_feat: [B, 768, H/16, W/16]
        x = torch.cat([f4, vit_feat], dim=1)
        x = self.bottleneck(x)

        x = self.up3(x)
        x = torch.cat([x, f3], dim=1)
        feat3 = self.conv3(x)   # [B, 512, H/8, W/8]

        x = self.up2(feat3)
        x = torch.cat([x, f2], dim=1)
        feat2 = self.conv2(x)   # [B, 256, H/4, W/4]

        x = self.up1(feat2)
        x = torch.cat([x, f1], dim=1)
        feat1 = self.conv1(x)   # [B, 128, H/2, W/2]

        x = self.up0(feat1)
        feat0 = self.conv0(x)   # [B, 128, H, W]

        return feat3, feat2, feat1, feat0

class RTE(nn.Module):
    """Tri-Source RTI: RGB + Depth (object) + Global Depth (surrounding context).

    Three tokens per object region:
    - mask_rgb:     RGB features pooled inside the object mask
    - mask_depth:   Depth features pooled inside the object mask
    - global_depth: Depth features pooled OUTSIDE the object mask (reversed mask)
                    → captures surrounding spatial context for richer reasoning
    """

    def __init__(self, hidden_dim=1024, vit_dim=768):
        super().__init__()
        # RGB path: encoder + decoder
        self.rgb_encoder = UNetEncoder(in_channels=3)
        self.rgb_decoder = UNetDecoder(vit_dim=vit_dim, hidden_dim=hidden_dim)

        # Depth path (object interior): encoder + decoder
        self.depth_encoder = UNetEncoder(in_channels=1)
        self.depth_decoder = UNetDecoder(vit_dim=vit_dim, hidden_dim=hidden_dim)

        # Global depth path (surrounding context): shares encoder with depth,
        # but has its own decoder to learn different spatial features
        self.global_depth_decoder = UNetDecoder(vit_dim=vit_dim, hidden_dim=hidden_dim)

    def _multiscale_mask_pool(self, feat3, feat2, feat1, feat0, mask_t, b, decoder):
        """Multi-scale mask pooling: pool at 4 decoder levels and concat.

        Args:
            feat3: [B, 512, H/8,  W/8 ]
            feat2: [B, 256, H/4,  W/4 ]
            feat1: [B, 128, H/2,  W/2 ]
            feat0: [B, 128, H,    W   ]
            mask_t: [H, W] boolean mask at full resolution
            b: batch index
            decoder: UNetDecoder (to access proj layer)

        Returns:
            [1024] token — projected multi-scale features
        """
        # Downscale mask to each level's resolution
        mask_float = mask_t.unsqueeze(0).unsqueeze(0).float()  # [1, 1, H, W]

        # Level 0: full resolution — direct mask
        pixels_0 = feat0[b, :, mask_t].T             # [N0, 128]
        pooled_0 = pixels_0.mean(dim=0)              # [128]

        # Level 1: /2 resolution
        mask_1 = F.interpolate(mask_float, size=feat1.shape[2:], mode='nearest').squeeze() > 0.5
        if mask_1.sum() > 0:
            pixels_1 = feat1[b, :, mask_1].T         # [N1, 128]
            pooled_1 = pixels_1.mean(dim=0)          # [128]
        else:
            pooled_1 = feat1.new_zeros(128)

        # Level 2: /4 resolution
        mask_2 = F.interpolate(mask_float, size=feat2.shape[2:], mode='nearest').squeeze() > 0.5
        if mask_2.sum() > 0:
            pixels_2 = feat2[b, :, mask_2].T         # [N2, 256]
            pooled_2 = pixels_2.mean(dim=0)          # [256]
        else:
            pooled_2 = feat2.new_zeros(256)

        # Level 3: /8 resolution
        mask_3 = F.interpolate(mask_float, size=feat3.shape[2:], mode='nearest').squeeze() > 0.5
        if mask_3.sum() > 0:
            pixels_3 = feat3[b, :, mask_3].T         # [N3, 512]
            pooled_3 = pixels_3.mean(dim=0)          # [512]
        else:
            pooled_3 = feat3.new_zeros(512)

        # Concat all scales: 512 + 256 + 128 + 128 = 1024
        multi_scale = torch.cat([pooled_3, pooled_2, pooled_1, pooled_0], dim=-1)  # [1024]

        # Alignment projection
        return decoder.proj(multi_scale)  # [1024]

    def forward(
        self,
        rgb_images:     torch.Tensor,                  # [B, 3, H, W]
        depth_maps:     torch.Tensor,                  # [B, H, W]
        rle_list:       List[List[dict]],               # [B][num_masks_b]
        image_grid_thw: torch.Tensor,                  # [B, 3]
        decoded_masks:  List[List[dict]] = None,        # [B][num_masks_b] pre-decoded
        vit_feat:       torch.Tensor = None,            # [B, C, h, w]
    ) -> List[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        B, H, W = rgb_images.shape[0], rgb_images.shape[2], rgb_images.shape[3]
        dev = rgb_images.device
        dtype = rgb_images.dtype

        # Forward U-Net Encoders
        f1_r, f2_r, f3_r, f4_r = self.rgb_encoder(rgb_images)
        f1_d, f2_d, f3_d, f4_d = self.depth_encoder(depth_maps.unsqueeze(1).to(dtype))

        if vit_feat is None:
            vit_feat = torch.zeros(B, 768, f4_r.shape[2], f4_r.shape[3], device=dev, dtype=dtype)

        # Forward U-Net Decoders — returns multi-scale feature maps
        rgb_f3, rgb_f2, rgb_f1, rgb_f0 = self.rgb_decoder(f1_r, f2_r, f3_r, f4_r, vit_feat)
        dep_f3, dep_f2, dep_f1, dep_f0 = self.depth_decoder(f1_d, f2_d, f3_d, f4_d, vit_feat)

        # Global depth decoder: same encoder features, separate decoder weights
        gdep_f3, gdep_f2, gdep_f1, gdep_f0 = self.global_depth_decoder(f1_d, f2_d, f3_d, f4_d, vit_feat)

        out_tokens = []
        for b in range(B):
            sample_tokens = []
            masks = rle_list[b] if rle_list else []
            d_masks = decoded_masks[b] if decoded_masks else [None] * len(masks)
            
            for m_idx, mask_dict in enumerate(masks):
                if d_masks[m_idx] is not None:
                    binary_mask = d_masks[m_idx]['binary']
                else:
                    binary_mask = mask_utils.decode(mask_dict).astype(bool)
                
                mask_t = torch.from_numpy(binary_mask).to(device=dev) if isinstance(binary_mask, np.ndarray) else binary_mask.to(device=dev, dtype=torch.bool)
                
                if mask_t.sum() == 0:
                    rgb_tok = torch.zeros(1024, device=dev, dtype=dtype)
                    dep_tok = torch.zeros(1024, device=dev, dtype=dtype)
                    gdep_tok = torch.zeros(1024, device=dev, dtype=dtype)
                    sample_tokens.append((rgb_tok.unsqueeze(0), dep_tok.unsqueeze(0), gdep_tok.unsqueeze(0)))
                    continue

                # Token 1: mask_rgb — RGB features inside object mask
                rgb_tok = self._multiscale_mask_pool(
                    rgb_f3, rgb_f2, rgb_f1, rgb_f0, mask_t, b, self.rgb_decoder
                )

                # Token 2: mask_depth — Depth features inside object mask
                dep_tok = self._multiscale_mask_pool(
                    dep_f3, dep_f2, dep_f1, dep_f0, mask_t, b, self.depth_decoder
                )

                # Token 3: global_depth — Depth features OUTSIDE object mask (reversed)
                # Invert the mask: capture surrounding spatial context instead of object interior
                reversed_mask = ~mask_t  # [H, W] boolean — everything except the object

                if reversed_mask.sum() == 0:
                    # Edge case: object fills entire image → no surrounding context
                    gdep_tok = torch.zeros(1024, device=dev, dtype=dtype)
                else:
                    gdep_tok = self._multiscale_mask_pool(
                        gdep_f3, gdep_f2, gdep_f1, gdep_f0, reversed_mask, b, self.global_depth_decoder
                    )

                sample_tokens.append((rgb_tok.unsqueeze(0), dep_tok.unsqueeze(0), gdep_tok.unsqueeze(0)))
                
            out_tokens.append(sample_tokens)
            
        return out_tokens

    def inject_into_text_embeds(
        self,
        text_embeds:    torch.Tensor,                
        mask_positions: List[List[int]],             
        region_tokens:  List[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
        mask_token_len: int = 3,                     
    ) -> torch.Tensor:                               
        B, L, D = text_embeds.shape

        for b in range(B):
            positions = mask_positions[b]
            tokens = region_tokens[b]

            for pos, (rgb, dep, gdep) in zip(positions, tokens):
                if pos + mask_token_len <= L:
                    text_embeds[b, pos,     :] = rgb.squeeze(0)   # mask_rgb
                    text_embeds[b, pos + 1, :] = dep.squeeze(0)   # mask_depth
                    text_embeds[b, pos + 2, :] = gdep.squeeze(0)  # global_depth (reversed mask)

        return text_embeds
