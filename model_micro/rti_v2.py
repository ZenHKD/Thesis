"""
MODULE: Region-Level Token Injection v2 (RTI v2)

Architecture:
- Dual U-Net (RGB + Depth) with shared ViT semantics at the bottleneck.
- Geo Encoder (64x64 CNN) for shape understanding.
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
    def __init__(self, vit_dim=768):
        super().__init__()
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512 + vit_dim, 1024, 1),
            nn.GELU()
        )
        self.up3 = nn.ConvTranspose2d(1024, 256, 2, stride=2)
        self.conv3 = nn.Sequential(nn.Conv2d(256 + 256, 256, 3, padding=1), nn.GELU())

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = nn.Sequential(nn.Conv2d(128 + 128, 128, 3, padding=1), nn.GELU())

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv1 = nn.Sequential(nn.Conv2d(64 + 64, 64, 3, padding=1), nn.GELU())

        self.up0 = nn.ConvTranspose2d(64, 128, 2, stride=2)
        self.conv0 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.GELU())

        self.pixel_expand = nn.Sequential(
            nn.Linear(128, 256),
            nn.GELU()
        )

        self.proj = nn.Sequential(
            nn.Linear(256, 1024),
            nn.LayerNorm(1024)
        )

    def forward(self, f1, f2, f3, f4, vit_feat):
        # f4: [B, 512, H/16, W/16]
        # vit_feat: [B, 768, H/16, W/16]
        x = torch.cat([f4, vit_feat], dim=1)
        x = self.bottleneck(x)

        x = self.up3(x)
        x = torch.cat([x, f3], dim=1)
        x = self.conv3(x)

        x = self.up2(x)
        x = torch.cat([x, f2], dim=1)
        x = self.conv2(x)

        x = self.up1(x)
        x = torch.cat([x, f1], dim=1)
        x = self.conv1(x)

        x = self.up0(x)
        x = self.conv0(x)

        return x

class RTE_v2(nn.Module):
    def __init__(self, hidden_dim=1024, vit_dim=768):
        super().__init__()
        self.rgb_encoder = UNetEncoder(in_channels=3)
        self.rgb_decoder = UNetDecoder(vit_dim=vit_dim)
        
        self.depth_encoder = UNetEncoder(in_channels=1)
        self.depth_decoder = UNetDecoder(vit_dim=vit_dim)

        self.geo_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

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

        # Forward U-Nets
        f1_r, f2_r, f3_r, f4_r = self.rgb_encoder(rgb_images)
        f1_d, f2_d, f3_d, f4_d = self.depth_encoder(depth_maps.unsqueeze(1).to(dtype))

        if vit_feat is None:
            vit_feat = torch.zeros(B, 768, f4_r.shape[2], f4_r.shape[3], device=dev, dtype=dtype)

        rgb_feat_map = self.rgb_decoder(f1_r, f2_r, f3_r, f4_r, vit_feat)
        depth_feat_map = self.depth_decoder(f1_d, f2_d, f3_d, f4_d, vit_feat)

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
                    geo_tok = torch.zeros(1024, device=dev, dtype=dtype)
                    sample_tokens.append((rgb_tok.unsqueeze(0), dep_tok.unsqueeze(0), geo_tok.unsqueeze(0)))
                    continue

                # Mask Pooling at full resolution
                rgb_pixels = rgb_feat_map[b, :, mask_t].T  # [N, 128]
                dep_pixels = depth_feat_map[b, :, mask_t].T  # [N, 128]
                
                # Expand to 256 BEFORE pooling
                rgb_expanded = self.rgb_decoder.pixel_expand(rgb_pixels)  # [N, 256]
                dep_expanded = self.depth_decoder.pixel_expand(dep_pixels)  # [N, 256]
                
                # Global Average Pooling
                rgb_pooled = rgb_expanded.mean(dim=0)  # [256]
                dep_pooled = dep_expanded.mean(dim=0)  # [256]
                
                # Final projection
                rgb_tok = self.rgb_decoder.proj(rgb_pooled)  # [1024]
                dep_tok = self.depth_decoder.proj(dep_pooled)  # [1024]

                # Geo feature (Crop and resize to 64x64)
                ys, xs = torch.where(mask_t)
                y0, y1 = ys.min().item(), ys.max().item()
                x0, x1 = xs.min().item(), xs.max().item()
                
                cropped_mask = mask_t[y0:y1+1, x0:x1+1].to(dtype).unsqueeze(0).unsqueeze(0)
                resized_mask = F.interpolate(cropped_mask, size=(64, 64), mode='bilinear', align_corners=False)
                
                geo_tok = self.geo_encoder(resized_mask).squeeze(0)
                
                sample_tokens.append((rgb_tok.unsqueeze(0), dep_tok.unsqueeze(0), geo_tok.unsqueeze(0)))
                
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

            for pos, (rgb, dep, geo) in zip(positions, tokens):
                if pos + mask_token_len <= L:
                    text_embeds[b, pos,     :] = rgb.squeeze(0)
                    text_embeds[b, pos + 1, :] = dep.squeeze(0)
                    text_embeds[b, pos + 2, :] = geo.squeeze(0)

        return text_embeds
