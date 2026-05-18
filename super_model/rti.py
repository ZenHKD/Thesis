"""
MODULE: Region-Level Token Injection (RTI) — DPT-Style ViT Feature Extraction

Architecture (replaces U-Net with DPT from frozen ViT):
- Multi-layer feature extraction from ViT blocks 3, 6, 9, 12 (0-indexed: 2, 5, 8, 11)
- DPT-style Reassemble + Fusion to create 4 feature maps [B, 256, h, w]
  (where h, w = patch grid = H/16, W/16 for patch_size=16)
- Multi-feature mask pooling:
  Token 1 (mask_rgb):     RGB ViT features pooled inside object mask → [1024]
  Token 2 (mask_depth):   Depth ViT features pooled inside object mask → [1024]
  Token 3 (global_depth): Depth ViT features pooled OUTSIDE mask (reversed) → [1024]

Key insight: Leverage the frozen ViT's multi-layer representations to create 
richer per-object tokens. The ViT already encodes both RGB and Depth images (batch of 2), 
so we get pretrained features for free.

Each DPT reassembly produces [B, 256, h, w] at 4 scales → concat 4×256 = 1024.
Mask pool at each scale, concat → [1024] per token.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import pycocotools.mask as mask_utils
import numpy as np


class DPTReassemble(nn.Module):
    """DPT-style reassembly: project ViT hidden states to spatial feature maps.
    
    Takes flattened ViT tokens [N_patches, vit_dim] from a specific layer,
    reshapes to spatial [h, w, vit_dim], and projects to [out_channels, h, w].
    """
    def __init__(self, vit_dim: int = 768, out_channels: int = 256):
        super().__init__()
        self.proj = nn.Linear(vit_dim, out_channels)
        self.norm = nn.LayerNorm(vit_dim)
    
    def forward(self, tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        Args:
            tokens: [B, N, vit_dim] — ViT hidden states at one layer
            h, w: spatial patch grid dimensions
        Returns:
            [B, out_channels, h, w]
        """
        x = self.norm(tokens)           # [B, N, vit_dim]
        x = self.proj(x)                # [B, N, out_channels]
        B, N, C = x.shape
        x = x.view(B, h, w, C)
        x = x.permute(0, 3, 1, 2)      # [B, out_channels, h, w]
        return x


class DPTFusion(nn.Module):
    """DPT-style fusion block: refine + fuse skip connection.
    
    ResidualConv → InterpolateUp (if needed) → FuseConv
    """
    def __init__(self, channels: int = 256):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
        )
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [B, C, h, w] — current feature map
            skip: [B, C, h, w] — skip connection from deeper layer (optional)
        Returns:
            [B, C, h, w]
        """
        x = self.residual(x) + x  # Residual refinement
        if skip is not None:
            # Match spatial dims if needed
            if skip.shape[2:] != x.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = x + skip
        x = self.fuse(x)
        return x


class DPTFeatureExtractor(nn.Module):
    """DPT-style multi-layer feature extractor from ViT.
    
    Extracts intermediate hidden states from ViT blocks at layers 3, 6, 9, 12
    (0-indexed: 2, 5, 8, 11), reassembles them into spatial feature maps,
    and fuses them bottom-up (deepest → shallowest) like DPT.
    
    Output: 4 feature maps, each [B, 256, h, w] at the same patch resolution.
    """
    
    # ViT block indices to hook (0-indexed) — layers 3, 6, 9, 12
    HOOK_LAYERS = [2, 5, 8, 11]
    
    def __init__(self, vit_dim: int = 768, feat_channels: int = 256):
        super().__init__()
        self.vit_dim = vit_dim
        self.feat_channels = feat_channels
        
        # Reassemble: one per hooked layer
        self.reassemble = nn.ModuleList([
            DPTReassemble(vit_dim, feat_channels) for _ in range(4)
        ])
        
        # Fusion: bottom-up (layer 12 → 9 → 6 → 3)
        # Layer 12 (deepest): no skip input
        self.fusion_blocks = nn.ModuleList([
            DPTFusion(feat_channels),  # for layer 3 (shallowest, receives skip from layer 6)
            DPTFusion(feat_channels),  # for layer 6 (receives skip from layer 9)
            DPTFusion(feat_channels),  # for layer 9 (receives skip from layer 12)
            DPTFusion(feat_channels),  # for layer 12 (deepest, no skip)
        ])
        
        # Self-Attention on fused multi-scale features
        self.self_attn = nn.TransformerEncoderLayer(
            d_model=feat_channels * 4,
            nhead=8,
            dim_feedforward=feat_channels * 4 * 4,
            batch_first=True,
            norm_first=True
        )
        
        # Final projection: aligned 1024
        self.proj = nn.Sequential(
            nn.Linear(feat_channels * 4, feat_channels * 4),
            nn.LayerNorm(feat_channels * 4),
        )
    
    def extract_intermediate_features(
        self,
        visual_model: nn.Module,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Run ViT forward and capture intermediate hidden states.
        
        Args:
            visual_model: Qwen's visual module (model.visual)
            pixel_values: [N_total_patches, C_patch] — preprocessed patches
            grid_thw: [B, 3] — temporal, height, width patch grid
            
        Returns:
            List of 4 tensors, each [B, N_patches_per_image, vit_dim]
            corresponding to outputs of blocks 2, 5, 8, 11.
        """
        # Step 1: Patch embedding
        x = visual_model.patch_embed(pixel_values)  # [N_total, vit_dim]
        
        # Step 2: Position embedding
        # Qwen3.5 uses RoPE-style or learned pos embed 
        # The pos_embed is applied within the blocks via rotary, not added here
        # But we need to handle the grid-based position embeddings
        pos_ids = []
        for i in range(grid_thw.shape[0]):
            t, h, w = grid_thw[i].tolist()
            t, h, w = int(t), int(h), int(w)
            hpos = torch.arange(h, device=x.device).unsqueeze(1).expand(-1, w).reshape(-1)
            wpos = torch.arange(w, device=x.device).unsqueeze(0).expand(h, -1).reshape(-1)
            hpos = hpos.unsqueeze(1).repeat(1, int(t))
            wpos = wpos.unsqueeze(1).repeat(1, int(t))
            tpos = torch.arange(t, device=x.device).unsqueeze(0).expand(h*w, -1)
            pos = torch.stack([tpos, hpos, wpos], dim=-1)  # [h*w, t, 3]
            pos = pos.reshape(-1, 3)  # [t*h*w, 3]
            pos_ids.append(pos)
        
        # Use the model's own pos_embed
        if hasattr(visual_model, 'pos_embed') and visual_model.pos_embed is not None:
            pos_ids_cat = torch.cat(pos_ids, dim=0)
            # pos_embed expects [N, 3] -> lookup
            pos_embed = visual_model.pos_embed(pos_ids_cat)
            x = x + pos_embed
        
        # Step 3: Run through blocks, hooking intermediates
        intermediates = []
        for block_idx, block in enumerate(visual_model.blocks):
            x = block(x)
            if block_idx in self.HOOK_LAYERS:
                intermediates.append(x.clone())
        
        # Step 4: Split per image and reshape
        B = grid_thw.shape[0]
        patches_per_image = [
            int(grid_thw[i, 0] * grid_thw[i, 1] * grid_thw[i, 2])
            for i in range(B)
        ]
        
        result = []
        for layer_feats in intermediates:
            # layer_feats: [N_total, vit_dim]
            split_feats = layer_feats.split(patches_per_image, dim=0)
            # Pad to same size and stack
            max_n = max(f.shape[0] for f in split_feats)
            stacked = torch.stack([
                F.pad(f, (0, 0, 0, max_n - f.shape[0])) for f in split_feats
            ])  # [B, max_n, vit_dim]
            result.append(stacked)
        
        return result  # 4 × [B, N, vit_dim]
    
    def forward(
        self,
        layer_features: List[torch.Tensor],  # 4 × [B, N, vit_dim]
        h: int,
        w: int,
    ) -> torch.Tensor:
        """Reassemble + fuse multi-layer features + self-attention.
        
        Args:
            layer_features: List of 4 tensors [B, N, vit_dim] from hook layers
            h, w: patch grid spatial dimensions
            
        Returns:
            out_spatial: [B, 1024, h, w] (Self-Attended fused feature map)
        """
        # Reassemble each layer to spatial feature maps
        # Order: [layer_3, layer_6, layer_9, layer_12]
        spatial_feats = []
        for i, (feats, reassemble) in enumerate(zip(layer_features, self.reassemble)):
            spatial = reassemble(feats, h, w)  # [B, 256, h, w]
            spatial_feats.append(spatial)
        
        # Bottom-up fusion: deepest (layer 12) → shallowest (layer 3)
        # spatial_feats[3] = layer 12, spatial_feats[2] = layer 9, etc.
        
        # Layer 12: no skip
        f3 = self.fusion_blocks[3](spatial_feats[3])
        # Layer 9: skip from layer 12
        f2 = self.fusion_blocks[2](spatial_feats[2], skip=f3)
        # Layer 6: skip from layer 9
        f1 = self.fusion_blocks[1](spatial_feats[1], skip=f2)
        # Layer 3: skip from layer 6
        f0 = self.fusion_blocks[0](spatial_feats[0], skip=f1)
        
        # Concat: [B, 256×4, h, w] = [B, 1024, h, w]
        fused = torch.cat([f3, f2, f1, f0], dim=1)
        
        # Self-Attention (pixels attend to pixels)
        B, C, H, W = fused.shape
        seq = fused.view(B, C, H * W).permute(0, 2, 1)  # [B, H*W, 1024]
        seq = self.self_attn(seq)                       # [B, H*W, 1024]
        out_spatial = seq.permute(0, 2, 1).view(B, C, H, W) # [B, 1024, H, W]
        
        return out_spatial


class RTE(nn.Module):
    """DPT-based RTI: Multi-layer ViT features for RGB + Depth mask pooling.

    Three tokens per object region (same interface as before):
    - mask_rgb:     RGB ViT features pooled inside the object mask → [1024]
    - mask_depth:   Depth ViT features pooled inside the object mask → [1024]
    - global_depth: Depth ViT features pooled OUTSIDE the object mask → [1024]

    Architecture change: U-Net encoders/decoders → DPT reassemble+fusion from ViT.
    The ViT intermediate features are extracted once for both RGB and Depth images
    (batched through the same ViT), then split and processed by separate DPT heads.
    """

    def __init__(self, hidden_dim=1024, vit_dim=768):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vit_dim = vit_dim
        
        # DPT feature extractors (trainable reassemble + fusion heads)
        # RGB path
        self.rgb_dpt = DPTFeatureExtractor(vit_dim=vit_dim, feat_channels=256)
        
        # Depth object path  
        self.depth_dpt = DPTFeatureExtractor(vit_dim=vit_dim, feat_channels=256)
        
        # Global depth path (surrounding context — separate fusion weights)
        self.global_depth_dpt = DPTFeatureExtractor(vit_dim=vit_dim, feat_channels=256)

    def _batch_mask_pool(self, f_spatial, masks, b, dpt):
        """Batched mask pooling from 1024-dim spatial map.

        Args:
            f_spatial: [B, 1024, h, w] — Self-Attended fused feature map
            masks: [N_masks, H, W] float tensor (0/1) at original image resolution
            b: batch index
            dpt: DPTFeatureExtractor (to access proj layer)

        Returns:
            [N_masks, 1024] — one token per mask
        """
        N = masks.shape[0]
        if N == 0:
            return torch.zeros(0, self.hidden_dim, device=f_spatial.device, dtype=f_spatial.dtype)

        masks_4d = masks.unsqueeze(1)  # [N, 1, H, W]

        # Interpolate masks to feature map resolution
        m = F.interpolate(masks_4d, size=f_spatial.shape[2:], mode='nearest')
        feat = f_spatial[b].unsqueeze(0)  # [1, 1024, h, w]
        
        # Pool
        area = m.sum(dim=(2, 3)).clamp(min=1)
        pooled = (feat * m).sum(dim=(2, 3)) / area  # [N, 1024]

        # Project: [N, 1024]
        return dpt.proj(pooled)

    def forward(
        self,
        rgb_vit_intermediates:   List[torch.Tensor],  # 4 × [B, N, 768]
        depth_vit_intermediates: List[torch.Tensor],  # 4 × [B, N, 768]
        rle_list:       List[List[dict]],              # [B][num_masks_b]
        image_grid_thw: torch.Tensor,                  # [B, 3] (for RGB)
        decoded_masks:  List[List[dict]] = None,
    ) -> List[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """
        Args:
            rgb_vit_intermediates:   4 × [B, N, 768] — intermediate ViT features for RGB
            depth_vit_intermediates: 4 × [B, N, 768] — intermediate ViT features for Depth
            rle_list:       [B][num_masks_b] — RLE masks
            image_grid_thw: [B, 3] — patch grid (t, h, w) for spatial reshape
            decoded_masks:  pre-decoded masks
            
        Returns:
            out_tokens: [B][num_masks_b] each = (rgb [1,1024], dep [1,1024], gdep [1,1024])
        """
        B = image_grid_thw.shape[0]
        dev = rgb_vit_intermediates[0].device
        dtype = rgb_vit_intermediates[0].dtype
        
        # Get patch grid dimensions (use first sample — all same resolution)
        _, h_patches, w_patches = [int(x) for x in image_grid_thw[0].tolist()]
        
        # Run DPT reassemble+fusion+attention for each modality path
        rgb_spatial = self.rgb_dpt(rgb_vit_intermediates, h_patches, w_patches)
        dep_spatial = self.depth_dpt(depth_vit_intermediates, h_patches, w_patches)
        gdep_spatial = self.global_depth_dpt(depth_vit_intermediates, h_patches, w_patches)

        out_tokens = []
        for b in range(B):
            masks_rle = rle_list[b] if rle_list else []
            d_masks = decoded_masks[b] if decoded_masks else [None] * len(masks_rle)
            n_masks = len(masks_rle)

            if n_masks == 0:
                out_tokens.append([])
                continue

            # Decode all masks and stack
            mask_list = []
            valid_flags = []
            for m_idx, mask_dict in enumerate(masks_rle):
                if d_masks[m_idx] is not None:
                    binary_mask = d_masks[m_idx]['binary']
                else:
                    binary_mask = mask_utils.decode(mask_dict).astype(bool)

                if isinstance(binary_mask, np.ndarray):
                    mask_t = torch.from_numpy(binary_mask).to(device=dev, dtype=torch.bool)
                else:
                    mask_t = binary_mask.to(device=dev, dtype=torch.bool)

                valid_flags.append(mask_t.sum() > 0)
                mask_list.append(mask_t)

            masks_stacked = torch.stack(mask_list, dim=0).to(dtype=dtype)  # [N, H, W]
            reversed_masks = 1.0 - masks_stacked

            # Batch pool all masks
            rgb_tokens = self._batch_mask_pool(
                rgb_spatial, masks_stacked, b, self.rgb_dpt
            )  # [N, 1024]

            dep_tokens = self._batch_mask_pool(
                dep_spatial, masks_stacked, b, self.depth_dpt
            )  # [N, 1024]

            gdep_tokens = self._batch_mask_pool(
                gdep_spatial, reversed_masks, b, self.global_depth_dpt
            )  # [N, 1024]

            # Zero out invalid masks
            sample_tokens = []
            for m_idx in range(n_masks):
                if valid_flags[m_idx]:
                    sample_tokens.append((
                        rgb_tokens[m_idx:m_idx+1],
                        dep_tokens[m_idx:m_idx+1],
                        gdep_tokens[m_idx:m_idx+1],
                    ))
                else:
                    z = torch.zeros(1, self.hidden_dim, device=dev, dtype=dtype)
                    sample_tokens.append((z, z, z))

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
