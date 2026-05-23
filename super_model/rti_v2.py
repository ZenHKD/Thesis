import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple

class PolyphaseMultiplexingModule(nn.Module):
    """
    Core implementation of Layer-Spatial Multiplexing for ViT patch extraction.
    Uses Multi-Head Mask-Guided Pooling to extract rich features from the full 
    16x16 patch resolution (256 pixels) without positional query bias.
    """
    def __init__(self, vit_dim=768, out_dim=1024, num_layers=4, num_heads=4):
        super().__init__()
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.chunk_dim = out_dim // num_layers  # Usually 1024 / 4 = 256
        self.num_heads = num_heads
        
        # Value Projections for each of the 4 layers
        self.v_projs = nn.ModuleList([nn.Linear(vit_dim, self.chunk_dim) for _ in range(num_layers)])
        
        # Pattern Translators: [256 pixels -> 128 -> num_heads]
        self.mask_translators = nn.ModuleList([nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, num_heads)
        ) for _ in range(num_layers)])
        
        # Output projections to fuse the multi-head information back to chunk_dim
        self.out_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_heads * self.chunk_dim, self.chunk_dim),
                nn.LayerNorm(self.chunk_dim)
            ) for _ in range(num_layers)
        ])

    def forward(self, expanded_vit_layers: List[torch.Tensor], masks: torch.Tensor):
        """
        Process a batch of masks across multiple images.
        Args:
            expanded_vit_layers: List of 4 tensors, each [N_total, 640, 768]
            masks: [N_total, 320, 512] binary masks
        Returns:
            final_tokens: [N_total, 1024]
        """
        N = masks.shape[0]
        vit_device = expanded_vit_layers[0].device
        
        if N == 0:
            return torch.zeros(0, self.out_dim, device=vit_device)
        
        # Unfold the 320x512 mask into a 20x32 grid of full 16x16 pixel patches
        # [N_total, 20, 32, 16, 16] -> [N_total, 640, 256]
        m_full = masks.unfold(1, 16, 16).unfold(2, 16, 16).contiguous().view(N, 640, 256)
        
        layer_tokens = []
        
        for i in range(self.num_layers):
            vit_patches = expanded_vit_layers[i] # [N_total, 640, 768]
            
            # Project Values
            V = self.v_projs[i](vit_patches) # [N_total, 640, 256]
            
            # Mask Translator generates multi-head attention weights based on shape
            mask_weight = self.mask_translators[i](m_full) # [N_total, 640, num_heads]
            
            # Identify completely empty patches to mask them out
            is_empty = (m_full.sum(dim=-1) == 0).unsqueeze(-1) # [N_total, 640, 1]
            mask_weight.masked_fill_(is_empty, float('-inf'))
            
            # Softmax over the spatial dimension (640 patches)
            attn_weights = F.softmax(mask_weight, dim=1) # [N_total, 640, num_heads]
            
            # Apply attention to Values: [N_total, num_heads, 256]
            out_heads = torch.einsum('nph, npd -> nhd', attn_weights, V)
            
            # Flatten heads and project back to chunk_dim
            out_flat = out_heads.reshape(N, self.num_heads * self.chunk_dim) # [N_total, num_heads * 256]
            out_i = self.out_projs[i](out_flat) # [N_total, 256]
            
            layer_tokens.append(out_i)
        
        final_tokens = torch.cat(layer_tokens, dim=-1) # [N_total, 1024]
        return final_tokens


class RegionTokenInjectorV2(nn.Module):
    """
    RTI V2: Drop-in replacement for RTE class.
    Extracts 3 separate tokens (RGB, Depth, GlobalDepth) per mask using Polyphase Multiplexing.
    """
    def __init__(self, hidden_dim=1024, vit_dim=768):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vit_dim = vit_dim
        
        # Polyphase Multiplexing Feature Extractors
        self.rgb_polyphase = PolyphaseMultiplexingModule(vit_dim=vit_dim, out_dim=hidden_dim)
        # Shared for mask_depth and global_depth
        self.depth_polyphase = PolyphaseMultiplexingModule(vit_dim=vit_dim, out_dim=hidden_dim)
        
        # Separate projections for each pooled token type (matches original RTE)
        self.proj_rgb = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.proj_dep = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.proj_gdep = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        rgb_vit_intermediates:   List[torch.Tensor],  # 4 × [B, 641, 768]
        depth_vit_intermediates: List[torch.Tensor],  # 4 × [B, 641, 768]
        rle_list:       List[List[dict]],              
        image_grid_thw: torch.Tensor,                  
        decoded_masks:  List[torch.Tensor] = None,    # Expected: List[B] of [N_b, H, W]
    ) -> List[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """
        Returns:
            out_tokens: [B][num_masks_b] each = (rgb [1,1024], dep [1,1024], gdep [1,1024])
        """
        B = image_grid_thw.shape[0]
        device = rgb_vit_intermediates[0].device
        
        mask_counts = []
        valid_masks = []
        
        for b in range(B):
            masks_b = decoded_masks[b].float() # [N_b, 320, 512]
            mask_counts.append(masks_b.shape[0])
            if masks_b.shape[0] > 0:
                valid_masks.append(masks_b)
                
        if sum(mask_counts) == 0:
            return [[] for _ in range(B)]
            
        # Stack all masks across the batch
        all_masks = torch.cat(valid_masks, dim=0) # [N_total, 320, 512]
        all_masks_inv = 1.0 - all_masks
        
        # Prepare expanded ViT patches (strip CLS token first)
        counts_tensor = torch.tensor(mask_counts, device=device)
        expanded_rgb_vit = []
        expanded_depth_vit = []
        
        for i in range(4):
            r_patch = rgb_vit_intermediates[i][:, 1:, :] # [B, 640, 768]
            d_patch = depth_vit_intermediates[i][:, 1:, :] # [B, 640, 768]
            
            expanded_rgb_vit.append(torch.repeat_interleave(r_patch, counts_tensor, dim=0)) # [N_total, 640, 768]
            expanded_depth_vit.append(torch.repeat_interleave(d_patch, counts_tensor, dim=0)) # [N_total, 640, 768]
            
        # 1. Extract 1024-dim features using Polyphase Multiplexing (Vectorized!)
        f_rgb = self.rgb_polyphase(expanded_rgb_vit, all_masks)       # [N_total, 1024]
        f_dep = self.depth_polyphase(expanded_depth_vit, all_masks)   # [N_total, 1024]
        f_gdep = self.depth_polyphase(expanded_depth_vit, all_masks_inv) # [N_total, 1024]
        
        # 2. Apply the final distinct projections
        t_rgb = self.proj_rgb(f_rgb)   # [N_total, 1024]
        t_dep = self.proj_dep(f_dep)   # [N_total, 1024]
        t_gdep = self.proj_gdep(f_gdep) # [N_total, 1024]
        
        # 3. Format exactly like the original RTE output
        out_tokens = []
        start_idx = 0
        for count in mask_counts:
            batch_item_tokens = []
            if count > 0:
                for n in range(count):
                    idx = start_idx + n
                    batch_item_tokens.append((
                        t_rgb[idx:idx+1],   # [1, 1024]
                        t_dep[idx:idx+1],   # [1, 1024]
                        t_gdep[idx:idx+1]   # [1, 1024]
                    ))
                start_idx += count
            out_tokens.append(batch_item_tokens)
            
        return out_tokens
