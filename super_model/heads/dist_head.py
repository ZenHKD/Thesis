"""
MODULE: Distance Head — Tri-Source + Visual Context Cross-Attention Regression

Position: After Decoder, parallel to LM Head
Input:  1. hidden state at <|dist|> token [B_dist, 1024] (decoder context)
        2. Per-mask RTI features, kept SEPARATE by modality:
           - rgb_list:  list of [N_masks, 1024] (appearance features)
           - dep_list:  list of [N_masks, 1024] (depth/spatial features)
           - gdep_list: list of [N_masks, 1024] (global depth — surrounding context)
        3. Visual tokens from ViT merger: list of [N_vis, 1024] (160 tokens)
Output: continuous scalar predictions [B_dist] (non-negative, meters)

Architecture (Tri-Source + Visual Context → Regression):
    1. Scene Context via Visual Cross-Attention:
       q_scene = h_dist → Linear(1024, 1024)
       scene_ctx = softmax(q_scene · vis_tokens^T / √1024) · vis_tokens  → [1024]

    2. Fused Query:
       q = Linear(concat(h_dist, scene_ctx)) → [1024]  (2048 → 1024)

    3. Tri-Source Attention (no key/value projection):
       att_rgb  = softmax(q · rgb^T  / √1024) · rgb   → [1024]
       att_dep  = softmax(q · dep^T  / √1024) · dep   → [1024]
       att_gdep = softmax(q · gdep^T / √1024) · gdep  → [1024]

    4. Regression MLP:
       concat(q, att_rgb, att_dep, att_gdep) [4096]
       → LN → Linear(4096, 1024) → GELU → Dropout → Linear(1024, 1) → relu

Params: ~7.35M
    scene_q_proj: Linear(1024, 1024, bias=False)          = 1,048,576
    fuse_proj:    Linear(2048, 1024)                       = 2,098,176
    regression:   LN(4096) + Linear(4096,1024) + Lin(1024,1) = ~4,204,545
    Total:        ~7,351,297 ≈ 7.35M
"""

import math
import torch
import torch.nn as nn


class DistanceHead(nn.Module):
    """Tri-Source + Visual Context cross-attention head for distance regression.

    Pattern: h_dist (query) first attends to ViT visual tokens for scene context,
             then attends independently to rgb, depth, global_depth keys/values
             at full 1024-dim → concat → regression MLP → scalar.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scale = math.sqrt(hidden_dim)

        # Scene context: cross-attention query projection
        self.scene_q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Fuse decoder context + scene context → attention query
        self.fuse_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )

        # Regression MLP: concat(q, att_rgb, att_dep, att_gdep) → scalar
        self.regression = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        h_token: torch.Tensor,         # [B_dist, 1024]
        rgb_list: list,                 # list of [N_masks, 1024]
        dep_list: list,                 # list of [N_masks, 1024]
        gdep_list: list,                # list of [N_masks, 1024]
        vis_list: list = None,          # list of [N_vis, 1024]
    ) -> torch.Tensor:
        """
        Args:
            h_token:   [B_dist, 1024] — hidden states at <|dist|> positions.
            rgb_list:  list of [N_masks_b, 1024] tensors.
            dep_list:  list of [N_masks_b, 1024] tensors.
            gdep_list: list of [N_masks_b, 1024] tensors.
            vis_list:  list of [N_vis, 1024] tensors (ViT visual tokens).
                       len(all lists) == B_dist.

        Returns:
            [B_dist] — predicted distances (non-negative via relu)
        """
        B_dist = h_token.shape[0]
        if B_dist == 0:
            return torch.zeros(0, device=h_token.device, dtype=h_token.dtype)

        preds = []
        for i in range(B_dist):
            h = h_token[i]  # [1024]

            # Step 1: Scene context via visual cross-attention
            if vis_list is not None and vis_list[i] is not None and vis_list[i].shape[0] > 0:
                q_scene = self.scene_q_proj(h.unsqueeze(0))  # [1, 1024]
                vis = vis_list[i]  # [N_vis, 1024]
                attn_scores = (q_scene @ vis.T) / self.scale  # [1, N_vis]
                scene_ctx = (torch.softmax(attn_scores, dim=-1) @ vis).squeeze(0)  # [1024]
            else:
                scene_ctx = torch.zeros_like(h)

            # Step 2: Fuse decoder context + scene context
            fused = torch.cat([h, scene_ctx], dim=-1)  # [2048]
            q = self.fuse_proj(fused.unsqueeze(0)).squeeze(0)  # [1024]

            rgb  = rgb_list[i]   # [N_masks, 1024]
            dep  = dep_list[i]   # [N_masks, 1024]
            gdep = gdep_list[i]  # [N_masks, 1024]

            # Step 3: Tri-Source Attention @ full 1024-dim
            score_rgb = (q @ rgb.T) / self.scale
            att_rgb = torch.softmax(score_rgb, dim=-1) @ rgb    # [1024]

            score_dep = (q @ dep.T) / self.scale
            att_dep = torch.softmax(score_dep, dim=-1) @ dep    # [1024]

            score_gdep = (q @ gdep.T) / self.scale
            att_gdep = torch.softmax(score_gdep, dim=-1) @ gdep  # [1024]

            # Step 4: Concat all sources + query → regression
            combined = torch.cat([q, att_rgb, att_dep, att_gdep], dim=-1)  # [4096]
            pred = self.regression(combined.unsqueeze(0)).squeeze(-1).squeeze(-1)
            preds.append(pred)

        return torch.relu(torch.stack(preds))  # [B_dist]
