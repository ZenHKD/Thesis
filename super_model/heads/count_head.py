"""
MODULE: Count Head — Mask-Centric Tri-Source Regression

Position: After Decoder, parallel to LM Head
Input:  1. hidden state at <|count|> token [B_count, 1024] (decoder context)
        2. Per-mask RTI features, kept SEPARATE by modality:
           - rgb_list:  list of [N_masks, 1024] (appearance features)
           - dep_list:  list of [N_masks, 1024] (depth/spatial features)
           - gdep_list: list of [N_masks, 1024] (global depth — surrounding context)
        3. Visual tokens from ViT merger: list of [N_vis, 1024] (160 tokens)
           (lightweight usage — counting is mostly mask-centric)
Output: continuous scalar predictions [B_count] (positive, rounded to int at inference)

Design Rationale (vs DistanceHead):
    Counting is fundamentally mask-centric: "how many objects match this description?"
    The answer is primarily encoded in the NUMBER and FEATURES of masks,
    not in precise spatial/metric relationships.

    Key architectural differences from DistanceHead:
    1. Lightweight scene context: scene_ctx is weighted DOWN (×0.5 gate)
       because counting relies more on mask features than global scene layout.
    2. Shallower MLP: 4096 → 512 → 1 (vs 4096 → 1024 → 1 for distance)
       Counting is a simpler mapping — fewer parameters prevent overfitting.
    3. Softplus output: smooth positive activation better suited for small
       integer targets (1, 2, 3, ...) vs relu's hard zero gradient at 0.
    4. Mask count feature: N_masks is injected as an explicit scalar feature
       into the regression MLP (cheap but informative prior for counting).

Architecture:
    1. Scene Context (lightweight):
       q_scene = h_count → Linear(1024, 1024)
       scene_ctx = softmax(q_scene · vis^T / √1024) · vis → [1024]
       scene_ctx *= 0.5  (down-weighted — counting is mask-centric)

    2. Fused Query:
       q = Linear(concat(h_count, scene_ctx)) → [1024]  (2048 → 1024)

    3. Tri-Source Attention:
       att_rgb  = softmax(q · rgb^T  / √1024) · rgb   → [1024]
       att_dep  = softmax(q · dep^T  / √1024) · dep   → [1024]
       att_gdep = softmax(q · gdep^T / √1024) · gdep  → [1024]

    4. Count Regression MLP (shallower):
       concat(q, att_rgb, att_dep, att_gdep) [4096]
       → LN → Linear(4096 + 1, 512) → GELU → Dropout → Linear(512, 1) → softplus
                         ↑
                    N_masks scalar appended

Params: ~2.36M
    scene_q_proj: Linear(1024, 1024, bias=False) = 1,048,576
    fuse_proj:    Linear(2048, 1024)             = 2,098,176 → shared dim
    regression:   LN(4096) + Linear(4097, 512) + Linear(512, 1)
                  = 4096 + 4097×512 + 512 + 512 + 1 = ~2,102,785
    Total:        ~5,249,537 ≈ 5.25M
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CountHead(nn.Module):
    """Mask-centric Tri-Source head for count regression.

    Compared to DistanceHead:
    - Shallower MLP (4096+1 → 512 → 1) — counting is a simpler mapping
    - Softplus output — smooth positive, better for small integers
    - Down-weighted scene context (×0.5) — counting is mask-centric
    - N_masks scalar injected as explicit feature
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

        # Count Regression MLP (shallower than DistanceHead)
        # Input: concat(q, att_rgb, att_dep, att_gdep) [4096] + N_masks [1] = 4097
        self.regression = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4 + 1),
            nn.Linear(hidden_dim * 4 + 1, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
        )

        # Learnable gate for scene context weighting
        self.scene_gate = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        h_token: torch.Tensor,         # [B_count, 1024]
        rgb_list: list,                 # list of [N_masks, 1024]
        dep_list: list,                 # list of [N_masks, 1024]
        gdep_list: list,                # list of [N_masks, 1024]
        vis_list: list = None,          # list of [N_vis, 1024]
    ) -> torch.Tensor:
        """
        Args:
            h_token:   [B_count, 1024] — hidden states at <|count|> positions.
            rgb_list:  list of [N_masks_b, 1024] tensors.
            dep_list:  list of [N_masks_b, 1024] tensors.
            gdep_list: list of [N_masks_b, 1024] tensors.
            vis_list:  list of [N_vis, 1024] tensors (ViT visual tokens).

        Returns:
            [B_count] — predicted counts (positive via softplus)
        """
        B_count = h_token.shape[0]
        if B_count == 0:
            return torch.zeros(0, device=h_token.device, dtype=h_token.dtype)

        preds = []
        for i in range(B_count):
            h = h_token[i]  # [1024]

            # Step 1: Scene context via visual cross-attention (down-weighted)
            if vis_list is not None and vis_list[i] is not None and vis_list[i].shape[0] > 0:
                q_scene = self.scene_q_proj(h.unsqueeze(0))  # [1, 1024]
                vis = vis_list[i]  # [N_vis, 1024]
                attn_scores = (q_scene @ vis.T) / self.scale  # [1, N_vis]
                scene_ctx = (torch.softmax(attn_scores, dim=-1) @ vis).squeeze(0)  # [1024]
                scene_ctx = scene_ctx * self.scene_gate  # Down-weight: counting is mask-centric
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

            # Step 4: Concat + N_masks scalar → shallower regression
            n_masks = torch.tensor([rgb.shape[0]], device=h.device, dtype=h.dtype)
            combined = torch.cat([q, att_rgb, att_dep, att_gdep, n_masks], dim=-1)  # [4097]
            pred = self.regression(combined.unsqueeze(0)).squeeze(-1).squeeze(-1)
            preds.append(pred)

        return F.softplus(torch.stack(preds))  # [B_count] — smooth positive
