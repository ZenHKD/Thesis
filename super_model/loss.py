"""
SpatialVLM Super — 4-Head Training Loss
============================================================================

Training loss (5 components):

    L = CE + w_dist · L_Dist + w_count · L_Count + w_mcq · L_MCQ + w_lr · L_LR

Where:
    - CE:       Cross-entropy on LM head output (label smoothing)
    - L_Dist:   Normalized MSE for Distance Head (target / dist_scale)
    - L_Count:  Normalized MSE for Count Head (target / count_scale)
    - L_MCQ:    Focal Loss for MCQ Head (multi-class over N masks)
    - L_LR:     Binary CE for LeftRight Head (2-class)

Each head has its own weight to allow independent tuning without
cross-task interference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialLoss(nn.Module):
    """SpatialVLM Super training loss — 4 independent head losses.

    Each head contributes independently to the total loss:
        1. LM CE:     Cross-entropy on text logits (category label)
        2. Distance:  Normalized MSE (pred/scale vs gt/scale)
        3. Count:     Normalized MSE (pred/scale vs gt/scale)
        4. MCQ:       Focal Loss over N candidate masks
        5. LeftRight: Binary CE over 2 masks

    Compared to model_micro SpatialLoss:
        - Separate weights for distance vs count (was shared weight_sl1)
        - Separate weights for MCQ vs LR (was shared weight_cat)
        - LR uses standard CE instead of Focal (only 2 classes, no imbalance)
    """

    def __init__(
        self,
        weight_dist:     float = 2.0,
        weight_count:    float = 2.0,
        weight_mcq:      float = 2.0,
        weight_lr:       float = 2.0,
        ignore_index:    int   = -100,
        label_smoothing: float = 0.1,
        focal_gamma:     float = 2.0,
        dist_scale:      float = 1.0,   # raw meters, no normalization
        count_scale:     float = 1.0,   # counts in [1, 4] → as-is
    ):
        super().__init__()
        self.weight_dist  = weight_dist
        self.weight_count = weight_count
        self.weight_mcq   = weight_mcq
        self.weight_lr    = weight_lr

        self.ignore_index    = ignore_index
        self.label_smoothing = label_smoothing
        self.focal_gamma     = focal_gamma
        self.dist_scale      = dist_scale
        self.count_scale     = count_scale

    def forward(
        self,
        logits:           torch.Tensor,         # [B, L, V] — LM head output
        lm_targets:       torch.Tensor,         # [B, L]    — token labels (-100 masked)
        # Distance Head
        dist_pred:        torch.Tensor,         # [B]       — distance predictions
        dist_gt:          torch.Tensor = None,   # [B]       — distance ground truth
        is_distance:      torch.Tensor = None,   # [B] bool  — True if distance task
        # Count Head
        count_pred:       torch.Tensor = None,   # [B]       — count predictions
        count_gt:         torch.Tensor = None,   # [B]       — count ground truth
        is_count:         torch.Tensor = None,   # [B] bool  — True if count task
        # MCQ Head
        mcq_logits:       list = None,           # list of [N_masks] tensors or None
        mcq_targets:      torch.Tensor = None,   # [B] target mask indices
        is_mcq:           torch.Tensor = None,   # [B] bool  — True if mcq task
        # LeftRight Head
        lr_logits:        list = None,           # list of [2] tensors or None
        lr_targets:       torch.Tensor = None,   # [B] target (0=left, 1=right)
        is_lr:            torch.Tensor = None,   # [B] bool  — True if left_right task
        # Options
        return_components: bool = False,
    ) -> torch.Tensor:
        """Compute total training loss.

        L = CE + w_dist·L_Dist + w_count·L_Count + w_mcq·L_MCQ + w_lr·L_LR

        Returns:
            Scalar loss, or (loss, components_dict) if return_components=True.
        """
        device = lm_targets.device

        # ─── 1. Cross-entropy on LM head ───
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = lm_targets[:, 1:].contiguous()

        if (shift_labels != self.ignore_index).sum() == 0:
            loss_ce = shift_logits.sum() * 0.0
        else:
            loss_ce = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.ignore_index,
                label_smoothing=self.label_smoothing,
            )

        # ─── 2. Distance Head — Normalized MSE ───
        loss_dist = torch.tensor(0.0, device=device)
        if is_distance is not None and is_distance.any():
            pred = dist_pred[is_distance].float() / self.dist_scale
            target = dist_gt[is_distance].float() / self.dist_scale
            loss_dist = F.mse_loss(pred, target, reduction='mean')

        # ─── 3. Count Head — Normalized MSE ───
        loss_count = torch.tensor(0.0, device=device)
        if is_count is not None and is_count.any():
            pred = count_pred[is_count].float() / self.count_scale
            target = count_gt[is_count].float() / self.count_scale
            loss_count = F.mse_loss(pred, target, reduction='mean')

        # ─── 4. MCQ Head — Focal Loss ───
        loss_mcq = torch.tensor(0.0, device=device)
        if mcq_logits is not None and is_mcq is not None and is_mcq.any():
            mcq_losses = []
            for b in range(len(mcq_logits)):
                if is_mcq[b] and mcq_logits[b] is not None:
                    logits_b = mcq_logits[b].float().unsqueeze(0)   # [1, N_masks]
                    target_b = mcq_targets[b].unsqueeze(0).to(device)  # [1]

                    # Validate target index is within range
                    if target_b.item() < logits_b.shape[1]:
                        ce = F.cross_entropy(logits_b, target_b, reduction='none')
                        pt = torch.exp(-ce)
                        focal = ((1 - pt) ** self.focal_gamma) * ce
                        mcq_losses.append(focal.mean())

            if mcq_losses:
                loss_mcq = torch.stack(mcq_losses).mean()

        # ─── 5. LeftRight Head — Binary CE ───
        #     Standard CE (no focal): only 2 balanced classes, no imbalance concern.
        loss_lr = torch.tensor(0.0, device=device)
        if lr_logits is not None and is_lr is not None and is_lr.any():
            lr_losses = []
            for b in range(len(lr_logits)):
                if is_lr[b] and lr_logits[b] is not None:
                    logits_b = lr_logits[b].float().unsqueeze(0)   # [1, 2]
                    target_b = lr_targets[b].unsqueeze(0).to(device)  # [1]

                    if target_b.item() < logits_b.shape[1]:
                        lr_losses.append(
                            F.cross_entropy(logits_b, target_b, reduction='mean')
                        )

            if lr_losses:
                loss_lr = torch.stack(lr_losses).mean()

        # ─── Total ───
        total = (
            loss_ce
            + self.weight_dist  * loss_dist
            + self.weight_count * loss_count
            + self.weight_mcq   * loss_mcq
            + self.weight_lr    * loss_lr
        )

        if return_components:
            return total, {
                'ce':    loss_ce.item(),
                'dist':  loss_dist.item(),
                'count': loss_count.item(),
                'mcq':   loss_mcq.item(),
                'lr':    loss_lr.item(),
            }
        return total
