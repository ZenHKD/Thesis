"""
SpatialVLM Micro — CE + SmoothL1 + Category Focal Loss
============================================================================

Training loss:

    L = CE(label_smoothing) + weight_sl1 · L_SmoothL1 + weight_cat · L_CatFocal

Where:
    - CE:         Cross-entropy on LM head output (single pass, no per-step)
    - L_SmoothL1: SmoothL1 (Huber) on Number Head predictions (distance + count)
    - L_CatFocal: Focal Loss on Category Head predictions (mcq + left_right)
    - weight_sl1: Weight for SmoothL1 loss (default 0.5)
    - weight_cat: Weight for Category Focal loss (default 2.0)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialLoss(nn.Module):
    """SpatialVLM Micro training loss.

    Combines:
        1. Cross-entropy loss on LM head output
        2. SmoothL1 loss for numeric regression (Number Head)
        3. Focal loss for category classification (Category Head)

    Args:
        weight_sl1:      Weight for SmoothL1 loss relative to CE loss.
        weight_cat:      Weight for Category Focal loss relative to CE loss.
        ignore_index:    Token index to ignore in CE loss (default: -100).
        label_smoothing: Label smoothing for CE loss (default: 0.1).
        focal_gamma:     Exponent for Focal Loss.
    """

    def __init__(
        self,
        weight_sl1: float = 0.5,
        weight_cat: float = 2.0,
        ignore_index: int = -100,
        label_smoothing: float = 0.1,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.weight_sl1 = weight_sl1
        self.weight_cat = weight_cat
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.focal_gamma = focal_gamma

    def forward(
        self,
        logits:          torch.Tensor,         # [B, L, V]
        lm_targets:      torch.Tensor,         # [B, L]
        num_pred:        torch.Tensor,         # [B]
        num_gt:          torch.Tensor,         # [B]
        is_numeric:      torch.Tensor,         # [B]
        cat_logits:      list = None,          # list of [N_masks] tensors or None
        cat_targets:     torch.Tensor = None,  # [B] target indices
        is_categorical:  torch.Tensor = None,  # [B] bool
        return_components: bool = False,
    ) -> torch.Tensor:
        """Compute training loss.

        L = CE + α · L_SmoothL1 + γ · L_CatCE

        Returns:
            Scalar loss (or tuple with component dict if return_components=True).
        """
        # --- 1. Cross-entropy ---
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

        # --- 2. Log-L1 Loss (Number Head) for Scale & Relative Invariance ---
        if is_numeric.any():
            pred = num_pred[is_numeric].float()
            target = num_gt[is_numeric].float()
            
            # Clamp prediction to be non-negative to avoid NaN in log
            pred = torch.clamp(pred, min=0.0)
            
            # Convert to log space (add 1 to prevent log(0))
            # This compresses large distances and naturally penalizes relative percentage errors.
            log_pred = torch.log(pred + 1.0)
            log_target = torch.log(target + 1.0)
            
            # SmoothL1 in Log space (Huber Log Loss). 
            # beta=0.1 means if relative error is <10%, use L2. Otherwise L1.
            loss_sl1 = F.smooth_l1_loss(log_pred, log_target, reduction='mean', beta=0.1)
        else:
            loss_sl1 = torch.tensor(0.0, device=lm_targets.device)

        # --- 3. Category Head CE loss (MCQ + Left/Right) ---
        loss_cat = torch.tensor(0.0, device=lm_targets.device)
        if cat_logits is not None and is_categorical is not None and is_categorical.any():
            cat_losses = []
            for b in range(len(cat_logits)):
                if is_categorical[b] and cat_logits[b] is not None:
                    # cat_logits[b]: [N_masks] raw scores
                    # cat_targets[b]: int (target mask index)
                    logits_b = cat_logits[b].float().unsqueeze(0)  # [1, N_masks]
                    target_b = cat_targets[b].unsqueeze(0).to(logits_b.device)  # [1]
                    
                    # Validate target index is within range
                    if target_b.item() < logits_b.shape[1]:
                        # Focal loss implementation (alpha=1.0)
                        ce_loss = F.cross_entropy(logits_b, target_b, reduction='none')
                        pt = torch.exp(-ce_loss)
                        f_loss = ((1 - pt) ** self.focal_gamma) * ce_loss
                        cat_losses.append(f_loss.mean())
            
            if cat_losses:
                loss_cat = torch.stack(cat_losses).mean()

        total = loss_ce + self.weight_sl1 * loss_sl1 + self.weight_cat * loss_cat

        if return_components:
            return total, {
                'ce': loss_ce.item(),
                'sl1': loss_sl1.item(),
                'cat': loss_cat.item(),
            }
        return total
