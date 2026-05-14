"""
SpatialVLM Micro — CE + Normalized MSE + Category Focal Loss
============================================================================

Training loss:

    L = CE(label_smoothing) + weight_sl1 · L_NormMSE + weight_cat · L_CatFocal

Where:
    - CE:          Cross-entropy on LM head output (single pass, no per-step)
    - L_NormMSE:   Normalized MSE on Number Head (distance/16, count/4)
    - L_CatFocal:  Focal Loss on Category Head predictions (mcq + left_right)
    - weight_sl1:  Weight for Normalized MSE loss (default 2.0)
    - weight_cat:  Weight for Category Focal loss (default 2.0)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialLoss(nn.Module):
    """SpatialVLM Micro training loss.

    Combines:
        1. Cross-entropy loss on LM head output
        2. Normalized MSE loss for numeric regression (Number Head)
           - distance targets normalized by /16.0 (domain max range)
           - count targets normalized by /4.0 (domain max count)
        3. Focal loss for category classification (Category Head)

    Args:
        weight_sl1:      Weight for Normalized MSE loss relative to CE loss.
        weight_cat:      Weight for Category Focal loss relative to CE loss.
        ignore_index:    Token index to ignore in CE loss (default: -100).
        label_smoothing: Label smoothing for CE loss (default: 0.1).
        focal_gamma:     Exponent for Focal Loss.
        dist_scale:      Normalization constant for distance targets (default: 16.0).
        count_scale:     Normalization constant for count targets (default: 4.0).
    """

    def __init__(
        self,
        weight_sl1: float = 2.0,
        weight_cat: float = 2.0,
        ignore_index: int = -100,
        label_smoothing: float = 0.1,
        focal_gamma: float = 2.0,
        dist_scale: float = 4.0,   # 16m / 4.0 = range [0, 4]
        count_scale: float = 1.0,  # 4 / 1.0 = range [1, 4]
    ):
        super().__init__()
        self.weight_sl1 = weight_sl1
        self.weight_cat = weight_cat
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.focal_gamma = focal_gamma
        self.dist_scale = dist_scale
        self.count_scale = count_scale

    def forward(
        self,
        logits:          torch.Tensor,         # [B, L, V]
        lm_targets:      torch.Tensor,         # [B, L]
        num_pred:        torch.Tensor,         # [B]
        num_gt:          torch.Tensor,         # [B]
        is_numeric:      torch.Tensor,         # [B]
        num_is_distance: torch.Tensor = None,  # [B] bool: True=distance, False=count
        cat_logits:      list = None,          # list of [N_masks] tensors or None
        cat_targets:     torch.Tensor = None,  # [B] target indices
        is_categorical:  torch.Tensor = None,  # [B] bool
        return_components: bool = False,
    ) -> torch.Tensor:
        """Compute training loss.

        L = CE + α · L_NormMSE + γ · L_CatFocal

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

        # --- 2. Normalized MSE Loss (Number Head) ---
        #     distance targets / 16.0 → [0, ~1], count targets / 4.0 → [0.25, 1.0]
        #     This balances gradient magnitudes between the two tasks.
        if is_numeric.any():
            pred = num_pred[is_numeric].float()
            target = num_gt[is_numeric].float()
            
            # Apply domain-aware normalization
            if num_is_distance is not None:
                dist_mask = num_is_distance[is_numeric]
                scale = torch.where(dist_mask, self.dist_scale, self.count_scale)
                pred = pred / scale
                target = target / scale
            
            loss_sl1 = F.mse_loss(pred, target, reduction='mean')
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
