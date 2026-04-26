"""
SpatialVLM Micro — CE + SmoothL1 + CategoryCE
============================================================================

Training loss:

    L = CE(label_smoothing=0.1) + α · L_SmoothL1 + γ · L_CatCE

Where:
    - CE:         Cross-entropy on LM head output (single pass, no per-step)
    - L_SmoothL1: SmoothL1 (Huber) on Number Head predictions (distance + count)
    - L_CatCE:    Cross-entropy on Category Head predictions (mcq + left_right)
    - α:          Weight for SmoothL1 (default 0.1)
    - γ:          Weight for CategoryCE (default 1.0)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialLoss(nn.Module):
    """SpatialVLM Micro training loss.

    Combines:
        1. Cross-entropy loss on LM head output
        2. SmoothL1 loss for numeric regression (Number Head)
        3. Cross-entropy loss for category classification (Category Head)

    Args:
        alpha:           Weight for SmoothL1 loss relative to CE loss.
        gamma:           Weight for CategoryCE loss relative to CE loss.
        ignore_index:    Token index to ignore in CE loss (default: -100).
        label_smoothing: Label smoothing for CE loss (default: 0.1).
    """

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 1.0,
        ignore_index: int = -100,
        label_smoothing: float = 0.1,
        beta: float = 0.05,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.beta = beta

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

        # --- 2. SmoothL1 loss (Number Head) with Dynamic Beta ---
        if is_numeric.any():
            pred = num_pred[is_numeric].float()
            target = num_gt[is_numeric].float()
            
            abs_diff = torch.abs(pred - target)
            
            # Target-Scaled Dynamic Beta (L2 zone proportional to target magnitude)
            # count=1 → β=0.05, count=3 → β=0.15, distance=10m → β=0.50
            dynamic_beta = torch.abs(target) * self.beta + 1e-6
            
            # SmoothL1 with dynamic beta (no relative error penalty)
            smooth_l1 = torch.where(
                abs_diff < dynamic_beta,
                0.5 * abs_diff**2 / dynamic_beta,
                abs_diff - 0.5 * dynamic_beta
            )
            
            loss_sl1 = smooth_l1.mean()
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
                        # Focal loss implementation (alpha=0.25, gamma=2.0)
                        ce_loss = F.cross_entropy(logits_b, target_b, reduction='none')
                        pt = torch.exp(-ce_loss)
                        f_loss = 0.25 * ((1 - pt) ** 2.0) * ce_loss
                        cat_losses.append(f_loss.mean())
            
            if cat_losses:
                loss_cat = torch.stack(cat_losses).mean()

        total = loss_ce + self.alpha * loss_sl1 + self.gamma * loss_cat

        if return_components:
            return total, {
                'ce': loss_ce.item(),
                'sl1': loss_sl1.item(),
                'cat': loss_cat.item(),
            }
        return total
