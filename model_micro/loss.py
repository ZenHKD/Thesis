"""
SpatialVLM Micro — CE + SmoothL1
============================================================================

Training loss:

    L = CE(label_smoothing=0.1) + α · L_SmoothL1

Where:
    - CE:         Cross-entropy on LM head output (single pass, no per-step)
    - L_SmoothL1: SmoothL1 (Huber) on Number Head predictions (distance + count)
    - α:          Weight for SmoothL1 (default 0.1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialLoss(nn.Module):
    """SpatialVLM Micro v2 training loss.

    Combines:
        1. Cross-entropy loss on LM head output
        2. SmoothL1 loss for numeric regression (Number Head)

    Args:
        alpha:           Weight for SmoothL1 loss relative to CE loss.
        ignore_index:    Token index to ignore in CE loss (default: -100).
        label_smoothing: Label smoothing for CE loss (default: 0.1).
    """

    def __init__(
        self,
        alpha: float = 0.1,
        ignore_index: int = -100,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits:          torch.Tensor,         # [B, L, V]
        lm_targets:      torch.Tensor,         # [B, L]
        num_pred:        torch.Tensor,         # [B]
        num_gt:          torch.Tensor,         # [B]
        is_numeric:      torch.Tensor,         # [B]
        return_components: bool = False,
    ) -> torch.Tensor:
        """Compute training loss.

        L = CE + α · L_SmoothL1

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

        # --- 2. SmoothL1 loss (Number Head) ---
        if is_numeric.any():
            loss_sl1 = F.smooth_l1_loss(
                num_pred[is_numeric].float(),
                num_gt[is_numeric].float(),
                beta=1.0,
            )
        else:
            loss_sl1 = torch.tensor(0.0, device=lm_targets.device)

        total = loss_ce + self.alpha * loss_sl1

        if return_components:
            return total, {
                'ce': loss_ce.item(),
                'sl1': loss_sl1.item(),
            }
        return total
