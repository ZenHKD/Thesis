"""
SpatialVLM Micro — Uniform Per-Step CE + SmoothL1
============================================================================

Training loss (single-stage):

    L = (1/T) · Σ_t CE^(t) + α · L_SmoothL1

Where:
    - CE^(t):  Per-step cross-entropy at loop step t
    - T:       Number of loop steps (T_max = 4)
    - α:       Weight for SmoothL1 (Number Head) loss

All loop steps receive equal gradient weight (simple LoopLM).

L_SmoothL1: SmoothL1 (Huber) loss on Number Head predictions.
       Active only for numeric samples (distance + count).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialLoss(nn.Module):
    """Uniform-weighted LoopLM loss for SpatialVLM Micro.

    Combines:
        1. Uniform-averaged per-step CE loss across all loop steps
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

    def _per_step_ce(
        self,
        logits: torch.Tensor,   # [B, L, V]
        targets: torch.Tensor,  # [B, L]
    ) -> torch.Tensor:
        """Compute CE loss for a single loop step.

        Returns:
            Scalar CE loss (averaged over non-ignored tokens).
        """
        # Shift: logits[t] predicts targets[t+1]
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = targets[:, 1:].contiguous()

        if (shift_labels != self.ignore_index).sum() == 0:
            return shift_logits.sum() * 0.0

        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )

    def forward(
        self,
        logits_per_step: list[torch.Tensor],  # T_max × [B, L, V]
        lm_targets:      torch.Tensor,         # [B, L]
        num_pred:        torch.Tensor,         # [B]
        num_gt:          torch.Tensor,         # [B]
        is_numeric:      torch.Tensor,         # [B]
        return_components: bool = False,
    ) -> torch.Tensor:
        """Compute uniform-weighted LoopLM training loss.

        L = (1/T) · Σ_t CE^(t) + α · L_SmoothL1

        Returns:
            Scalar loss (or tuple with component dict if return_components=True).
        """
        T_max = len(logits_per_step)

        # --- 1. Per-step CE losses ---
        ce_per_step = []
        for t in range(T_max):
            ce_t = self._per_step_ce(logits_per_step[t], lm_targets)
            ce_per_step.append(ce_t)

        ce_stack = torch.stack(ce_per_step)  # [T_max]

        # --- 2. Uniform-weighted CE: (1/T) · Σ_t CE^(t) ---
        loss_ce = ce_stack.mean()

        # --- 3. SmoothL1 loss (Number Head) ---
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
                'ce_per_step': ce_stack.detach().cpu().tolist(),
            }
        return total
