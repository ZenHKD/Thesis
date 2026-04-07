"""
SpatialVLM Micro — Combined Loss: CE (text) + MSE (numeric)
=============================================================

L = L_CE + α · L_MSE

L_CE:  Standard autoregressive CrossEntropy on structured text targets.
       Active on ALL samples (category tokens + answer tokens).
       Ignores prompt tokens (masked as -100).

L_MSE: Mean Squared Error on Number Head predictions.
       Active only for numeric samples (distance + count).
       Directly optimizes the RMSE benchmark metric.

The RTI injection changes sequence length (each <mask> becomes 2 tokens
instead of 3). Labels are front-trimmed to align with logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialLoss(nn.Module):
    """Combined CE (text) + MSE (numeric) loss for SpatialVLM Micro.

    Args:
        alpha:        Weight for MSE loss relative to CE loss.
        ignore_index: Token index to ignore in CE loss (default: -100).
        remap_fn:     Optional callable to remap label IDs (old → new vocab).
                      Pass pipeline.remap_to_new for pruned vocab models.
    """

    def __init__(self, alpha: float = 1.0, ignore_index: int = -100,
                 remap_fn=None):
        super().__init__()
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.remap_fn = remap_fn

    def forward(
        self,
        lm_logits:  torch.Tensor,   # [B, L, V] — LM Head output (V=319)
        lm_targets: torch.Tensor,   # [B, L']   — token targets (old IDs, -100 for prompt)
        num_pred:   torch.Tensor,   # [B]       — Number Head output
        num_gt:     torch.Tensor,   # [B]       — ground truth number (distance or count)
        is_numeric: torch.Tensor,   # [B]       — boolean mask for numeric samples
    ) -> torch.Tensor:
        """
        Returns:
            Scalar loss = L_CE + α · L_MSE
        """
        # --- Remap labels: old Qwen IDs → new pruned IDs [0..318] ---
        if self.remap_fn is not None:
            lm_targets = self.remap_fn(lm_targets)

        # --- Align labels with logits (RTI changes sequence length) ---
        if lm_targets.shape[1] > lm_logits.shape[1]:
            diff = lm_targets.shape[1] - lm_logits.shape[1]
            lm_targets = lm_targets[:, diff:]

        # --- CE Loss: shift logits[t] predicts targets[t+1] ---
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = lm_targets[:, 1:].contiguous()

        # Guard: return zero if all tokens are ignored (avoids NaN)
        if (shift_labels != self.ignore_index).sum() == 0:
            loss_ce = (lm_logits * 0.0).sum()
        else:
            loss_ce = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.ignore_index,
            )

        # --- MSE Loss: only on numeric samples (distance + count) ---
        if is_numeric.any():
            loss_mse = F.mse_loss(
                num_pred[is_numeric],
                num_gt[is_numeric],
            )
        else:
            loss_mse = torch.tensor(0.0, device=lm_logits.device)

        return loss_ce + self.alpha * loss_mse

