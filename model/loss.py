"""
SpatialVLM Training Loss
=========================

L = L_lm (standard autoregressive CrossEntropy over the structured target)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialVLMLoss(nn.Module):
    """Autoregressive LM loss for SpatialVLM.

    Args:
        ignore_index: Token index to ignore in CE loss (default: -100)
    """

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Standard autoregressive language modeling loss.

        Args:
            logits: [1, L', vocab_size] -- text-only logits from pipeline.forward()
            labels: [1, L]              -- -100 for prompt, token ids for answer

        L' may be shorter than L when RTI replaces each <mask> span (3 subtokens)
        with 2 region tokens (-1 per mask). Trimmed from the END because the
        shortening happens mid-sequence; trimming from the front would shift
        answer tokens off-alignment.

        Returns:
            Scalar CE loss
        """
        # Align lengths: trim labels from the end to match logits
        if logits.shape[1] < labels.shape[1]:
            labels = labels[:, :logits.shape[1]]

        # Shift: logits[t] predicts labels[t+1]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Guard: F.cross_entropy returns NaN when ALL tokens are ignored (0/0).
        # Return a zero scalar that still has a grad_fn so backward() is safe.
        if (shift_labels != self.ignore_index).sum() == 0:
            return (logits * 0.0).sum()

        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.ignore_index,
        )
