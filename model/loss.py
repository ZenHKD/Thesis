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

        Why L' < L and why trim from FRONT:
            RTI replaces each <mask> (3 subtokens) with 2 region tokens.
            All <mask> tokens are in the USER prompt, never in the answer.
            This shortens the PROMPT portion of the embedding sequence by 1 per mask.

            Original:  [prompt: L_p tokens | answer: L_a tokens]  (L = L_p + L_a)
            After RTI: [prompt: L_p-n tokens | answer: L_a tokens] (L' = L - n)

            The answer tokens shift LEFT by n positions in the embedding space.
            Trimming n tokens from the FRONT of the label tensor matches that
            shift -- so label[L_p - n] correctly aligns with logit[L_p - n - 1].

        Returns:
            Scalar CE loss
        """
        # Align: trim labels from the FRONT to match the RTI-shortened logits.
        # diff = n_masks (one per <mask> token in the prompt).
        if labels.shape[1] > logits.shape[1]:
            diff = labels.shape[1] - logits.shape[1]
            labels = labels[:, diff:]

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
