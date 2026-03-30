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
            logits: [B, L', vocab_size] -- model output logits (text tokens only)
            labels: [B, L]  -- target token ids, with -100 for ignored positions
                    L' may be shorter than L when RTI replaces <mask> (3 subtokens)
                    with 2 region tokens, shortening the sequence by 1 per mask.
                    (shifted internally: logits[:-1] predicts labels[1:])

        Returns:
            Scalar CE loss
        """
        # Align: RTI may shorten the text sequence (3 mask subtokens -> 2 region tokens)
        # Trim leading labels (all -100 prompt tokens) to match logits length
        if logits.shape[1] != labels.shape[1]:
            diff = labels.shape[1] - logits.shape[1]
            if diff > 0:
                labels = labels[:, diff:]

        # Shift: predict next token
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Flatten for CrossEntropy
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.ignore_index,
        )
        return loss


# Build labels from input_ids
def build_labels(
    input_ids: torch.Tensor,
    answer_start_pos: int,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Build training labels from input_ids by masking everything before the answer.

    The model should only be trained to predict the answer portion
    (category | value), not the prompt.

    Args:
        input_ids:        [B, T] full input sequence
        answer_start_pos: token position where the answer starts
                          (i.e. the first token after <|im_start|>assistant\\n)
        ignore_index:     value to mask non-target positions (default: -100)

    Returns:
        labels: [B, T] with ignore_index for prompt tokens, input_ids for answer tokens
    """
    labels = input_ids.clone()
    labels[:, :answer_start_pos] = ignore_index
    return labels
