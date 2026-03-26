"""
SpatialVLM Training Losses
==========================

L_total = L_lm + lambda_fmt * L_fmt

    L_lm  : Standard autoregressive CrossEntropy over the full structured target.
    L_fmt : Binary format-consistency penalty (0 if type_ok, 1 otherwise).
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F


# Answer-type validators (mirrors pipeline._ANSWER_TYPE)
_ANSWER_TYPE = {
    "left_right": lambda a: a.strip('"') in ("left", "right"),
    "mcq":        lambda a: a.startswith('"') and a.strip('"').isdigit(),
    "distance":   lambda a: '.' in a and not a.startswith('"'),
    "count":      lambda a: a.isdigit(),
}

# Output parsing regex (same as pipeline.py)
_OUTPUT_RE = re.compile(
    r"CATEGORY:\s*(?P<category>\S+)\s*\|\s*"
    r"ANSWER:\s*(?P<answer>.+?)\s*\|\s*"
    r"FREE_ANSWER:\s*(?P<free_answer>.+)",
    re.IGNORECASE,
)


def check_format(decoded_text: str) -> bool:
    """Check if decoded model output follows the structured format
    AND has consistent CATEGORY <-> ANSWER types.

    Returns True if format is valid, False otherwise.
    """
    m = _OUTPUT_RE.search(decoded_text)
    if not m:
        return False
    category = m.group("category").strip().lower()
    answer = m.group("answer").strip()
    validator = _ANSWER_TYPE.get(category, lambda _: False)
    return validator(answer)


class SpatialVLMLoss(nn.Module):
    """Combined loss: L_total = L_lm + lambda_fmt * L_fmt

    Args:
        lambda_fmt:  Weight for format-consistency loss (default: 0.1)
        ignore_index: Token index to ignore in CE loss (default: -100)
    """

    def __init__(self, lambda_fmt: float = 0.1, ignore_index: int = -100):
        super().__init__()
        self.lambda_fmt = lambda_fmt
        self.ignore_index = ignore_index

    def lm_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Standard autoregressive language modeling loss.

        Args:
            logits: [B, T, vocab_size] -- model output logits
            labels: [B, T] -- target token ids, with -100 for ignored positions
                    (shifted internally: logits[:-1] predicts labels[1:])

        Returns:
            Scalar CE loss
        """
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

    def fmt_loss(
        self,
        decoded_outputs: list[str],
    ) -> torch.Tensor:
        """Binary format-consistency penalty.

        For each sample in the batch, checks if the decoded output:
        1. Matches the structured format: CATEGORY: ... | ANSWER: ... | FREE_ANSWER: ...
        2. Has consistent CATEGORY <-> ANSWER types (e.g. distance -> float)

        Args:
            decoded_outputs: list of decoded strings, one per batch element

        Returns:
            Scalar penalty in [0, 1] -- mean of binary penalties across batch.
            Not differentiable -- acts as a scaling regularizer on L_lm.
        """
        penalties = []
        for text in decoded_outputs:
            ok = check_format(text)
            penalties.append(0.0 if ok else 1.0)

        return torch.tensor(
            sum(penalties) / len(penalties),
            dtype=torch.float32,
        )

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        decoded_outputs: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute total loss.

        Args:
            logits:          [B, T, vocab_size] model output
            labels:          [B, T] target token ids (-100 for masked positions)
            decoded_outputs: list[str] decoded greedy outputs for L_fmt
                             (None to skip format loss -- useful during warmup)

        Returns:
            dict with keys:
                'loss':   L_total = L_lm + lambda_fmt * L_fmt  (for backward)
                'l_lm':   L_lm scalar
                'l_fmt':  L_fmt scalar (0 if skipped)
        """
        l_lm = self.lm_loss(logits, labels)

        if decoded_outputs is not None and self.lambda_fmt > 0:
            l_fmt = self.fmt_loss(decoded_outputs).to(l_lm.device)
            loss = l_lm + self.lambda_fmt * l_fmt
        else:
            l_fmt = torch.tensor(0.0, device=l_lm.device)
            loss = l_lm

        return {
            "loss":  loss,
            "l_lm":  l_lm.detach(),
            "l_fmt": l_fmt.detach(),
        }


# Build labels from input_ids
def build_labels(
    input_ids: torch.Tensor,
    answer_start_pos: int,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Build training labels from input_ids by masking everything before the answer.

    The model should only be trained to predict the answer portion
    (CATEGORY: ... | ANSWER: ... | FREE_ANSWER: ...), not the prompt.

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


if __name__ == "__main__":
    print("=" * 60)
    print("LOSS MODULE TEST")
    print("=" * 60)

    criterion = SpatialVLMLoss(lambda_fmt=0.2)

    # Simulate logits and labels (dummy inputs)
    B, T, V = 2, 50, 248320
    logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    labels[:, :20] = -100  # mask prompt

    # Good format
    decoded_good = [
        'CATEGORY: distance | ANSWER: 5.2 | FREE_ANSWER: The distance is 5.2 meters.',
        'CATEGORY: mcq | ANSWER: "2" | FREE_ANSWER: Option 2 is closest.',
    ]
    result = criterion(logits, labels, decoded_good)
    print(f"\n  Good format:")
    print(f"    L_lm    = {result['l_lm'].item():.4f}")
    print(f"    L_fmt   = {result['l_fmt'].item():.4f}  (expect 0.0)")
    print(f"    L_total = {result['loss'].item():.4f}")

    # Bad format
    decoded_bad = [
        'CATEGORY: left_right | ANSWER: left',  # missing FREE_ANSWER
        'CATEGORY: distance | ANSWER: "five" | FREE_ANSWER: wrong type',
    ]
    result = criterion(logits, labels, decoded_bad)
    print(f"\n  Bad format:")
    print(f"    L_lm    = {result['l_lm'].item():.4f}")
    print(f"    L_fmt   = {result['l_fmt'].item():.4f}  (expect 1.0)")
    print(f"    L_total = {result['loss'].item():.4f}")

    # Mixed format
    decoded_mixed = [
        'CATEGORY: count | ANSWER: 3 | FREE_ANSWER: There are 3.',
        'CATEGORY: mcq | ANSWER: five | FREE_ANSWER: wrong type',
    ]
    result = criterion(logits, labels, decoded_mixed)
    print(f"\n  Mixed format (1 good, 1 bad):")
    print(f"    L_lm    = {result['l_lm'].item():.4f}")
    print(f"    L_fmt   = {result['l_fmt'].item():.4f}  (expect 0.5)")
    print(f"    L_total = {result['loss'].item():.4f}")

    # No format loss
    result = criterion(logits, labels)
    print(f"\n  No format loss (skipped):")
    print(f"    L_lm    = {result['l_lm'].item():.4f}")
    print(f"    L_fmt   = {result['l_fmt'].item():.4f}  (expect 0.0)")
    print(f"    L_total = {result['loss'].item():.4f}")

    # build_labels test
    ids = torch.arange(100).unsqueeze(0)  # [1, 100]
    lbl = build_labels(ids, answer_start_pos=70)
    n_masked = (lbl == -100).sum().item()
    n_active = (lbl != -100).sum().item()
    print(f"\n  build_labels: {n_masked} masked, {n_active} active (expect 70/30)")

    print(f"\n{'='*60}")
    print("  Loss module [OK]")
    print(f"{'='*60}")