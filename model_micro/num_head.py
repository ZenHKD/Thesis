"""
MODULE: Number Head — xVal-style Regression for Distance & Count

Position: After Decoder, parallel to LM Head
Input:  hidden states at [NUM] token positions [B_num, 1024]
Output: continuous scalar predictions [B_num] (non-negative)

Used for:
    - distance tasks: predicts distance in meters (float)
    - count tasks:    predicts count (float, rounded to int at inference)

Both tasks use MSE loss during training, which directly optimizes
the RMSE metric used in the benchmark.

Params: ~262K
    LayerNorm(1024):     2 × 1024 =    2,048
    Linear(1024, 256):   1024 × 256 = 262,144 + 256 = 262,400
    Linear(256, 1):      256 × 1 =    256 + 1 =       257
    Total:               ~264,705 ≈ 0.26M
"""

import torch
import torch.nn as nn


class NumberHead(nn.Module):
    """Predicts continuous values from decoder hidden states at [NUM] positions.

    Bypasses tokenization entirely — no digit tokens, no CE loss for numbers.
    Instead, reads the hidden state at the [NUM] token and regresses a scalar.

    Architecture:
        LayerNorm(1024) → Linear(1024, 256) → GELU → Linear(256, 1) → .abs()
    """

    def __init__(self, hidden_dim: int = 1024, intermediate_dim: int = 256):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, 1),
        )

    def forward(self, h_num: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_num: [B_num, 1024] — hidden states at [NUM] token positions.
                   B_num = number of numeric samples in the batch (distance + count).
                   May be 0 if no numeric samples in batch.

        Returns:
            [B_num] — predicted values (non-negative via .abs())
                      distance: meters (e.g. 5.73)
                      count:    float to be rounded at inference (e.g. 3.0)
        """
        if h_num.shape[0] == 0:
            return torch.zeros(0, device=h_num.device, dtype=h_num.dtype)
        return self.head(h_num).squeeze(-1).abs()
