"""
MODULE: Number Head — xVal-style Regression for Distance & Count

Position: After Decoder, parallel to LM Head
Input:  hidden states at <|num|> token positions [B_num, 1024]
Output: continuous scalar predictions [B_num] (non-negative)

Used for:
    - distance tasks: predicts distance in meters (float)
    - count tasks:    predicts count (float, rounded to int at inference)

Both tasks use SmoothL1 loss during training, which provides
bounded gradients while still optimizing for the RMSE eval metric.

Params: ~658K
    LayerNorm(1024):     2 × 1024 =    2,048
    Linear(1024, 512):   1024 × 512 = 524,288 + 512 = 524,800
    Linear(512, 256):    512 × 256 = 131,072 + 256 = 131,328
    Linear(256, 1):      256 × 1 =    256 + 1 =       257
    Total:               ~658,433 ≈ 0.66M
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NumberHead(nn.Module):
    """Predicts continuous values from decoder hidden states at <|num|> positions.

    Bypasses tokenization entirely — no digit tokens, no CE loss for numbers.
    Instead, reads the hidden state at the <|num|> token and regresses a scalar.

    Architecture:
        LayerNorm(1024) -> Linear(1024, 512) -> GELU -> Linear(512, 256) -> GELU -> Linear(256, 1) -> softplus()
    """

    def __init__(self, hidden_dim: int = 1024, int1_dim: int = 512, int2_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, int1_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int1_dim, int2_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int2_dim, 1),
        )

    def forward(self, h_num: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_num: [B_num, 1024] — hidden states at <|num|> token positions.
                   B_num = number of numeric samples in the batch (distance + count).
                   May be 0 if no numeric samples in batch.

        Returns:
            [B_num] — predicted values (non-negative via .softplus())
                      distance: meters (e.g. 5.73)
                      count:    float to be rounded at inference (e.g. 3.0)
        """
        if h_num.shape[0] == 0:
            return torch.zeros(0, device=h_num.device, dtype=h_num.dtype)
        return F.softplus(self.head(h_num).squeeze(-1))
