import torch
import torch.nn as nn


class FDA_Module(nn.Module):
    """
    Feature Denoising Adapter for VGGT.

    Input/Output shape:
        (B, S, D) -> (B, S, D)
    """

    def __init__(self, dim: int):
        super().__init__()
        hidden_mid = max(8, dim // 2)
        hidden_low = max(8, dim // 4)

        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_mid),
            nn.GELU(),
            nn.Linear(hidden_mid, hidden_mid),
            nn.GELU(),
            nn.Linear(hidden_mid, hidden_low),
            nn.GELU(),
            nn.Linear(hidden_low, dim),
        )

        # Keep identity behavior at init to avoid disrupting pretrained backbone.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))
