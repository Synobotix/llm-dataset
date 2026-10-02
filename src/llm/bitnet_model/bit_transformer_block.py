import torch
import torch.nn as nn

from llm.bitnet_model.bit_attention import BitAttention
from llm.bitnet_model.bit_mlp import BitMLP


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = torch.sqrt(
            torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps
        )

        x = x / rms

        return self.weight * x


class BitTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        hidden_dim: int,
        bias: bool = False,
    ):
        super().__init__()

        self.norm1 = RMSNorm(d_model)
        self.attention = BitAttention(
            d_model=d_model,
            num_heads=num_heads,
            bias=bias,
        )

        self.norm2 = RMSNorm(d_model)
        self.mlp = BitMLP(
            d_model=d_model,
            hidden_dim=hidden_dim,
            bias=bias,
        )

    def forward(self, x):
        # Attention + connexion résiduelle
        x = x + self.attention(self.norm1(x))

        # MLP + connexion résiduelle
        x = x + self.mlp(self.norm2(x))

        return x