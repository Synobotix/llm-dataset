import torch
import torch.nn as nn


def debug_tensor(name: str, tensor: torch.Tensor):
    """ print(f"\n--- {name} ---")
    print("shape :", tensor.shape)
    print("dtype :", tensor.dtype)
    print("min   :", tensor.min().item())
    print("max   :", tensor.max().item())
    print("mean  :", tensor.mean().item())
    print("std   :", tensor.std().item())
    print("NaN   :", torch.isnan(tensor).any().item())
    print("Inf   :", torch.isinf(tensor).any().item()) """


class FeedForward(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                embedding_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                embedding_dim,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        print("\n" + "=" * 60)
        print("FEED FORWARD - FORWARD")
        print("=" * 60)

        debug_tensor("FFN input", x)

        x = self.network[0](x)
        debug_tensor("Après Linear 1", x)

        x = self.network[1](x)
        debug_tensor("Après GELU", x)

        x = self.network[2](x)
        debug_tensor("Après Linear 2", x)

        return x