import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    Réseau Feed Forward utilisé dans un Transformer Block.

    Architecture :

        256
         ↓
        1024
         ↓
        GELU
         ↓
        256
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dim: int = 1024
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                embedding_dim,
                hidden_dim
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                embedding_dim
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:
                Tensor de forme
                [batch, sequence_length, embedding_dim]

        Returns:
            Tensor de forme
            [batch, sequence_length, embedding_dim]
        """

        return self.network(x)