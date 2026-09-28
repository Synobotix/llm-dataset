import torch
import torch.nn as nn

from llm.model.attention import CausalSelfAttention
from llm.model.feed_forward import FeedForward


class TransformerBlock(nn.Module):
    """
    Un bloc Transformer causal.

    Architecture :

        x
        │
        ├── LayerNorm
        │
        ├── Causal Self-Attention
        │
        └── Résiduel
              ↓
        LayerNorm
              ↓
        Feed Forward
              ↓
        Résiduel
              ↓
             sortie
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        num_heads: int = 8,
        hidden_dim: int = 1024,
        max_sequence_length: int = 256
    ):
        super().__init__()

        # Première normalisation
        self.attention_norm = nn.LayerNorm(
            embedding_dim
        )

        # Causal Self-Attention
        self.attention = CausalSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            max_sequence_length=max_sequence_length
        )

        # Deuxième normalisation
        self.feed_forward_norm = nn.LayerNorm(
            embedding_dim
        )

        # Feed Forward Network
        self.feed_forward = FeedForward(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:
                [batch, sequence_length, embedding_dim]

        Returns:
            [batch, sequence_length, embedding_dim]
        """

        # --------------------------------------------------------
        # 1. Causal Self-Attention + connexion résiduelle
        # --------------------------------------------------------

        x = x + self.attention(
            self.attention_norm(x)
        )

        # --------------------------------------------------------
        # 2. Feed Forward + connexion résiduelle
        # --------------------------------------------------------

        x = x + self.feed_forward(
            self.feed_forward_norm(x)
        )

        return x