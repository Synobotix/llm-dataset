import math

import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention causale.

    Configuration par défaut :
        embedding_dim = 256
        num_heads = 8
        max_sequence_length = 256
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        num_heads: int = 8,
        max_sequence_length: int = 256
    ):
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim doit être divisible par num_heads."
            )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # Projection simultanée vers Q, K et V
        self.qkv_projection = nn.Linear(
            embedding_dim,
            3 * embedding_dim
        )

        # Projection finale après concaténation des heads
        self.output_projection = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        # Masque causal.
        #
        # False = position autorisée
        # True  = position interdite
        #
        # Exemple :
        #
        # False True  True
        # False False True
        # False False False
        #
        causal_mask = torch.triu(
            torch.ones(
                max_sequence_length,
                max_sequence_length,
                dtype=torch.bool
            ),
            diagonal=1
        )

        self.register_buffer(
            "causal_mask",
            causal_mask,
            persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:
                [batch, sequence_length, embedding_dim]

        Returns:
            [batch, sequence_length, embedding_dim]
        """

        batch_size, sequence_length, _ = x.shape

        # --------------------------------------------------------
        # Q, K, V
        # --------------------------------------------------------

        qkv = self.qkv_projection(x)

        # [B, T, 3C]
        query, key, value = qkv.chunk(3, dim=-1)

        # --------------------------------------------------------
        # Découpage en plusieurs heads
        # --------------------------------------------------------

        query = query.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        )

        key = key.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        )

        value = value.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        )

        # [B, T, H, D]
        # →
        # [B, H, T, D]

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # --------------------------------------------------------
        # Scores d'attention
        # --------------------------------------------------------

        scores = torch.matmul(
            query,
            key.transpose(-2, -1)
        )

        scores = scores / math.sqrt(self.head_dim)

        # --------------------------------------------------------
        # Masque causal
        # --------------------------------------------------------

        mask = self.causal_mask[
            :sequence_length,
            :sequence_length
        ]

        scores = scores.masked_fill(
            mask,
            float("-inf")
        )

        # --------------------------------------------------------
        # Softmax
        # --------------------------------------------------------

        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        # --------------------------------------------------------
        # Attention × Value
        # --------------------------------------------------------

        attention_output = torch.matmul(
            attention_weights,
            value
        )

        # [B, H, T, D]
        #
        # →
        #
        # [B, T, H, D]

        attention_output = attention_output.transpose(1, 2)

        # Fusion des heads
        attention_output = attention_output.contiguous().view(
            batch_size,
            sequence_length,
            self.embedding_dim
        )

        # Projection finale
        output = self.output_projection(
            attention_output
        )

        return output