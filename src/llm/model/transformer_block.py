import torch
import torch.nn as nn

from llm.model.attention import MultiHeadCausalSelfAttention
from llm.model.feed_forward import FeedForward


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        ffn_hidden_dim: int,
        max_sequence_length: int,
    ):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embedding_dim)

        self.attention = MultiHeadCausalSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            max_sequence_length=max_sequence_length,
        )

        self.layer_norm_2 = nn.LayerNorm(embedding_dim)

        self.feed_forward = FeedForward(
            embedding_dim=embedding_dim,
            hidden_dim=ffn_hidden_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # 1. Attention + connexion résiduelle
        attention_input = self.layer_norm_1(x)

        attention_output = self.attention(
            attention_input
        )

        x = x + attention_output

        # 2. Feed Forward + connexion résiduelle
        feed_forward_input = self.layer_norm_2(x)

        feed_forward_output = self.feed_forward(
            feed_forward_input
        )

        x = x + feed_forward_output

        return x