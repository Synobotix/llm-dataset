import torch
import torch.nn as nn

from llm.model.embedding import TokenEmbedding
from llm.model.positional_encoding import PositionalEncoding
from llm.model.transformer_block import TransformerBlock


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_heads: int,
        ffn_hidden_dim: int,
        num_layers: int,
        max_sequence_length: int,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.ffn_hidden_dim = ffn_hidden_dim
        self.num_layers = num_layers
        self.max_sequence_length = max_sequence_length

        # 1. Token Embedding
        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
        )

        # 2. Positional Encoding
        self.positional_encoding = PositionalEncoding(
            embedding_dim=embedding_dim,
            max_sequence_length=max_sequence_length,
        )

        # 3. Transformer Blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    ffn_hidden_dim=ffn_hidden_dim,
                    max_sequence_length=max_sequence_length,
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Final LayerNorm
        self.final_layer_norm = nn.LayerNorm(
            embedding_dim
        )

        # 5. Projection vers le vocabulaire
        self.output_projection = nn.Linear(
            embedding_dim,
            vocab_size,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:

        # Token Embedding
        x = self.token_embedding(input_ids)

        # Positional Encoding
        x = self.positional_encoding(x)

        # Transformer Blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Final LayerNorm
        x = self.final_layer_norm(x)

        # Projection vers le vocabulaire
        logits = self.output_projection(x)

        return logits