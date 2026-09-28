import torch
import torch.nn as nn

from llm.model.embedding import TokenEmbedding
from llm.model.positional_encoding import PositionalEncoding
from llm.model.transformer_block import TransformerBlock


class CausalTransformer(nn.Module):
    """
    Transformer causal complet pour la prédiction du prochain token.

    Architecture :

        Token Embedding
              +
        Position Embedding
              ↓
        Transformer Block × 4
              ↓
        Final LayerNorm
              ↓
        Linear
              ↓
        Logits
    """

    def __init__(
        self,
        vocab_size: int = 16000,
        embedding_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_hidden_dim: int = 1024,
        max_sequence_length: int = 256
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_hidden_dim = ff_hidden_dim
        self.max_sequence_length = max_sequence_length

        # --------------------------------------------------------
        # 1. Token Embedding
        # --------------------------------------------------------

        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim
        )

        # --------------------------------------------------------
        # 2. Position Embedding
        # --------------------------------------------------------

        self.position_embedding = PositionalEncoding(
            max_sequence_length=max_sequence_length,
            embedding_dim=embedding_dim
        )

        # --------------------------------------------------------
        # 3. Transformer Blocks
        # --------------------------------------------------------

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                hidden_dim=ff_hidden_dim,
                max_sequence_length=max_sequence_length
            )
            for _ in range(num_layers)
        ])

        # --------------------------------------------------------
        # 4. Final LayerNorm
        # --------------------------------------------------------

        self.final_norm = nn.LayerNorm(
            embedding_dim
        )

        # --------------------------------------------------------
        # 5. Language Model Head
        # --------------------------------------------------------

        self.lm_head = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(
        self,
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            input_ids:
                Tensor [batch, sequence_length]

        Returns:
            logits:
                Tensor [batch, sequence_length, vocab_size]
        """

        # --------------------------------------------------------
        # Token embeddings
        # --------------------------------------------------------

        x = self.token_embedding(input_ids)

        # --------------------------------------------------------
        # Position embeddings
        # --------------------------------------------------------

        x = self.position_embedding(x)

        # --------------------------------------------------------
        # Transformer Blocks
        # --------------------------------------------------------

        for block in self.blocks:
            x = block(x)

        # --------------------------------------------------------
        # Final normalization
        # --------------------------------------------------------

        x = self.final_norm(x)

        # --------------------------------------------------------
        # Projection vers le vocabulaire
        # --------------------------------------------------------

        logits = self.lm_head(x)

        return logits