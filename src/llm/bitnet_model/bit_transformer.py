import torch
import torch.nn as nn

from llm.bitnet_model.bit_transformer_block import BitTransformerBlock


class BitTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        hidden_dim: int,
        num_blocks: int,
        max_sequence_length: int,
        bias: bool = False,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.max_sequence_length = max_sequence_length

        # Embedding classique
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
        )

        # Empilement des blocs Transformer
        self.blocks = nn.ModuleList(
            [
                BitTransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    bias=bias,
                )
                for _ in range(num_blocks)
            ]
        )

        # Normalisation finale
        from llm.bitnet_model.bit_transformer_block import RMSNorm

        self.final_norm = RMSNorm(d_model)

    def forward(self, input_ids):
        batch_size, sequence_length = input_ids.shape

        if sequence_length > self.max_sequence_length:
            raise ValueError(
                f"La séquence ({sequence_length}) dépasse "
                f"max_sequence_length ({self.max_sequence_length})."
            )

        # Token IDs → vecteurs
        x = self.token_embedding(input_ids)

        # Passage dans tous les blocks
        for block in self.blocks:
            x = block(x)

        # Normalisation finale
        x = self.final_norm(x)

        return x