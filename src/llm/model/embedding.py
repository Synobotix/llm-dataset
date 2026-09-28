import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Transforme les IDs des tokens en vecteurs.
    """

    def __init__(
        self,
        vocab_size: int = 16000,
        embedding_dim: int = 256
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids:
                Tensor de forme [batch, sequence_length]

        Returns:
            Tensor de forme [batch, sequence_length, embedding_dim]
        """

        return self.embedding(input_ids)