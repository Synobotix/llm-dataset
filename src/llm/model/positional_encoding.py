import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Ajoute une représentation de position aux embeddings des tokens.

    Les positions sont apprises par le modèle.
    """

    def __init__(
        self,
        max_sequence_length: int = 256,
        embedding_dim: int = 256
    ):
        super().__init__()

        self.position_embedding = nn.Embedding(
            num_embeddings=max_sequence_length,
            embedding_dim=embedding_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:
                Tensor de forme [batch, sequence_length, embedding_dim]

        Returns:
            Tensor de même forme avec l'information de position ajoutée.
        """

        batch_size, sequence_length, _ = x.shape

        # Création des positions : 0, 1, 2, ..., sequence_length - 1
        positions = torch.arange(
            sequence_length,
            device=x.device
        )

        # Embeddings des positions
        position_embeddings = self.position_embedding(positions)

        # Ajout des positions aux embeddings des tokens
        return x + position_embeddings