import torch

from llm.model.positional_encoding import PositionalEncoding


def main():
    print("=" * 60)
    print("TEST POSITIONAL ENCODING")
    print("=" * 60)

    # Configuration
    max_sequence_length = 256
    embedding_dim = 256
    batch_size = 2
    sequence_length = 256

    # Création du module
    positional_encoding = PositionalEncoding(
        max_sequence_length=max_sequence_length,
        embedding_dim=embedding_dim
    )

    # Faux embeddings provenant du TokenEmbedding
    x = torch.randn(
        batch_size,
        sequence_length,
        embedding_dim
    )

    # Ajout des positions
    output = positional_encoding(x)

    print()
    print("Configuration :")
    print(f"Max sequence length : {max_sequence_length}")
    print(f"Embedding dimension : {embedding_dim}")
    print(f"Batch size          : {batch_size}")
    print(f"Sequence length     : {sequence_length}")

    print()
    print("Shapes :")
    print(f"Input  : {x.shape}")
    print(f"Output : {output.shape}")

    # Vérifications
    assert x.shape == (2, 256, 256)
    assert output.shape == (2, 256, 256)

    # Vérification que les valeurs ont changé
    assert not torch.equal(x, output)

    print()
    print("✓ Input shape correcte")
    print("✓ Output shape correcte")
    print("✓ Information de position ajoutée")
    print()
    print("TEST RÉUSSI")


if __name__ == "__main__":
    main()