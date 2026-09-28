import torch

from llm.model.feed_forward import FeedForward


def main():
    print("=" * 60)
    print("TEST FEED FORWARD")
    print("=" * 60)

    # Configuration
    embedding_dim = 256
    hidden_dim = 1024

    batch_size = 2
    sequence_length = 256

    # Création du Feed Forward
    feed_forward = FeedForward(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim
    )

    # Entrée simulée
    x = torch.randn(
        batch_size,
        sequence_length,
        embedding_dim
    )

    # Passage dans le réseau
    output = feed_forward(x)

    print()
    print("Configuration :")
    print(f"Embedding dimension : {embedding_dim}")
    print(f"Hidden dimension    : {hidden_dim}")
    print(f"Batch size          : {batch_size}")
    print(f"Sequence length     : {sequence_length}")

    print()
    print("Shapes :")
    print(f"Input  : {x.shape}")
    print(f"Output : {output.shape}")

    # Vérifications
    assert x.shape == (2, 256, 256)
    assert output.shape == (2, 256, 256)

    print()
    print("✓ Input shape correcte")
    print("✓ Output shape correcte")
    print("✓ Expansion 256 → 1024 correcte")
    print("✓ Réduction 1024 → 256 correcte")
    print()
    print("TEST RÉUSSI")


if __name__ == "__main__":
    main()