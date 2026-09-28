import torch

from llm.model.embedding import TokenEmbedding


def main():
    print("=" * 60)
    print("TEST TOKEN EMBEDDING")
    print("=" * 60)

    # Configuration du test
    vocab_size = 16000
    embedding_dim = 256
    batch_size = 2
    sequence_length = 256

    # Création du modèle
    embedding = TokenEmbedding(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim
    )

    # Création de faux input IDs
    input_ids = torch.randint(
        0,
        vocab_size,
        (batch_size, sequence_length)
    )

    # Passage dans l'embedding
    output = embedding(input_ids)

    # Affichage
    print()
    print("Configuration :")
    print(f"Vocabulaire        : {vocab_size}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Batch size         : {batch_size}")
    print(f"Sequence length    : {sequence_length}")

    print()
    print("Shapes :")
    print(f"Input  : {input_ids.shape}")
    print(f"Output : {output.shape}")

    # Vérifications
    assert input_ids.shape == (2, 256)
    assert output.shape == (2, 256, 256)

    print()
    print("✓ Input shape correcte")
    print("✓ Output shape correcte")
    print()
    print("TEST RÉUSSI")


if __name__ == "__main__":
    main()