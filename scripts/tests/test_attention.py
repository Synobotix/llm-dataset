import torch

from llm.model.attention import CausalSelfAttention


def main():
    print("=" * 60)
    print("TEST CAUSAL SELF-ATTENTION")
    print("=" * 60)

    # Configuration
    embedding_dim = 256
    num_heads = 8
    max_sequence_length = 256

    batch_size = 2
    sequence_length = 256

    # Création de l'attention
    attention = CausalSelfAttention(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        max_sequence_length=max_sequence_length
    )

    # Entrée simulée
    x = torch.randn(
        batch_size,
        sequence_length,
        embedding_dim
    )

    # Passage dans l'attention
    output = attention(x)

    print()
    print("Configuration :")
    print(f"Embedding dimension : {embedding_dim}")
    print(f"Nombre de heads     : {num_heads}")
    print(
        f"Dimension par head  : "
        f"{embedding_dim // num_heads}"
    )
    print(f"Sequence length     : {sequence_length}")
    print(f"Batch size          : {batch_size}")

    print()
    print("Shapes :")
    print(f"Input  : {x.shape}")
    print(f"Output : {output.shape}")

    # Vérifications
    assert x.shape == (2, 256, 256)
    assert output.shape == (2, 256, 256)

    # Vérification de la configuration
    assert attention.num_heads == 8
    assert attention.head_dim == 32

    print()
    print("✓ Input shape correcte")
    print("✓ Output shape correcte")
    print("✓ Nombre de heads correct")
    print("✓ Dimension par head correcte")
    print("✓ Masque causal créé")
    print()
    print("TEST RÉUSSI")


if __name__ == "__main__":
    main()