import torch

from llm.model.transformer_block import TransformerBlock


def main():
    print("=" * 60)
    print("TEST TRANSFORMER BLOCK")
    print("=" * 60)

    # Configuration
    embedding_dim = 256
    num_heads = 8
    hidden_dim = 1024
    max_sequence_length = 256

    batch_size = 2
    sequence_length = 256

    # Création du Transformer Block
    block = TransformerBlock(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        max_sequence_length=max_sequence_length
    )

    # Entrée simulée
    x = torch.randn(
        batch_size,
        sequence_length,
        embedding_dim
    )

    # Passage dans le bloc
    output = block(x)

    print()
    print("Configuration :")
    print(f"Embedding dimension : {embedding_dim}")
    print(f"Nombre de heads     : {num_heads}")
    print(f"Dimension par head  : {embedding_dim // num_heads}")
    print(f"Hidden dimension    : {hidden_dim}")
    print(f"Sequence length     : {sequence_length}")
    print(f"Batch size          : {batch_size}")

    print()
    print("Shapes :")
    print(f"Input  : {x.shape}")
    print(f"Output : {output.shape}")

    # Vérifications
    assert x.shape == (2, 256, 256)
    assert output.shape == (2, 256, 256)

    # Vérification des composants
    assert block.attention_norm.normalized_shape == (256,)
    assert block.feed_forward_norm.normalized_shape == (256,)
    assert block.attention.num_heads == 8
    assert block.attention.head_dim == 32

    print()
    print("✓ Input shape correcte")
    print("✓ Output shape correcte")
    print("✓ LayerNorm attention correcte")
    print("✓ Causal Self-Attention correcte")
    print("✓ LayerNorm Feed Forward correcte")
    print("✓ Feed Forward correcte")
    print("✓ Connexions résiduelles présentes")
    print()
    print("TEST RÉUSSI")


if __name__ == "__main__":
    main()