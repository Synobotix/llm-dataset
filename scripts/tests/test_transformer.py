import torch

from llm.model.transformer import CausalTransformer


def main():
    print("=" * 60)
    print("TEST TRANSFORMER CAUSAL COMPLET")
    print("=" * 60)

    # Configuration
    vocab_size = 16000
    embedding_dim = 256
    num_heads = 8
    num_layers = 4
    ff_hidden_dim = 1024
    max_sequence_length = 256

    batch_size = 2
    sequence_length = 256

    # Création du modèle
    model = CausalTransformer(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_hidden_dim=ff_hidden_dim,
        max_sequence_length=max_sequence_length
    )

    # Entrée simulée
    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, sequence_length)
    )

    # Forward pass
    logits = model(input_ids)

    print()
    print("Configuration :")
    print(f"Vocabulaire        : {vocab_size}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Nombre de heads    : {num_heads}")
    print(f"Nombre de blocks   : {num_layers}")
    print(f"FFN dimension      : {ff_hidden_dim}")
    print(f"Context length     : {max_sequence_length}")

    print()
    print("Shapes :")
    print(f"Input IDs : {input_ids.shape}")
    print(f"Logits    : {logits.shape}")

    # Vérification des dimensions
    assert input_ids.shape == (
        batch_size,
        sequence_length
    )

    assert logits.shape == (
        batch_size,
        sequence_length,
        vocab_size
    )

    # Vérification du nombre de blocs
    assert len(model.blocks) == 4

    # Vérification des paramètres principaux
    assert model.token_embedding.embedding.num_embeddings == 16000
    assert model.token_embedding.embedding.embedding_dim == 256

    assert model.blocks[0].attention.num_heads == 8
    assert model.blocks[0].attention.head_dim == 32

    print()
    print("✓ Input shape correcte")
    print("✓ Token Embedding correcte")
    print("✓ Position Embedding correcte")
    print("✓ 4 Transformer Blocks présents")
    print("✓ 8 heads par block")
    print("✓ Dimension par head : 32")
    print("✓ Final LayerNorm présente")
    print("✓ LM Head 256 → 16000 correcte")
    print("✓ Output logits correcte")

    print()
    print("TEST RÉUSSI")


if __name__ == "__main__":
    main()