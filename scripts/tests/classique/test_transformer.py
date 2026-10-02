import torch

from llm.data.dataloader import create_train_dataloader
from llm.tokenizer.tokenizer import get_vocab_size
from llm.model.transformer import CausalTransformer


EMBEDDING_DIM = 120
NUM_HEADS = 3
FFN_HIDDEN_DIM = 480
NUM_LAYERS = 4
MAX_SEQUENCE_LENGTH = 128


def main():
    print("=" * 60)
    print("TEST DU TRANSFORMER CAUSAL COMPLET")
    print("=" * 60)

    # Taille réelle du vocabulaire
    vocab_size = get_vocab_size()

    print("\nTaille réelle du vocabulaire :")
    print(vocab_size)

    # Dataset
    train_loader = create_train_dataloader()

    batch = next(iter(train_loader))

    input_ids = batch["input_ids"]

    print("\nInput IDs :")
    print(input_ids.shape)

    # Modèle
    model = CausalTransformer(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        ffn_hidden_dim=FFN_HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
    )

    # Forward pass
    logits = model(input_ids)

    print("\nLogits :")
    print(logits.shape)

    print("\n--- CONFIGURATION ---")
    print(f"Vocabulaire        : {vocab_size}")
    print(f"Embedding dimension: {EMBEDDING_DIM}")
    print(f"Nombre de heads    : {NUM_HEADS}")
    print(f"Head dimension     : {EMBEDDING_DIM // NUM_HEADS}")
    print(f"FFN hidden         : {FFN_HIDDEN_DIM}")
    print(f"Nombre de layers   : {NUM_LAYERS}")
    print(f"Sequence length    : {MAX_SEQUENCE_LENGTH}")

    print("\n--- PARAMÈTRES DU MODÈLE ---")

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"Paramètres totaux      : {total_parameters:,}")
    print(f"Paramètres entraînables: {trainable_parameters:,}")

    print("\n--- VÉRIFICATION ---")

    assert input_ids.shape == (
        2,
        MAX_SEQUENCE_LENGTH,
    )

    assert logits.shape == (
        2,
        MAX_SEQUENCE_LENGTH,
        vocab_size,
    )

    assert model.vocab_size == vocab_size
    assert model.embedding_dim == EMBEDDING_DIM
    assert model.num_heads == NUM_HEADS
    assert model.num_layers == NUM_LAYERS

    assert len(model.transformer_blocks) == NUM_LAYERS

    print("✓ Vocabulaire réel correctement chargé")
    print("✓ Token Embedding fonctionne")
    print("✓ Positional Encoding fonctionne")
    print("✓ 4 Transformer Blocks fonctionnent")
    print("✓ Final LayerNorm fonctionne")
    print("✓ Projection vers le vocabulaire fonctionne")
    print("✓ Shape des logits correcte")
    print("✓ Modèle Transformer causal fonctionnel")


if __name__ == "__main__":
    main()