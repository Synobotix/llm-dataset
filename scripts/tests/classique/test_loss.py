import torch

from llm.data.dataloader import create_train_dataloader
from llm.tokenizer.tokenizer import get_vocab_size
from llm.model.transformer import CausalTransformer
from llm.training.loss import CausalLanguageModelingLoss


EMBEDDING_DIM = 120
NUM_HEADS = 3
FFN_HIDDEN_DIM = 480
NUM_LAYERS = 4
MAX_SEQUENCE_LENGTH = 128


def main():
    print("=" * 60)
    print("TEST DE LA CAUSAL LANGUAGE MODELING LOSS")
    print("=" * 60)

    # Taille réelle du vocabulaire
    vocab_size = get_vocab_size()

    print("\nTaille du vocabulaire :")
    print(vocab_size)

    # Dataset
    train_loader = create_train_dataloader()

    batch = next(iter(train_loader))

    input_ids = batch["input_ids"]
    labels = batch["labels"]

    print("\nInput IDs :")
    print(input_ids.shape)

    print("\nLabels :")
    print(labels.shape)

    # Modèle
    model = CausalTransformer(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        ffn_hidden_dim=FFN_HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
    )

    # Forward
    logits = model(input_ids)

    print("\nLogits :")
    print(logits.shape)

    # Loss
    loss_function = CausalLanguageModelingLoss()

    loss = loss_function(
        logits,
        labels,
    )

    print("\nLoss :")
    print(loss)

    print("\nValeur numérique de la Loss :")
    print(loss.item())

    print("\n--- VÉRIFICATION ---")

    assert input_ids.shape == (
        2,
        MAX_SEQUENCE_LENGTH,
    )

    assert labels.shape == (
        2,
        MAX_SEQUENCE_LENGTH,
    )

    assert logits.shape == (
        2,
        MAX_SEQUENCE_LENGTH,
        vocab_size,
    )

    assert loss.dim() == 0

    assert torch.isfinite(loss)

    assert loss.item() > 0

    print("✓ Input IDs correctement reçus")
    print("✓ Labels correctement reçus")
    print("✓ Logits correctement générés")
    print("✓ Cross-Entropy Loss calculable")
    print("✓ Loss scalaire")
    print("✓ Loss finie")
    print("✓ Loss positive")
    print("✓ Test de Loss réussi")


if __name__ == "__main__":
    main()