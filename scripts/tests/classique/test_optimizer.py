import torch

from llm.data.dataloader import create_train_dataloader
from llm.tokenizer.tokenizer import get_vocab_size
from llm.model.transformer import CausalTransformer
from llm.training.loss import CausalLanguageModelingLoss
from llm.training.optimizer import create_optimizer


EMBEDDING_DIM = 120
NUM_HEADS = 3
FFN_HIDDEN_DIM = 480
NUM_LAYERS = 4
MAX_SEQUENCE_LENGTH = 128

LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.01


def main():
    print("=" * 60)
    print("TEST DE L'OPTIMIZER ADAMW")
    print("=" * 60)

    vocab_size = get_vocab_size()

    train_loader = create_train_dataloader()

    batch = next(iter(train_loader))

    input_ids = batch["input_ids"]
    labels = batch["labels"]

    # Modèle
    model = CausalTransformer(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        ffn_hidden_dim=FFN_HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
    )

    # Loss
    loss_function = CausalLanguageModelingLoss()

    # Optimizer
    optimizer = create_optimizer(
        model=model,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------
    # Poids avant mise à jour
    # --------------------------------------------------

    parameter_name = "token_embedding.embedding.weight"

    parameter = dict(
        model.named_parameters()
    )[parameter_name]

    weights_before = parameter.detach().clone()

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------

    logits = model(input_ids)

    loss = loss_function(
        logits,
        labels,
    )

    print("\nLoss avant mise à jour :")
    print(loss.item())

    # --------------------------------------------------
    # Backpropagation
    # --------------------------------------------------

    optimizer.zero_grad()

    loss.backward()

    print("\nBackpropagation effectuée.")

    # --------------------------------------------------
    # Mise à jour des poids
    # --------------------------------------------------

    optimizer.step()

    print("Optimizer.step() effectué.")

    # --------------------------------------------------
    # Poids après mise à jour
    # --------------------------------------------------

    weights_after = parameter.detach().clone()

    difference = (
        weights_after - weights_before
    )

    number_of_changed_weights = (
        difference != 0
    ).sum().item()

    total_difference = (
        difference.abs().sum().item()
    )

    print("\n--- MODIFICATION DES POIDS ---")

    print(
        f"Poids modifiés : "
        f"{number_of_changed_weights:,}"
    )

    print(
        f"Différence absolue totale : "
        f"{total_difference}"
    )

    print("\n--- EXEMPLE ---")

    print(
        "Premier poids avant :",
        weights_before.flatten()[0].item(),
    )

    print(
        "Premier poids après :",
        weights_after.flatten()[0].item(),
    )

    print("\n--- CONFIGURATION ---")

    print(
        f"Learning rate : {LEARNING_RATE}"
    )

    print(
        f"Weight decay  : {WEIGHT_DECAY}"
    )

    print("\n--- VÉRIFICATION ---")

    assert loss.item() > 0

    assert number_of_changed_weights > 0

    assert total_difference > 0

    assert not torch.equal(
        weights_before,
        weights_after,
    )

    print("✓ AdamW correctement créé")
    print("✓ Forward effectué")
    print("✓ Loss calculée")
    print("✓ Backpropagation effectuée")
    print("✓ Optimizer.step() effectué")
    print("✓ Les poids ont été modifiés")
    print("✓ AdamW fonctionne")


if __name__ == "__main__":
    main()