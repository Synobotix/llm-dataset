
import torch

from llm.data.dataloader import create_train_dataloader
from llm.tokenizer.tokenizer import get_vocab_size
from llm.model.transformer import CausalTransformer
from llm.training.optimizer import create_optimizer
from llm.training.scheduler import create_scheduler
from llm.training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
)


EMBEDDING_DIM = 120
NUM_HEADS = 3
FFN_HIDDEN_DIM = 480
NUM_LAYERS = 4
MAX_SEQUENCE_LENGTH = 128

LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.01

CHECKPOINT_PATH = (
    "checkpoints/student_v1/test_checkpoint.pt"
)


def create_model():
    vocab_size = get_vocab_size()

    return CausalTransformer(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        ffn_hidden_dim=FFN_HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
    )


def main():

    print("=" * 60)
    print("TEST DU CHECKPOINT")
    print("=" * 60)

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    train_dataloader = create_train_dataloader()

    print("\nDataset :")
    print(
        f"Train : "
        f"{len(train_dataloader.dataset)} séquences"
    )

    # --------------------------------------------------
    # Premier modèle
    # --------------------------------------------------

    print("\nCréation du premier modèle...")

    model = create_model()

    optimizer = create_optimizer(
        model=model,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    total_steps = len(train_dataloader)

    scheduler = create_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
    )

    # --------------------------------------------------
    # Simulation d'une étape d'entraînement
    # --------------------------------------------------

    print("\nSimulation d'une étape...")

    batch = next(iter(train_dataloader))

    input_ids = batch["input_ids"]
    labels = batch["labels"]

    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    logits = model(input_ids)

    loss_function = torch.nn.CrossEntropyLoss()

    batch_size = logits.size(0)
    sequence_length = logits.size(1)
    vocab_size = logits.size(2)

    loss = loss_function(
        logits.view(
            batch_size * sequence_length,
            vocab_size,
        ),
        labels.view(
            batch_size * sequence_length,
        ),
    )

    loss.backward()

    optimizer.step()

    scheduler.step()

    global_step = 1

    print(
        f"Loss : {loss.item():.4f}"
    )

    print(
        f"Learning rate : "
        f"{optimizer.param_groups[0]['lr']:.8f}"
    )

    print(
        f"Global step : {global_step}"
    )

    # --------------------------------------------------
    # Sauvegarde
    # --------------------------------------------------

    print("\nSauvegarde du checkpoint...")

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        global_step=global_step,
        checkpoint_path=CHECKPOINT_PATH,
    )

    # --------------------------------------------------
    # Vérification du fichier
    # --------------------------------------------------

    checkpoint_path = torch.path if False else None

    from pathlib import Path

    checkpoint_file = Path(
        CHECKPOINT_PATH
    )

    assert checkpoint_file.exists()

    print(
        "✓ Fichier checkpoint créé"
    )

    # --------------------------------------------------
    # Sauvegarde des paramètres
    # --------------------------------------------------

    original_model_state = {
        key: value.clone()
        for key, value in model.state_dict().items()
    }

    original_learning_rate = (
        optimizer.param_groups[0]["lr"]
    )

    original_global_step = global_step

    # --------------------------------------------------
    # Nouveau modèle
    # --------------------------------------------------

    print(
        "\nCréation d'un nouveau modèle..."
    )

    restored_model = create_model()

    restored_optimizer = create_optimizer(
        model=restored_model,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    restored_scheduler = create_scheduler(
        optimizer=restored_optimizer,
        total_steps=total_steps,
    )

    # --------------------------------------------------
    # Chargement
    # --------------------------------------------------

    print(
        "\nChargement du checkpoint..."
    )

    restored_global_step = load_checkpoint(
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        checkpoint_path=CHECKPOINT_PATH,
        device="cpu",
    )

    # --------------------------------------------------
    # Vérification du modèle
    # --------------------------------------------------

    print(
        "\nVérification des paramètres du modèle..."
    )

    for key, original_value in original_model_state.items():

        restored_value = (
            restored_model.state_dict()[key]
        )

        assert torch.equal(
            original_value,
            restored_value,
        ), f"Paramètre différent : {key}"

    print(
        "✓ Paramètres du modèle restaurés"
    )

    # --------------------------------------------------
    # Vérification du learning rate
    # --------------------------------------------------

    restored_learning_rate = (
        restored_optimizer.param_groups[0]["lr"]
    )

    assert (
        original_learning_rate
        == restored_learning_rate
    )

    print(
        "✓ Learning rate restauré"
    )

    # --------------------------------------------------
    # Vérification du global step
    # --------------------------------------------------

    assert (
        original_global_step
        == restored_global_step
    )

    print(
        "✓ Global step restauré"
    )

    # --------------------------------------------------
    # Vérification du scheduler
    # --------------------------------------------------

    assert (
        restored_scheduler.last_epoch
        == scheduler.last_epoch
    )

    print(
        "✓ État du scheduler restauré"
    )

    # --------------------------------------------------
    # Résultat
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("✓ TEST DU CHECKPOINT RÉUSSI")
    print("=" * 60)


if __name__ == "__main__":
    main()
