
import torch

from llm.data.dataloader import (
    create_train_dataloader,
    create_validation_dataloader,
)

from llm.tokenizer.tokenizer import get_vocab_size

from llm.model.transformer import CausalTransformer

from llm.training.optimizer import create_optimizer

from llm.training.scheduler import create_scheduler

from llm.training.trainer import Trainer


EMBEDDING_DIM = 120
NUM_HEADS = 3
FFN_HIDDEN_DIM = 480
NUM_LAYERS = 4
MAX_SEQUENCE_LENGTH = 128

LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.01

EPOCHS = 3


def main():

    print("=" * 60)
    print("TEST COMPLET DU TRAINER")
    print("=" * 60)

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    train_dataloader = create_train_dataloader()

    validation_dataloader = (
        create_validation_dataloader()
    )

    print("\nDataset :")

    print(
        f"Train      : "
        f"{len(train_dataloader.dataset)} séquences"
    )

    print(
        f"Validation : "
        f"{len(validation_dataloader.dataset)} séquences"
    )

    # --------------------------------------------------
    # Vocabulaire
    # --------------------------------------------------

    vocab_size = get_vocab_size()

    print(
        f"\nVocabulary size : {vocab_size}"
    )

    # --------------------------------------------------
    # Modèle
    # --------------------------------------------------

    model = CausalTransformer(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        ffn_hidden_dim=FFN_HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
    )

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    optimizer = create_optimizer(
        model=model,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------
    # Scheduler
    # --------------------------------------------------

    steps_per_epoch = len(
        train_dataloader
    )

    total_steps = (
        steps_per_epoch * EPOCHS
    )

    scheduler = create_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
    )

    print("\nScheduler :")

    print(
        f"Steps per epoch : "
        f"{steps_per_epoch}"
    )

    print(
        f"Total steps     : "
        f"{total_steps}"
    )

    print(
        f"Learning rate initial : "
        f"{optimizer.param_groups[0]['lr']:.8f}"
    )

    # --------------------------------------------------
    # Trainer
    # --------------------------------------------------

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device="cpu",
        epochs=EPOCHS,
        gradient_clip=1.0,
        eval_every=5,
        save_every=10,
        checkpoint_dir="checkpoints/student_v1",
    )

    # --------------------------------------------------
    # Entraînement
    # --------------------------------------------------

    trainer.train()

    # --------------------------------------------------
    # Vérification du global step
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("VÉRIFICATIONS")
    print("=" * 60)

    print(
        f"\nGlobal step : "
        f"{trainer.global_step}"
    )

    expected_steps = (
        steps_per_epoch * EPOCHS
    )

    assert (
        trainer.global_step
        == expected_steps
    )

    print(
        "✓ Toutes les étapes "
        "d'entraînement ont été exécutées"
    )

    # --------------------------------------------------
    # Vérification du learning rate
    # --------------------------------------------------

    final_learning_rate = (
        optimizer.param_groups[0]["lr"]
    )

    print(
        f"\nLearning rate final : "
        f"{final_learning_rate:.8f}"
    )

    assert (
        final_learning_rate
        <= LEARNING_RATE
    )

    print(
        "✓ Scheduler fonctionnel"
    )

    # --------------------------------------------------
    # Vérification du checkpoint final
    # --------------------------------------------------

    checkpoint_path = (
        trainer.checkpoint_dir
        / "student_v1_final.pt"
    )

    print(
        f"\nCheckpoint : "
        f"{checkpoint_path}"
    )

    assert checkpoint_path.exists()

    print(
        "✓ Checkpoint final créé"
    )

    # --------------------------------------------------
    # Chargement du checkpoint
    # --------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    # --------------------------------------------------
    # Vérification du contenu
    # --------------------------------------------------

    print(
        "\nContenu du checkpoint :"
    )

    for key in checkpoint:
        print(
            f"  → {key}"
        )

    assert (
        "global_step"
        in checkpoint
    )

    assert (
        "model_state_dict"
        in checkpoint
    )

    assert (
        "optimizer_state_dict"
        in checkpoint
    )

    assert (
        "scheduler_state_dict"
        in checkpoint
    )

    print(
        "\n✓ État du modèle présent"
    )

    print(
        "✓ État de l'optimizer présent"
    )

    print(
        "✓ État du scheduler présent"
    )

    print(
        "✓ Global step présent"
    )

    # --------------------------------------------------
    # Vérification du global step
    # --------------------------------------------------

    assert (
        checkpoint["global_step"]
        == trainer.global_step
    )

    print(
        "✓ Global step du checkpoint correct"
    )

    # --------------------------------------------------
    # Résultat
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("✓ TEST DU TRAINER RÉUSSI")
    print("=" * 60)


if __name__ == "__main__":
    main()
