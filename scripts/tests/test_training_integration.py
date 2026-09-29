
from pathlib import Path

import numpy as np
import torch
import yaml

from llm.data.dataloader import create_train_dataloader
from llm.model.transformer import CausalTransformer
from llm.training.trainer import Trainer


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

MODEL_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "model.yaml"
)

TRAINING_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "training.yaml"
)

TRAIN_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train_tokens.jsonl"
)

TEST_CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "student_v1_test"
)


# ============================================================
# UTILITAIRES
# ============================================================

def load_yaml(path: Path) -> dict:

    with path.open(
        "r",
        encoding="utf-8"
    ) as file:
        return yaml.safe_load(file)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("TEST D'INTÉGRATION TRAINER")
    print("=" * 60)

    # ========================================================
    # CONFIGURATION
    # ========================================================

    model_config = load_yaml(
        MODEL_CONFIG_PATH
    )

    training_config = load_yaml(
        TRAINING_CONFIG_PATH
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device : {device}")

    # ========================================================
    # DATALOADER
    # ========================================================

    dataloader = create_train_dataloader(
        dataset_path=TRAIN_DATA_PATH,
        batch_size=training_config["batch_size"],
        num_workers=0,
    )

    print(
        f"Dataset : "
        f"{len(dataloader.dataset)} séquences"
    )

    print(
        f"Batches : "
        f"{len(dataloader)}"
    )

    # ========================================================
    # MODÈLE
    # ========================================================

    model = CausalTransformer(
        vocab_size=model_config["vocab_size"],
        embedding_dim=model_config["d_model"],
        num_heads=model_config["num_heads"],
        num_layers=model_config["num_layers"],
        ff_hidden_dim=model_config["d_ff"],
        max_sequence_length=model_config["max_seq_len"],
    )

    model.to(device)

    print("Modèle : OK")

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )

    # ========================================================
    # SCHEDULER
    # ========================================================

    total_steps = len(dataloader)

    warmup_steps = min(
        training_config.get("warmup_steps", 0),
        total_steps
    )

    def lr_lambda(current_step: int):

        if current_step < warmup_steps:

            if warmup_steps == 0:
                return 1.0

            return (
                float(current_step + 1)
                / float(warmup_steps)
            )

        progress = (
            current_step - warmup_steps
        ) / max(
            total_steps - warmup_steps,
            1
        )

        return 0.5 * (
            1.0
            + np.cos(
                np.pi * progress
            )
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda
    )

    print("Optimizer : OK")
    print("Scheduler : OK")

    # ========================================================
    # TRAINER
    # ========================================================

    trainer = Trainer(
        model=model,
        train_dataloader=dataloader,
        validation_dataloader=None,
        optimizer=optimizer,
        scheduler=scheduler,
        device=str(device),
        epochs=1,
        gradient_clip=training_config.get(
            "gradient_clip",
            1.0
        ),
        eval_every=1,
        save_every=1,
        checkpoint_dir=TEST_CHECKPOINT_DIR,
    )

    print("Trainer : OK")

    # ========================================================
    # TEST D'UN SEUL BATCH
    # ========================================================

    print()
    print("Exécution d'un seul batch...")

    trainer.model.train()

    batch = next(iter(dataloader))

    input_ids = batch["input_ids"].to(
        trainer.device
    )

    labels = batch["labels"].to(
        trainer.device
    )

    trainer.optimizer.zero_grad(
        set_to_none=True
    )

    logits = trainer.model(
        input_ids
    )

    loss = trainer._compute_loss(
        logits,
        labels
    )

    print(
        f"Loss avant backward : "
        f"{loss.item():.4f}"
    )

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        trainer.model.parameters(),
        trainer.gradient_clip
    )

    trainer.optimizer.step()
    trainer.scheduler.step()

    trainer.global_step += 1

    print(
        f"Global step : "
        f"{trainer.global_step}"
    )

    print(
        f"Learning rate : "
        f"{trainer.optimizer.param_groups[0]['lr']:.8f}"
    )

    # ========================================================
    # CHECKPOINT
    # ========================================================

    trainer.save_checkpoint(
        filename="integration_test.pt"
    )

    checkpoint_path = (
        TEST_CHECKPOINT_DIR
        / "integration_test.pt"
    )

    if not checkpoint_path.exists():
        raise RuntimeError(
            "Le checkpoint n'a pas été créé."
        )

    print(
        f"Checkpoint : "
        f"{checkpoint_path}"
    )

    # ========================================================
    # FIN
    # ========================================================

    print()
    print("=" * 60)
    print("TEST D'INTÉGRATION RÉUSSI")
    print("=" * 60)


if __name__ == "__main__":
    main()
