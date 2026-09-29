
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from llm.data.dataloader import (
    create_train_dataloader,
    create_validation_dataloader,
)
from llm.model.transformer import CausalTransformer
from llm.training.trainer import Trainer


# ============================================================
# CHEMINS DU PROJET
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

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

VALIDATION_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "validation_tokens.jsonl"
)


# ============================================================
# UTILITAIRES
# ============================================================

def load_yaml(path: Path) -> dict:
    """
    Charge un fichier YAML.
    """

    with path.open(
        "r",
        encoding="utf-8"
    ) as file:
        return yaml.safe_load(file)


def set_seed(seed: int) -> None:
    """
    Configure les seeds pour rendre
    l'entraînement reproductible.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(
    model: torch.nn.Module
) -> int:
    """
    Compte le nombre total de paramètres
    du modèle.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # CONFIGURATION
    # ========================================================

    model_config = load_yaml(
        MODEL_CONFIG_PATH
    )

    training_config = load_yaml(
        TRAINING_CONFIG_PATH
    )

    seed = training_config.get(
        "seed",
        42
    )

    set_seed(seed)

    # ========================================================
    # DEVICE
    # ========================================================

    configured_device = training_config.get(
        "device",
        "cuda"
    )

    if (
        configured_device == "cuda"
        and not torch.cuda.is_available()
    ):

        print(
            "⚠️ CUDA demandée mais indisponible."
        )

        print(
            "Utilisation du CPU."
        )

        device = "cpu"

    else:

        device = configured_device

    # ========================================================
    # INFORMATIONS ENVIRONNEMENT
    # ========================================================

    print("=" * 60)
    print("ENTRAÎNEMENT STUDENT V1")
    print("=" * 60)

    print(
        f"Device       : {device}"
    )

    print(
        f"PyTorch      : {torch.__version__}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU          : "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"CUDA         : "
            f"{torch.version.cuda}"
        )

    print()

    # ========================================================
    # CONFIGURATION DU MODÈLE
    # ========================================================

    vocab_size = model_config[
        "vocab_size"
    ]

    embedding_dim = model_config[
        "d_model"
    ]

    num_heads = model_config[
        "num_heads"
    ]

    num_layers = model_config[
        "num_layers"
    ]

    ff_hidden_dim = model_config[
        "d_ff"
    ]

    max_sequence_length = model_config[
        "max_seq_len"
    ]

    dropout = model_config.get(
        "dropout",
        0.1
    )

    print(
        "Configuration du modèle :"
    )

    print(
        f"  vocab_size          : "
        f"{vocab_size}"
    )

    print(
        f"  d_model             : "
        f"{embedding_dim}"
    )

    print(
        f"  num_heads           : "
        f"{num_heads}"
    )

    print(
        f"  num_layers          : "
        f"{num_layers}"
    )

    print(
        f"  d_ff                : "
        f"{ff_hidden_dim}"
    )

    print(
        f"  max_seq_len         : "
        f"{max_sequence_length}"
    )

    print(
        f"  dropout             : "
        f"{dropout}"
    )

    # ========================================================
    # CRÉATION DU MODÈLE
    # ========================================================

    model = CausalTransformer(

        vocab_size=vocab_size,

        embedding_dim=embedding_dim,

        num_heads=num_heads,

        num_layers=num_layers,

        ff_hidden_dim=ff_hidden_dim,

        max_sequence_length=max_sequence_length,
    )

    total_parameters = count_parameters(
        model
    )

    print()

    print(
        f"Nombre de paramètres : "
        f"{total_parameters:,}"
    )

    # ========================================================
    # DATASETS
    # ========================================================

    batch_size = training_config[
        "batch_size"
    ]

    print()

    print(
        "Chargement des datasets..."
    )

    train_dataloader = (
        create_train_dataloader(
            dataset_path=TRAIN_DATA_PATH,
            batch_size=batch_size,
            num_workers=0,
        )
    )

    validation_dataloader = (
        create_validation_dataloader(
            dataset_path=VALIDATION_DATA_PATH,
            batch_size=batch_size,
            num_workers=0,
        )
    )

    print(
        f"Train       : "
        f"{len(train_dataloader.dataset)} "
        f"séquences"
    )

    print(
        f"Validation  : "
        f"{len(validation_dataloader.dataset)} "
        f"séquences"
    )

    print(
        f"Batch size  : "
        f"{batch_size}"
    )

    print(
        f"Batches/epoch : "
        f"{len(train_dataloader)}"
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    learning_rate = training_config[
        "learning_rate"
    ]

    weight_decay = training_config[
        "weight_decay"
    ]

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=learning_rate,

        weight_decay=weight_decay,
    )

    # ========================================================
    # SCHEDULER
    # ========================================================

    epochs = training_config[
        "epochs"
    ]

    steps_per_epoch = len(
        train_dataloader
    )

    total_steps = (
        steps_per_epoch * epochs
    )

    scheduler_name = (
        training_config.get(
            "scheduler",
            "cosine"
        )
    )

    scheduler = None

    if scheduler_name == "cosine":

        warmup_steps = training_config.get(
            "warmup_steps",
            0
        )

        warmup_steps = min(
            warmup_steps,
            total_steps
        )

        def lr_lambda(
            current_step: int
        ):

            # ------------------------------
            # Warmup
            # ------------------------------

            if current_step < warmup_steps:

                if warmup_steps == 0:
                    return 1.0

                return (
                    float(current_step + 1)
                    / float(warmup_steps)
                )

            # ------------------------------
            # Cosine decay
            # ------------------------------

            progress = (
                current_step
                - warmup_steps
            ) / max(
                total_steps
                - warmup_steps,
                1
            )

            return 0.5 * (
                1.0
                + np.cos(
                    np.pi * progress
                )
            )

        scheduler = (
            torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda
            )
        )

    # ========================================================
    # CONFIGURATION ENTRAÎNEMENT
    # ========================================================

    print()

    print(
        "Configuration entraînement :"
    )

    print(
        f"  batch_size       : "
        f"{batch_size}"
    )

    print(
        f"  epochs           : "
        f"{epochs}"
    )

    print(
        f"  learning_rate    : "
        f"{learning_rate}"
    )

    print(
        f"  weight_decay     : "
        f"{weight_decay}"
    )

    print(
        f"  scheduler        : "
        f"{scheduler_name}"
    )

    print(
        f"  warmup_steps     : "
        f"{training_config.get('warmup_steps', 0)}"
    )

    print(
        f"  total_steps      : "
        f"{total_steps}"
    )

    print(
        f"  gradient_clip    : "
        f"{training_config.get('gradient_clip', 1.0)}"
    )

    # ========================================================
    # TRAINER
    # ========================================================

    trainer = Trainer(

        model=model,

        train_dataloader=train_dataloader,

        validation_dataloader=(
            validation_dataloader
        ),

        optimizer=optimizer,

        scheduler=scheduler,

        device=device,

        epochs=epochs,

        gradient_clip=(
            training_config.get(
                "gradient_clip",
                1.0
            )
        ),

        eval_every=(
            training_config.get(
                "eval_every",
                20
            )
        ),

        save_every=(
            training_config.get(
                "save_every",
                50
            )
        ),

        checkpoint_dir=(
            PROJECT_ROOT
            / training_config.get(
                "checkpoint_dir",
                "checkpoints/student_v1"
            )
        ),
    )

    # ========================================================
    # ENTRAÎNEMENT
    # ========================================================

    trainer.train()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()

