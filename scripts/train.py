import random

import numpy as np
import torch

from llm.data.dataloader import (
    create_train_dataloader,
    create_validation_dataloader,
)
from llm.model.transformer import Transformer
from llm.training.trainer import Trainer
from llm.training.checkpoint import load_checkpoint


# ============================================================
# CONFIGURATION
# ============================================================

VOCAB_SIZE = 994

D_MODEL = 120
NUM_HEADS = 3
NUM_LAYERS = 2
FFN_HIDDEN_DIM = 480

MAX_SEQ_LEN = 128

EPOCHS = 20

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01

DEVICE = "cpu"

SEED = 42

GRADIENT_CLIP = 1.0
SAVE_EVERY = 50


# ============================================================
# CHECKPOINT
# ============================================================

CHECKPOINT_TO_LOAD = (
    "checkpoints/student_v1/student_v1_25docs.pt"
)

CHECKPOINT_TO_SAVE = (
    "student_v1_25docs2.pt"
)


# ============================================================
# SEED
# ============================================================

def set_seed(seed: int) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# NOMBRE DE PARAMÈTRES
# ============================================================

def count_parameters(model):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(SEED)

    print("=" * 60)
    print("REPRISE DE L'ENTRAÎNEMENT STUDENT V1")
    print("=" * 60)

    print(f"Device       : {DEVICE}")
    print(f"PyTorch      : {torch.__version__}")
    print()

    # ========================================================
    # CONFIGURATION
    # ========================================================

    print("Configuration du modèle :")

    print(f"  vocab_size          : {VOCAB_SIZE}")
    print(f"  d_model             : {D_MODEL}")
    print(f"  num_heads           : {NUM_HEADS}")
    print(f"  num_layers          : {NUM_LAYERS}")
    print(f"  ffn_hidden_dim      : {FFN_HIDDEN_DIM}")
    print(f"  max_sequence_length : {MAX_SEQ_LEN}")

    print()

    # ========================================================
    # DATALOADERS
    # ========================================================

    print("Création des DataLoaders...")

    train_dataloader = create_train_dataloader()

    validation_dataloader = create_validation_dataloader()

    print(
        f"Train samples       : "
        f"{len(train_dataloader.dataset)}"
    )

    print(
        f"Validation samples  : "
        f"{len(validation_dataloader.dataset)}"
    )

    print(
        f"Batch size          : "
        f"{train_dataloader.batch_size}"
    )

    print(
        f"Batches par époque  : "
        f"{len(train_dataloader)}"
    )

    print()

    # ========================================================
    # MODÈLE
    # ========================================================

    print("Création du Transformer...")

    model = Transformer(
        vocab_size=VOCAB_SIZE,
        embedding_dim=D_MODEL,
        num_heads=NUM_HEADS,
        ffn_hidden_dim=FFN_HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        max_sequence_length=MAX_SEQ_LEN,
    )

    print("Modèle créé.")

    print(
        f"Nombre de paramètres : "
        f"{count_parameters(model):,}"
    )

    print()

    # ========================================================
    # VÉRIFICATION RÉELLE
    # ========================================================

    print("Vérification du modèle :")

    print(
        f"  model.vocab_size       : "
        f"{model.vocab_size}"
    )

    print(
        f"  model.embedding_dim    : "
        f"{model.embedding_dim}"
    )

    print(
        f"  model.num_heads       : "
        f"{model.num_heads}"
    )

    print(
        f"  model.ffn_hidden_dim  : "
        f"{model.ffn_hidden_dim}"
    )

    print(
        f"  model.num_layers      : "
        f"{model.num_layers}"
    )

    print(
        f"  model.max_sequence_length : "
        f"{model.max_sequence_length}"
    )

    print()

    # ========================================================
    # VÉRIFICATION DE L'EMBEDDING
    # ========================================================

    print("Vérification de la matrice d'embedding :")

    print(
        "  Embedding shape :",
        model.token_embedding.embedding.weight.shape
    )

    print()

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    print("Optimizer : AdamW")
    print(f"Learning rate : {LEARNING_RATE}")
    print(f"Weight decay  : {WEIGHT_DECAY}")

    print()

    # ========================================================
    # CHARGEMENT DU CHECKPOINT
    # ========================================================

    print("=" * 60)
    print("CHARGEMENT DU CHECKPOINT")
    print("=" * 60)

    global_step = load_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        checkpoint_path=CHECKPOINT_TO_LOAD,
        device=DEVICE,
    )

    print()

    # ========================================================
    # SCHEDULER
    # ========================================================

    total_training_steps = (
        EPOCHS * len(train_dataloader)
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_training_steps,
    )

    # Restaurer la position du scheduler
    # à partir du global_step du checkpoint.
    scheduler.last_epoch = global_step

    print("Scheduler : CosineAnnealingLR")

    print(
        f"Total training steps : "
        f"{total_training_steps}"
    )

    print(
        f"Global step restauré : "
        f"{global_step}"
    )

    print(
        f"Scheduler last_epoch : "
        f"{scheduler.last_epoch}"
    )

    print(
        f"Learning rate actuel : "
        f"{optimizer.param_groups[0]['lr']}"
    )

    print()

    # ========================================================
    # TRAINER
    # ========================================================

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        epochs=EPOCHS,
        gradient_clip=GRADIENT_CLIP,
        save_every=SAVE_EVERY,
        checkpoint_dir="checkpoints/student_v1",
    )

    # Restaurer le compteur global
    trainer.global_step = global_step

    print("Trainer créé.")

    print(
        f"Global step : "
        f"{trainer.global_step}"
    )

    print()

    # ========================================================
    # ENTRAÎNEMENT
    # ========================================================

    trainer.train()

    # ========================================================
    # SAUVEGARDE DU NOUVEAU CHECKPOINT
    # ========================================================

    trainer.save_checkpoint(
        filename=CHECKPOINT_TO_SAVE
    )

    print()

    print("=" * 60)
    print("NOUVEAU CHECKPOINT")
    print("=" * 60)

    print(
        f"Sauvegardé dans : "
        f"checkpoints/student_v1/{CHECKPOINT_TO_SAVE}"
    )


# ============================================================
# POINT D'ENTRÉE
# ============================================================

if __name__ == "__main__":
    main()