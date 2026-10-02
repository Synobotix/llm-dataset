import torch
import torch.nn as nn

from llm.data.dataloader import create_train_dataloader
from llm.bitnet_model.bit_transformer import BitTransformer
from llm.bitnet_model.lm_head import LMHead


# ============================================================
# CONFIGURATION
# ============================================================

VOCAB_SIZE = 994
D_MODEL = 256
NUM_HEADS = 4
HIDDEN_DIM = 680
NUM_BLOCKS = 2
MAX_SEQUENCE_LENGTH = 128

LEARNING_RATE = 1e-3
NUM_BATCHES = 5


# ============================================================
# TEST
# ============================================================

def main():

    print("=" * 60)
    print("TEST ENTRAÎNEMENT BITNET SUR DATASET RÉEL")
    print("=" * 60)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nDevice : {device}")

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    print("\nChargement du DataLoader...")

    dataloader = create_train_dataloader()

    print(
        f"Nombre de séquences : "
        f"{len(dataloader.dataset)}"
    )

    print(
        f"Nombre de batches : "
        f"{len(dataloader)}"
    )

    print(
        f"Batch size : "
        f"{dataloader.batch_size}"
    )

    # --------------------------------------------------------
    # Modèle
    # --------------------------------------------------------

    print("\nCréation du BitTransformer...")

    transformer = BitTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        bias=False,
    ).to(device)

    lm_head = LMHead(
        d_model=D_MODEL,
        vocab_size=VOCAB_SIZE,
        bias=False,
    ).to(device)

    print("✓ BitTransformer créé")
    print("✓ LM Head créée")

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        list(transformer.parameters())
        + list(lm_head.parameters()),
        lr=LEARNING_RATE,
    )

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Entraînement
    # --------------------------------------------------------

    transformer.train()
    lm_head.train()

    print("\nDébut de l'entraînement...")
    print("-" * 60)

    for batch_index, batch in enumerate(
        dataloader,
        start=1,
    ):

        if batch_index > NUM_BATCHES:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        hidden_states = transformer(
            input_ids
        )

        logits = lm_head(
            hidden_states
        )

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        batch_size, sequence_length, vocab_size = logits.shape

        logits_for_loss = logits.reshape(
            batch_size * sequence_length,
            vocab_size,
        )

        labels_for_loss = labels.reshape(
            batch_size * sequence_length
        )

        loss = criterion(
            logits_for_loss,
            labels_for_loss,
        )

        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        # ----------------------------------------------------
        # Vérification gradients
        # ----------------------------------------------------

        gradient_count = 0

        for parameter in (
            list(transformer.parameters())
            + list(lm_head.parameters())
        ):

            if parameter.grad is not None:

                gradient_count += 1

        # ----------------------------------------------------
        # Optimizer step
        # ----------------------------------------------------

        optimizer.step()

        print(
            f"Batch {batch_index:02d} | "
            f"Loss : {loss.item():.6f} | "
            f"Input : {tuple(input_ids.shape)} | "
            f"Logits : {tuple(logits.shape)} | "
            f"Gradients : {gradient_count}"
        )

    # --------------------------------------------------------
    # Résultat
    # --------------------------------------------------------

    print("-" * 60)

    print(
        "\n✓ Entraînement sur les données réelles terminé."
    )

    print(
        "✓ DataLoader réel utilisé."
    )

    print(
        "✓ Forward réussi."
    )

    print(
        "✓ Loss calculée."
    )

    print(
        "✓ Backpropagation réussie."
    )

    print(
        "✓ Optimizer.step() réussi."
    )

    print("=" * 60)


if __name__ == "__main__":
    main()