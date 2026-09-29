
import torch

from llm.data.dataloader import create_train_dataloader
from llm.model.transformer import CausalTransformer


def main():

    print("=" * 60)
    print("TEST TRAINER - UN BATCH")
    print("=" * 60)

    # ========================================================
    # Configuration
    # ========================================================

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    batch_size = 32

    print(f"Device : {device}")

    if torch.cuda.is_available():
        print(
            f"GPU    : "
            f"{torch.cuda.get_device_name(0)}"
        )

    # ========================================================
    # DataLoader
    # ========================================================

    dataloader = create_train_dataloader(
        dataset_path="data/processed/train_tokens.jsonl",
        batch_size=batch_size,
        num_workers=0,
    )

    batch = next(iter(dataloader))

    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    print()
    print("Batch :")
    print(f"  input_ids : {input_ids.shape}")
    print(f"  labels    : {labels.shape}")
    print(f"  dtype     : {input_ids.dtype}")

    # ========================================================
    # Modèle
    # ========================================================

    model = CausalTransformer(
        vocab_size=16000,
        embedding_dim=256,
        num_heads=8,
        num_layers=4,
        ff_hidden_dim=1024,
        max_sequence_length=256,
    )

    model.to(device)

    print()
    print("Modèle chargé.")

    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0003,
        weight_decay=0.01,
    )

    # ========================================================
    # Forward
    # ========================================================

    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    print()
    print("Forward...")

    logits = model(input_ids)

    print(
        f"Logits : {logits.shape}"
    )

    # ========================================================
    # Loss
    # ========================================================

    batch_size_actual = logits.size(0)
    sequence_length = logits.size(1)
    vocab_size = logits.size(2)

    logits_flat = logits.reshape(
        batch_size_actual * sequence_length,
        vocab_size
    )

    labels_flat = labels.reshape(
        batch_size_actual * sequence_length
    )

    loss = torch.nn.functional.cross_entropy(
        logits_flat,
        labels_flat
    )

    print(
        f"Loss initiale : {loss.item():.4f}"
    )

    # ========================================================
    # Backward
    # ========================================================

    print()
    print("Backward...")

    loss.backward()

    # ========================================================
    # Vérification des gradients
    # ========================================================

    total_gradient = 0.0
    gradient_count = 0

    for parameter in model.parameters():

        if parameter.grad is not None:

            total_gradient += (
                parameter.grad.detach()
                .abs()
                .mean()
                .item()
            )

            gradient_count += 1

    print(
        f"Paramètres avec gradient : "
        f"{gradient_count}"
    )

    print(
        f"Moyenne des gradients : "
        f"{total_gradient / max(gradient_count, 1):.8f}"
    )

    # ========================================================
    # Gradient clipping
    # ========================================================

    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
    )

    print(
        f"Gradient norm avant clipping : "
        f"{gradient_norm.item():.4f}"
    )

    # ========================================================
    # Optimizer step
    # ========================================================

    print()
    print("Optimizer step...")

    optimizer.step()

    print()
    print("=" * 60)
    print("TEST RÉUSSI")
    print("=" * 60)


if __name__ == "__main__":
    main()
