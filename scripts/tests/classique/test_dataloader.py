import torch

from llm.data.dataloader import (
    create_train_dataloader,
    create_validation_dataloader,
)


def main():

    print("=" * 60)
    print("TEST DU DATALOADER")
    print("=" * 60)

    train_loader = create_train_dataloader()

    validation_loader = (
        create_validation_dataloader()
    )

    print("\n--- TRAIN ---")

    print(
        f"Nombre de batches : "
        f"{len(train_loader)}"
    )

    print("\n--- VALIDATION ---")

    print(
        f"Nombre de batches : "
        f"{len(validation_loader)}"
    )

    print("\n--- PREMIER BATCH TRAIN ---")

    batch = next(iter(train_loader))

    input_ids = batch["input_ids"]
    labels = batch["labels"]

    print(
        "\ninput_ids :"
    )

    print(input_ids)

    print(
        "\nlabels :"
    )

    print(labels)

    print(
        "\nType input_ids :"
    )

    print(input_ids.dtype)

    print(
        "\nType labels :"
    )

    print(labels.dtype)

    print(
        "\nShape input_ids :"
    )

    print(input_ids.shape)

    print(
        "\nShape labels :"
    )

    print(labels.shape)

    print(
        "\nDevice :"
    )

    print(input_ids.device)

    print(
        "\n--- VÉRIFICATION ---"
    )

    assert isinstance(
        input_ids,
        torch.Tensor
    )

    assert isinstance(
        labels,
        torch.Tensor
    )

    assert input_ids.dtype == torch.long

    assert labels.dtype == torch.long

    assert input_ids.shape == labels.shape

    assert input_ids.shape[0] == 2

    assert input_ids.shape[1] == 128

    print(
        "\n✓ Dataset correctement chargé"
    )

    print(
        "✓ Input et labels sont des tensors"
    )

    print(
        "✓ Les dimensions sont correctes"
    )

    print(
        "✓ DataLoader fonctionne"
    )


if __name__ == "__main__":
    main()