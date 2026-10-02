import torch

from llm.bitnet_model.bit_mlp import BitMLP


def main():

    # --------------------------------------------------
    # Configuration
    # --------------------------------------------------

    batch_size = 2
    sequence_length = 8

    d_model = 256
    hidden_dim = 680

    # --------------------------------------------------
    # Création du BitMLP
    # --------------------------------------------------

    mlp = BitMLP(
        d_model=d_model,
        hidden_dim=hidden_dim,
        bias=False,
    )

    print("BitMLP :")
    print(mlp)

    # --------------------------------------------------
    # Entrée
    # --------------------------------------------------

    x = torch.randn(
        batch_size,
        sequence_length,
        d_model,
    )

    print("\nShape entrée :")
    print(x.shape)

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------

    output = mlp(x)

    print("\nShape sortie :")
    print(output.shape)

    # --------------------------------------------------
    # Vérification shape
    # --------------------------------------------------

    expected_shape = (
        batch_size,
        sequence_length,
        d_model,
    )

    assert output.shape == expected_shape

    print("\n✓ Forward OK")

    # --------------------------------------------------
    # Vérification BitLinear 1
    # --------------------------------------------------

    weight_1 = mlp.fc1.quantize_weights(
        mlp.fc1.weight
    )

    unique_values_1 = torch.unique(
        weight_1
    )

    print("\nValeurs uniques fc1 :")
    print(unique_values_1)

    # --------------------------------------------------
    # Vérification BitLinear 2
    # --------------------------------------------------

    weight_2 = mlp.fc2.quantize_weights(
        mlp.fc2.weight
    )

    unique_values_2 = torch.unique(
        weight_2
    )

    print("\nValeurs uniques fc2 :")
    print(unique_values_2)

    # --------------------------------------------------
    # Backward
    # --------------------------------------------------

    loss = output.mean()

    loss.backward()

    print("\n✓ Backward OK")

    # --------------------------------------------------
    # Vérification gradients
    # --------------------------------------------------

    assert mlp.fc1.weight.grad is not None
    assert mlp.fc2.weight.grad is not None

    print("✓ Gradient fc1 OK")
    print("✓ Gradient fc2 OK")

    print("\nBitMLP fonctionne correctement.")


if __name__ == "__main__":
    main()