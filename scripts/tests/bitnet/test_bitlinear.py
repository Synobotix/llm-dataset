import torch

from llm.bitnet_model.bitlinear import BitLinear


def main():

    # --------------------------------------------------
    # Paramètres
    # --------------------------------------------------

    batch_size = 2
    sequence_length = 8
    in_features = 16
    out_features = 32

    # --------------------------------------------------
    # Création de BitLinear
    # --------------------------------------------------

    layer = BitLinear(
        in_features=in_features,
        out_features=out_features,
        bias=False
    )

    print("BitLinear :")
    print(layer)

    # --------------------------------------------------
    # Entrée
    # --------------------------------------------------

    x = torch.randn(
        batch_size,
        sequence_length,
        in_features
    )

    print("\nShape entrée :")
    print(x.shape)

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------

    output = layer(x)

    print("\nShape sortie :")
    print(output.shape)

    # --------------------------------------------------
    # Vérification
    # --------------------------------------------------

    expected_shape = (
        batch_size,
        sequence_length,
        out_features
    )

    assert output.shape == expected_shape

    print("\n✓ Forward OK")

    # --------------------------------------------------
    # Test des poids quantifiés
    # --------------------------------------------------

    weight_quantized = layer.quantize_weights(
        layer.weight
    )

    unique_values = torch.unique(
        weight_quantized
    )

    print("\nValeurs uniques des poids quantifiés :")
    print(unique_values)

    # --------------------------------------------------
    # Backward
    # --------------------------------------------------

    loss = output.mean()

    loss.backward()

    print("\n✓ Backward OK")

    # --------------------------------------------------
    # Vérification gradient
    # --------------------------------------------------

    print("\nGradient des poids :")

    print(layer.weight.grad)

    assert layer.weight.grad is not None

    print("\n✓ Gradient OK")

    print("\nBitLinear fonctionne correctement.")


if __name__ == "__main__":
    main()