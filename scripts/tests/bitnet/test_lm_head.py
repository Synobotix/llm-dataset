import torch

from llm.bitnet_model.lm_head import LMHead


def main():
    batch_size = 2
    sequence_length = 8
    d_model = 256
    vocab_size = 994

    lm_head = LMHead(
        d_model=d_model,
        vocab_size=vocab_size,
        bias=False,
    )

    print("LM Head :")
    print(lm_head)

    x = torch.randn(
        batch_size,
        sequence_length,
        d_model,
    )

    print("\nShape entrée :")
    print(x.shape)

    logits = lm_head(x)

    print("\nShape sortie :")
    print(logits.shape)

    expected_shape = (
        batch_size,
        sequence_length,
        vocab_size,
    )

    assert logits.shape == expected_shape

    print("\n✓ Forward OK")

    # Vérification des poids ternaires
    quantized_weight = lm_head.projection.quantize_weights(
        lm_head.projection.weight
    )

    unique_values = torch.unique(quantized_weight)

    print("\nValeurs uniques des poids :")
    print(unique_values)

    # Backward
    loss = logits.mean()

    loss.backward()

    print("\n✓ Backward OK")

    assert lm_head.projection.weight.grad is not None

    print("✓ Gradient LM Head OK")

    print("\nLM Head fonctionne correctement.")


if __name__ == "__main__":
    main()