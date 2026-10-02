import torch

from llm.bitnet_model.bit_transformer_block import BitTransformerBlock


def main():
    batch_size = 2
    sequence_length = 8
    d_model = 256
    num_heads = 4
    hidden_dim = 680

    block = BitTransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        bias=False,
    )

    print("BitTransformerBlock :")
    print(block)

    x = torch.randn(
        batch_size,
        sequence_length,
        d_model,
    )

    print("\nShape entrée :")
    print(x.shape)

    output = block(x)

    print("\nShape sortie :")
    print(output.shape)

    expected_shape = (
        batch_size,
        sequence_length,
        d_model,
    )

    assert output.shape == expected_shape

    print("\n✓ Forward OK")

    # Vérification RMSNorm
    rms = block.norm1(x)

    assert rms.shape == x.shape

    print("✓ RMSNorm OK")

    # Vérification que les poids BitLinear sont ternaires
    q_weight = block.attention.q_proj.quantize_weights(
        block.attention.q_proj.weight
    )

    q_unique = torch.unique(q_weight)

    print("\nValeurs uniques Q :")
    print(q_unique)

    k_weight = block.attention.k_proj.quantize_weights(
        block.attention.k_proj.weight
    )

    k_unique = torch.unique(k_weight)

    print("\nValeurs uniques K :")
    print(k_unique)

    fc1_weight = block.mlp.fc1.quantize_weights(
        block.mlp.fc1.weight
    )

    fc1_unique = torch.unique(fc1_weight)

    print("\nValeurs uniques MLP fc1 :")
    print(fc1_unique)

    fc2_weight = block.mlp.fc2.quantize_weights(
        block.mlp.fc2.weight
    )

    fc2_unique = torch.unique(fc2_weight)

    print("\nValeurs uniques MLP fc2 :")
    print(fc2_unique)

    # Vérification backward
    loss = output.mean()

    loss.backward()

    print("\n✓ Backward OK")

    assert block.attention.q_proj.weight.grad is not None
    assert block.attention.k_proj.weight.grad is not None
    assert block.attention.v_proj.weight.grad is not None
    assert block.attention.out_proj.weight.grad is not None

    assert block.mlp.fc1.weight.grad is not None
    assert block.mlp.fc2.weight.grad is not None

    assert block.norm1.weight.grad is not None
    assert block.norm2.weight.grad is not None

    print("✓ Gradient Attention OK")
    print("✓ Gradient MLP OK")
    print("✓ Gradient RMSNorm OK")

    print("\nBitTransformerBlock fonctionne correctement.")


if __name__ == "__main__":
    main()