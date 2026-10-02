import torch

from llm.bitnet_model.bit_transformer import BitTransformer


def test_model(num_blocks):
    print("=" * 60)
    print(f"TEST AVEC {num_blocks} BLOCK(S)")
    print("=" * 60)

    vocab_size = 994
    d_model = 256
    num_heads = 4
    hidden_dim = 680
    max_sequence_length = 128

    batch_size = 2
    sequence_length = 8

    model = BitTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        max_sequence_length=max_sequence_length,
        bias=False,
    )

    print("\nNombre de blocks :")
    print(len(model.blocks))

    assert len(model.blocks) == num_blocks

    print("\nEmbedding :")
    print(model.token_embedding)

    print("\nShape input :")

    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, sequence_length),
    )

    print(input_ids.shape)

    output = model(input_ids)

    print("\nShape output :")
    print(output.shape)

    expected_shape = (
        batch_size,
        sequence_length,
        d_model,
    )

    assert output.shape == expected_shape

    print("\n✓ Forward OK")

    # Vérification de l'embedding
    embedding_output = model.token_embedding(input_ids)

    assert embedding_output.shape == (
        batch_size,
        sequence_length,
        d_model,
    )

    print("✓ Embedding OK")

    # Vérification des blocks
    assert len(model.blocks) == num_blocks

    for index, block in enumerate(model.blocks):
        assert block is not None

        print(f"✓ Block {index + 1} OK")

    # Vérification RMSNorm final
    assert model.final_norm is not None

    print("✓ RMSNorm final OK")

    # Backward
    loss = output.mean()

    loss.backward()

    print("\n✓ Backward OK")

    # Gradient embedding
    assert model.token_embedding.weight.grad is not None

    print("✓ Gradient Embedding OK")

    # Gradient de chaque block
    for index, block in enumerate(model.blocks):

        assert block.attention.q_proj.weight.grad is not None
        assert block.attention.k_proj.weight.grad is not None
        assert block.attention.v_proj.weight.grad is not None
        assert block.attention.out_proj.weight.grad is not None

        assert block.mlp.fc1.weight.grad is not None
        assert block.mlp.fc2.weight.grad is not None

        print(f"✓ Gradients Block {index + 1} OK")

    # Gradient RMSNorm final
    assert model.final_norm.weight.grad is not None

    print("✓ Gradient RMSNorm final OK")

    print("\nBitTransformer fonctionne correctement.")
    print()


def main():

    # Premier test : 1 seul block
    test_model(num_blocks=1)

    # Deuxième test : 2 blocks
    test_model(num_blocks=2)


if __name__ == "__main__":
    main()