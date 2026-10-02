import torch

from llm.bitnet_model.bit_attention import BitAttention


def main():

    # --------------------------------------------------
    # Configuration
    # --------------------------------------------------

    batch_size = 2
    sequence_length = 8

    d_model = 256
    num_heads = 4

    # --------------------------------------------------
    # Création de BitAttention
    # --------------------------------------------------

    attention = BitAttention(
        d_model=d_model,
        num_heads=num_heads,
        bias=False,
    )

    print("BitAttention :")
    print(attention)

    print("\nNombre de têtes :")
    print(attention.num_heads)

    print("\nDimension par tête :")
    print(attention.head_dim)

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

    output = attention(x)

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
    # Test causal mask
    # --------------------------------------------------

    mask = attention.create_causal_mask(
        sequence_length=sequence_length,
        device=x.device,
    )

    print("\nCausal mask :")
    print(mask)

    assert mask.shape == (
        sequence_length,
        sequence_length,
    )

    # La diagonale et la partie inférieure
    # doivent être False.

    assert mask[0, 0].item() is False
    assert mask[1, 0].item() is False
    assert mask[1, 1].item() is False

    # La partie supérieure doit être True.

    assert mask[0, 1].item() is True
    assert mask[0, 7].item() is True
    assert mask[3, 7].item() is True

    print("\n✓ Causal mask OK")

    # --------------------------------------------------
    # Test quantification Q
    # --------------------------------------------------

    q_weight = attention.q_proj.quantize_weights(
        attention.q_proj.weight
    )

    q_unique = torch.unique(
        q_weight
    )

    print("\nValeurs uniques Q :")
    print(q_unique)

    # --------------------------------------------------
    # Test quantification K
    # --------------------------------------------------

    k_weight = attention.k_proj.quantize_weights(
        attention.k_proj.weight
    )

    k_unique = torch.unique(
        k_weight
    )

    print("\nValeurs uniques K :")
    print(k_unique)

    # --------------------------------------------------
    # Test quantification V
    # --------------------------------------------------

    v_weight = attention.v_proj.quantize_weights(
        attention.v_proj.weight
    )

    v_unique = torch.unique(
        v_weight
    )

    print("\nValeurs uniques V :")
    print(v_unique)

    # --------------------------------------------------
    # Backward
    # --------------------------------------------------

    loss = output.mean()

    loss.backward()

    print("\n✓ Backward OK")

    # --------------------------------------------------
    # Vérification gradients
    # --------------------------------------------------

    assert attention.q_proj.weight.grad is not None
    assert attention.k_proj.weight.grad is not None
    assert attention.v_proj.weight.grad is not None
    assert attention.out_proj.weight.grad is not None

    print("✓ Gradient Q OK")
    print("✓ Gradient K OK")
    print("✓ Gradient V OK")
    print("✓ Gradient Output OK")

    print("\nBitAttention fonctionne correctement.")


if __name__ == "__main__":
    main()