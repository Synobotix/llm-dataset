import torch

from llm.model.attention import MultiHeadCausalSelfAttention


def main():

    print("=" * 60)
    print("TEST ATTENTION CAUSALE")
    print("=" * 60)

    # Configuration volontairement petite
    embedding_dim = 12
    num_heads = 3
    max_sequence_length = 5

    attention = MultiHeadCausalSelfAttention(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        max_sequence_length=max_sequence_length,
    )

    attention.eval()

    # --------------------------------------------------
    # Séquence de 5 tokens
    # --------------------------------------------------

    batch_size = 1
    sequence_length = 5

    x = torch.randn(
        batch_size,
        sequence_length,
        embedding_dim,
    )

    print("\nEntrée :")
    print("shape :", x.shape)

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------

    with torch.no_grad():

        output = attention(x)

    # --------------------------------------------------
    # Résultat
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("RÉSULTAT")
    print("=" * 60)

    print(
        "Sortie shape :",
        output.shape,
    )

    print(
        "NaN :",
        torch.isnan(output).any().item(),
    )

    print(
        "Inf :",
        torch.isinf(output).any().item(),
    )


if __name__ == "__main__":
    main()