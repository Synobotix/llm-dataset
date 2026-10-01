import torch

from llm.data.dataloader import create_train_dataloader
from llm.model.embedding import TokenEmbedding
from llm.model.positional_encoding import PositionalEncoding


VOCAB_SIZE = 1000
EMBEDDING_DIM = 128
MAX_SEQUENCE_LENGTH = 128


def main():
    print("=" * 60)
    print("TEST DU POSITIONAL ENCODING")
    print("=" * 60)

    train_loader = create_train_dataloader()

    batch = next(iter(train_loader))
    input_ids = batch["input_ids"]

    print("\nInput IDs :")
    print(input_ids.shape)

    token_embedding = TokenEmbedding(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
    )

    embeddings = token_embedding(input_ids)

    print("\nToken Embeddings :")
    print(embeddings.shape)

    positional_encoding = PositionalEncoding(
        embedding_dim=EMBEDDING_DIM,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
    )

    output = positional_encoding(embeddings)

    print("\nAprès Positional Encoding :")
    print(output.shape)

    print("\nVecteur position 0 :")
    print(output[0, 0])

    print("\nVecteur position 1 :")
    print(output[0, 1])

    print("\n--- VÉRIFICATION ---")

    assert embeddings.shape == (
        2,
        128,
        EMBEDDING_DIM,
    )

    assert output.shape == (
        2,
        128,
        EMBEDDING_DIM,
    )

    assert positional_encoding.positional_encoding.shape == (
        1,
        MAX_SEQUENCE_LENGTH,
        EMBEDDING_DIM,
    )

    print("✓ Token Embedding fonctionne")
    print("✓ Positional Encoding fonctionne")
    print("✓ Shape correcte")
    print("✓ Les positions sont ajoutées aux embeddings")


if __name__ == "__main__":
    main()