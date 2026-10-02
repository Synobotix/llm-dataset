import torch

from llm.data.dataloader import create_train_dataloader
from llm.model.embedding import TokenEmbedding


VOCAB_SIZE = 1000
EMBEDDING_DIM = 120


def main():
    print("=" * 60)
    print("TEST DE L'EMBEDDING")
    print("=" * 60)

    train_loader = create_train_dataloader()

    batch = next(iter(train_loader))

    input_ids = batch["input_ids"]

    print("\nInput IDs :")
    print(input_ids.shape)

    embedding = TokenEmbedding(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
    )

    embeddings = embedding(input_ids)

    print("\nEmbeddings :")
    print(embeddings.shape)

    print("\nType :")
    print(embeddings.dtype)

    print("\nExemple :")
    print("\nToken ID :", input_ids[0, 0].item())

    print("\nVecteur correspondant :")
    print(embeddings[0, 0])

    print("\n--- VÉRIFICATION ---")

    assert input_ids.shape == (2, 128)

    assert embeddings.shape == (
        2,
        128,
        EMBEDDING_DIM,
    )

    assert embeddings.dtype == torch.float32

    print("✓ Input IDs correctement reçus")
    print("✓ Embedding fonctionne")
    print("✓ Shape [batch, sequence, embedding] correcte")
    print("✓ Les IDs sont transformés en vecteurs")


if __name__ == "__main__":
    main()