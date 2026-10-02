import torch
from llm.tokenizer.tokenizer import get_vocab_size
from llm.data.dataloader import create_train_dataloader
from llm.model.embedding import TokenEmbedding
from llm.model.positional_encoding import PositionalEncoding
from llm.model.attention import MultiHeadCausalSelfAttention
from llm.model.feed_forward import FeedForward


EMBEDDING_DIM = 120
NUM_HEADS = 3
MAX_SEQUENCE_LENGTH = 128
FFN_HIDDEN_DIM = 480


def main():
    print("=" * 60)
    print("TEST DU FEED FORWARD NETWORK")
    print("=" * 60)

    train_loader = create_train_dataloader()

    batch = next(iter(train_loader))
    input_ids = batch["input_ids"]

    print("\nInput IDs :")
    print(input_ids.shape)

    # 1. Token Embedding
    token_embedding = TokenEmbedding(
        vocab_size = get_vocab_size(),
        embedding_dim=EMBEDDING_DIM,
    )

    embeddings = token_embedding(input_ids)

    print("\nToken Embeddings :")
    print(embeddings.shape)

    # 2. Positional Encoding
    positional_encoding = PositionalEncoding(
        embedding_dim=EMBEDDING_DIM,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
    )

    x = positional_encoding(embeddings)

    print("\nAprès Positional Encoding :")
    print(x.shape)

    # 3. Multi-Head Causal Self-Attention
    attention = MultiHeadCausalSelfAttention(
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
    )

    x = attention(x)

    print("\nAprès Multi-Head Causal Self-Attention :")
    print(x.shape)

    # 4. Feed Forward Network
    feed_forward = FeedForward(
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=FFN_HIDDEN_DIM,
    )

    output = feed_forward(x)

    print("\nAprès Feed Forward Network :")
    print(output.shape)

    print("\n--- CONFIGURATION ---")
    print(f"Embedding dimension : {EMBEDDING_DIM}")
    print(f"FFN hidden dimension : {FFN_HIDDEN_DIM}")
    print(f"Expansion : {FFN_HIDDEN_DIM // EMBEDDING_DIM}x")

    print("\n--- VÉRIFICATION ---")

    assert input_ids.shape == (2, 128)
    assert embeddings.shape == (2, 128, EMBEDDING_DIM)
    assert x.shape == (2, 128, EMBEDDING_DIM)
    assert output.shape == (2, 128, EMBEDDING_DIM)

    print("✓ Input IDs correctement reçus")
    print("✓ Token Embedding fonctionne")
    print("✓ Positional Encoding fonctionne")
    print("✓ Multi-Head Attention fonctionne")
    print("✓ Feed Forward fonctionne")
    print("✓ Dimension interne du FFN = 480")
    print("✓ Shape finale correcte")


if __name__ == "__main__":
    main()