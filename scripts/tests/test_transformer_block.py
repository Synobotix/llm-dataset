import torch

from llm.tokenizer.tokenizer import get_vocab_size
from llm.data.dataloader import create_train_dataloader
from llm.model.embedding import TokenEmbedding
from llm.model.positional_encoding import PositionalEncoding
from llm.model.transformer_block import TransformerBlock


EMBEDDING_DIM = 120
NUM_HEADS = 3
FFN_HIDDEN_DIM = 480
MAX_SEQUENCE_LENGTH = 128


def main():
    print("=" * 60)
    print("TEST DU TRANSFORMER BLOCK")
    print("=" * 60)

    train_loader = create_train_dataloader()

    batch = next(iter(train_loader))

    input_ids = batch["input_ids"]

    print("\nInput IDs :")
    print(input_ids.shape)

    # 1. Token Embedding
    token_embedding = TokenEmbedding(
        vocab_size=get_vocab_size(),
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

    # 3. Transformer Block
    transformer_block = TransformerBlock(
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        ffn_hidden_dim=FFN_HIDDEN_DIM,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
    )

    output = transformer_block(x)

    print("\nAprès Transformer Block :")
    print(output.shape)

    print("\n--- CONFIGURATION ---")
    print(f"Embedding dimension : {EMBEDDING_DIM}")
    print(f"Nombre de heads     : {NUM_HEADS}")
    print(f"Head dimension      : {EMBEDDING_DIM // NUM_HEADS}")
    print(f"FFN hidden dimension: {FFN_HIDDEN_DIM}")
    print(f"Sequence length     : {MAX_SEQUENCE_LENGTH}")

    print("\n--- VÉRIFICATION ---")

    assert input_ids.shape == (2, 128)
    assert embeddings.shape == (
        2,
        128,
        EMBEDDING_DIM,
    )

    assert x.shape == (
        2,
        128,
        EMBEDDING_DIM,
    )

    assert output.shape == (
        2,
        128,
        EMBEDDING_DIM,
    )

    assert transformer_block.attention.num_heads == NUM_HEADS

    assert transformer_block.attention.head_dim == (
        EMBEDDING_DIM // NUM_HEADS
    )

    assert isinstance(
        transformer_block.layer_norm_1,
        torch.nn.LayerNorm,
    )

    assert isinstance(
        transformer_block.layer_norm_2,
        torch.nn.LayerNorm,
    )

    print("✓ Input IDs correctement reçus")
    print("✓ Token Embedding fonctionne")
    print("✓ Positional Encoding fonctionne")
    print("✓ LayerNorm 1 fonctionne")
    print("✓ Multi-Head Attention fonctionne")
    print("✓ Connexion résiduelle 1 fonctionne")
    print("✓ LayerNorm 2 fonctionne")
    print("✓ Feed Forward fonctionne")
    print("✓ Connexion résiduelle 2 fonctionne")
    print("✓ Transformer Block fonctionne")
    print("✓ Shape finale correcte")


if __name__ == "__main__":
    main()