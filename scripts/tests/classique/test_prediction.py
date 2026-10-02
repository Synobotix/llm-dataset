import torch

from llm.model.transformer import CausalTransformer
from llm.tokenizer.tokenizer import load_tokenizer, get_vocab_size


EMBEDDING_DIM = 120
NUM_HEADS = 3
FFN_HIDDEN_DIM = 480
NUM_LAYERS = 4
MAX_SEQUENCE_LENGTH = 128

CHECKPOINT_PATH = (
    "checkpoints/student_v1/student_v1_final.pt"
)


def main():

    device = "cpu"

    tokenizer = load_tokenizer()
    vocab_size = get_vocab_size()

    model = CausalTransformer(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        ffn_hidden_dim=FFN_HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    prompt = "Kit point de "

    encoded = tokenizer.encode(prompt)

    input_ids = torch.tensor(
        [encoded.ids],
        dtype=torch.long,
        device=device,
    )

    print("=" * 60)
    print("TEST DE PRÉDICTION")
    print("=" * 60)

    print(f"\nPrompt : {prompt}")
    print(f"Tokens : {encoded.tokens}")
    print(f"IDs    : {encoded.ids}")

    with torch.no_grad():

        logits = model(input_ids)

        # Prédictions du dernier token
        next_token_logits = logits[:, -1, :]

        probabilities = torch.softmax(
            next_token_logits,
            dim=-1,
        )

        top_probabilities, top_indices = torch.topk(
            probabilities,
            k=10,
            dim=-1,
        )

    print("\nTop 10 prédictions :")
    print("-" * 60)

    for rank, (token_id, probability) in enumerate(
        zip(
            top_indices[0],
            top_probabilities[0],
        ),
        start=1,
    ):

        token_id = token_id.item()
        probability = probability.item()

        token = tokenizer.decode(
            [token_id]
        )

        print(
            f"{rank:2d}. "
            f"ID={token_id:4d} "
            f"Probabilité={probability:.4%} "
            f"Token={repr(token)}"
        )


if __name__ == "__main__":
    main()