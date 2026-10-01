from pathlib import Path

import torch

from llm.tokenizer.tokenizer import load_tokenizer
from llm.model.transformer import CausalTransformer


CHECKPOINT_PATH = Path(
    "checkpoints/student_v1/student_v1_final.pt"
)

PROMPT = "Kit point de"


def main():
    print("=" * 70)
    print("AUTOPSIE DU STUDENT")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. TOKENIZER
    # ---------------------------------------------------------

    tokenizer = load_tokenizer()

    encoded = tokenizer.encode(PROMPT)

    print("\n[1] TOKENIZER")
    print("-" * 70)

    print("Texte :", PROMPT)
    print("Tokens :", encoded.tokens)
    print("IDs    :", encoded.ids)
    print("Vocab  :", tokenizer.get_vocab_size())

    # ---------------------------------------------------------
    # 2. CHECKPOINT
    # ---------------------------------------------------------

    print("\n[2] CHECKPOINT")
    print("-" * 70)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint introuvable : {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

    print("Checkpoint :", CHECKPOINT_PATH)
    print("Global step :", checkpoint.get("global_step"))

    # ---------------------------------------------------------
    # 3. MODELE
    # ---------------------------------------------------------

    print("\n[3] MODELE")
    print("-" * 70)

    vocab_size = tokenizer.get_vocab_size()

    model = CausalTransformer(
        vocab_size=vocab_size,
        embedding_dim=120,
        num_heads=3,
        ffn_hidden_dim=480,
        num_layers=4,
        max_sequence_length=128,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("Vocab size       :", vocab_size)
    print("Embedding dim    :", 120)
    print("Heads            :", 3)
    print("Head dim         :", 40)
    print("FFN hidden dim   :", 480)
    print("Layers           :", 4)
    print("Sequence length  :", 128)

    # ---------------------------------------------------------
    # 4. ENTREE DU MODELE
    # ---------------------------------------------------------

    print("\n[4] ENTREE")
    print("-" * 70)

    input_ids = torch.tensor(
        [encoded.ids],
        dtype=torch.long,
    )

    print("Shape :", tuple(input_ids.shape))
    print("IDs   :", input_ids.tolist())

    # ---------------------------------------------------------
    # 5. FORWARD
    # ---------------------------------------------------------

    with torch.no_grad():
        logits = model(input_ids)

    print("\n[5] FORWARD")
    print("-" * 70)

    print("Logits shape :", tuple(logits.shape))

    # ---------------------------------------------------------
    # 6. DERNIERE POSITION
    # ---------------------------------------------------------

    next_token_logits = logits[:, -1, :]

    print("\n[6] DERNIERE POSITION")
    print("-" * 70)

    print("Shape :", tuple(next_token_logits.shape))

    # ---------------------------------------------------------
    # 7. PROBABILITES
    # ---------------------------------------------------------

    probabilities = torch.softmax(
        next_token_logits,
        dim=-1,
    )

    # ---------------------------------------------------------
    # 8. TOP 20
    # ---------------------------------------------------------

    top_probabilities, top_ids = torch.topk(
        probabilities[0],
        k=20,
    )

    print("\n[7] TOP 20 PREDICTIONS")
    print("-" * 70)

    for rank, (token_id, probability) in enumerate(
        zip(
            top_ids.tolist(),
            top_probabilities.tolist(),
        ),
        start=1,
    ):
        token = tokenizer.decode([token_id])

        print(
            f"{rank:2d}. "
            f"ID={token_id:4d} "
            f"P={probability:.8%} "
            f"Token={token!r}"
        )

    # ---------------------------------------------------------
    # 9. RECHERCHE DES TOKENS IMPORTANTS
    # ---------------------------------------------------------

    print("\n[8] TOKENS CIBLES")
    print("-" * 70)

    interesting_tokens = [
        " de",
        " croix",
        " quitte",
        " la",
        " le",
    ]

    for token in interesting_tokens:
        token_id = tokenizer.token_to_id(token)

        if token_id is None:
            print(
                f"{token!r} -> absent du vocabulaire"
            )
            continue

        probability = probabilities[
            0,
            token_id,
        ].item()

        logit = next_token_logits[
            0,
            token_id,
        ].item()

        print(
            f"{token!r} "
            f"ID={token_id} "
            f"logit={logit:.8f} "
            f"P={probability:.8%}"
        )


if __name__ == "__main__":
    main()