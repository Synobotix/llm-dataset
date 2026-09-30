
from pathlib import Path

import torch
import yaml

from llm.model.transformer import CausalTransformer
from llm.tokenizer.tokenizer import LLMTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_CONFIG_PATH = PROJECT_ROOT / "configs/model.yaml"
TOKENIZER_PATH = PROJECT_ROOT / "tokenizer/tokenizer.json"
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints/student_v1/student_v1_final.pt"
)


def load_model(device: torch.device) -> CausalTransformer:
    with open(MODEL_CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    model = CausalTransformer(
        vocab_size=config["vocab_size"],
        embedding_dim=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        ff_hidden_dim=config["d_ff"],
        max_sequence_length=config["max_seq_len"],
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model


@torch.no_grad()
def generate_text(
    model: CausalTransformer,
    tokenizer: LLMTokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 50,
) -> str:

    token_ids = tokenizer.encode(prompt)

    if not token_ids:
        return ""

    max_sequence_length = model.max_sequence_length

    generated_ids = token_ids.copy()

    for _ in range(max_new_tokens):

        context_ids = generated_ids[-max_sequence_length:]

        input_ids = torch.tensor(
            [context_ids],
            dtype=torch.long,
            device=device,
        )

        logits = model(input_ids)

        next_token_logits = logits[:, -1, :]

        next_token_id = torch.argmax(
            next_token_logits,
            dim=-1,
        ).item()

        generated_ids.append(next_token_id)

    new_token_ids = generated_ids[len(token_ids):]

    generated_text = tokenizer.decode(
        new_token_ids
    )

    return generated_text


def main() -> None:
    print("=" * 60)
    print("INFERENCE STUDENT V1")
    print("=" * 60)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint introuvable : {CHECKPOINT_PATH}"
        )

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"Tokenizer introuvable : {TOKENIZER_PATH}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device : {device}")

    if device.type == "cuda":
        print(
            f"GPU : {torch.cuda.get_device_name(0)}"
        )

    print()
    print("Chargement du tokenizer...")

    tokenizer = LLMTokenizer(
        TOKENIZER_PATH
    )

    print(
        f"Vocabulaire : {tokenizer.vocab_size}"
    )

    print()
    print("Chargement de Student v1...")

    model = load_model(device)

    print("Student v1 : OK")

    print()
    print("=" * 60)
    print("DIALOGUE")
    print("=" * 60)
    print("Tape 'exit' pour quitter.")
    print()

    while True:

        prompt = input("Vous : ").strip()

        if prompt.lower() == "exit":
            break

        if not prompt:
            continue

        response = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=50,
        )

        print()
        print(f"Student v1 : {response}")
        print()


if __name__ == "__main__":
    main()
