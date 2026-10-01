from pathlib import Path

from tokenizers import Tokenizer


TOKENIZER_FILE = Path("tokenizer/tokenizer.json")


def load_tokenizer() -> Tokenizer:
    if not TOKENIZER_FILE.exists():
        raise FileNotFoundError(
            f"Tokenizer introuvable : {TOKENIZER_FILE}"
        )

    return Tokenizer.from_file(
        str(TOKENIZER_FILE)
    )


def get_vocab_size() -> int:
    tokenizer = load_tokenizer()

    return tokenizer.get_vocab_size()