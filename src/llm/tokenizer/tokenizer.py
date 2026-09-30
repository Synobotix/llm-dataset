
from pathlib import Path

from tokenizers import Tokenizer


class LLMTokenizer:
    def __init__(self, tokenizer_path: str | Path):
        self.tokenizer_path = Path(tokenizer_path)

        if not self.tokenizer_path.exists():
            raise FileNotFoundError(
                f"Tokenizer introuvable : {self.tokenizer_path}"
            )

        self.tokenizer = Tokenizer.from_file(
            str(self.tokenizer_path)
        )

        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("Le texte doit être une chaîne de caractères.")

        return self.tokenizer.encode(text).ids

    def decode(self, token_ids: list[int]) -> str:
        if not isinstance(token_ids, list):
            token_ids = list(token_ids)

        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True
        )

