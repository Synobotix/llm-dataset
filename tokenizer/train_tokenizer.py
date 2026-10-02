from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer

from llm.config.parameters import (
    MAX_VOCAB_SIZE,
    MIN_FREQUENCY,
    SPECIAL_TOKENS,
    TRAIN_FILE,
    TOKENIZER_FILE,
    TOKENIZER_VOCAB_FILE,
    TOKENIZER_CONFIG_FILE,
)


def main():

    print("=" * 70)
    print("ENTRAÎNEMENT DU TOKENIZER")
    print("=" * 70)

    print()
    print(f"Dataset : {TRAIN_FILE}")
    print(f"Max vocab size : {MAX_VOCAB_SIZE}")
    print(f"Min frequency : {MIN_FREQUENCY}")

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {TRAIN_FILE}"
        )

    tokenizer = Tokenizer(
        BPE(
            unk_token="<unk>",
            byte_fallback=True,
        )
    )

    tokenizer.pre_tokenizer = ByteLevel(
        add_prefix_space=False
    )

    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=MAX_VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
    )

    tokenizer.train(
        files=[str(TRAIN_FILE)],
        trainer=trainer,
    )

    TOKENIZER_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer.save(
        str(TOKENIZER_FILE)
    )

    # --------------------------------------------------------
    # vocab.json
    # --------------------------------------------------------

    import json

    vocabulary = tokenizer.get_vocab()

    with TOKENIZER_VOCAB_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            vocabulary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # config.json
    # --------------------------------------------------------

    config = {
        "max_vocab_size": MAX_VOCAB_SIZE,
        "actual_vocab_size": tokenizer.get_vocab_size(),
        "min_frequency": MIN_FREQUENCY,
        "special_tokens": SPECIAL_TOKENS,
    }

    with TOKENIZER_CONFIG_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            config,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 70)
    print("TOKENIZER TERMINÉ")
    print("=" * 70)

    print()
    print(
        f"Vocabulaire maximum demandé : "
        f"{MAX_VOCAB_SIZE}"
    )

    print(
        f"Vocabulaire réel             : "
        f"{tokenizer.get_vocab_size()}"
    )

    print()
    print(f"Tokenizer : {TOKENIZER_FILE}")
    print(f"Vocab     : {TOKENIZER_VOCAB_FILE}")
    print(f"Config    : {TOKENIZER_CONFIG_FILE}")


if __name__ == "__main__":
    main()