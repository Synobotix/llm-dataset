import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer


TRAIN_FILE = Path("data/processed/train.jsonl")
TOKENIZER_DIR = Path("tokenizer")

TOKENIZER_FILE = TOKENIZER_DIR / "tokenizer.json"
VOCAB_FILE = TOKENIZER_DIR / "vocab.json"
CONFIG_FILE = TOKENIZER_DIR / "config.json"

VOCAB_SIZE = 16_000
MIN_FREQUENCY = 2

SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
]


def text_iterator(file_path: Path):
    """
    Lit le fichier JSONL et retourne uniquement
    les textes valides.
    """

    with file_path.open("r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                document = json.loads(line)

            except json.JSONDecodeError as error:

                print(
                    f"[WARNING] Ligne {line_number} ignorée : {error}"
                )

                continue

            text = document.get("text")

            if not isinstance(text, str):
                continue

            text = text.strip()

            if text:
                yield text


def create_tokenizer():
    """
    Crée le tokenizer BPE avec ByteLevel.
    """

    tokenizer = Tokenizer(
        BPE(
            unk_token="<unk>",
            byte_fallback=True,
        )
    )

    tokenizer.pre_tokenizer = ByteLevelPreTokenizer(
        add_prefix_space=False
    )

    tokenizer.decoder = ByteLevelDecoder()

    return tokenizer


def train_tokenizer(tokenizer):
    """
    Entraîne le tokenizer sur train.jsonl.
    """

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,

        special_tokens=SPECIAL_TOKENS,

        # Important :
        # conserve l'alphabet ByteLevel complet,
        # même lorsque certains caractères/bytes
        # sont absents ou rares dans notre petit dataset.
        initial_alphabet=ByteLevelPreTokenizer.alphabet(),

        show_progress=True,
    )

    tokenizer.train_from_iterator(
        text_iterator(TRAIN_FILE),
        trainer=trainer,
    )


def save_tokenizer(tokenizer):
    """
    Sauvegarde le tokenizer et son vocabulaire.
    """

    TOKENIZER_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # tokenizer.json
    tokenizer.save(
        str(TOKENIZER_FILE)
    )

    # vocab.json
    vocab = tokenizer.get_vocab()

    sorted_vocab = dict(
        sorted(
            vocab.items(),
            key=lambda item: item[1]
        )
    )

    with VOCAB_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            sorted_vocab,
            file,
            ensure_ascii=False,
            indent=2
        )

    # config.json
    config = {
        "type": "BPE",
        "vocab_size": tokenizer.get_vocab_size(),

        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "bos_token": "<bos>",
        "eos_token": "<eos>",

        "byte_fallback": True,
        "pre_tokenizer": "ByteLevel",
    }

    with CONFIG_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            config,
            file,
            ensure_ascii=False,
            indent=2
        )


def test_tokenizer():
    """
    Test classique du tokenizer.
    """

    tokenizer = Tokenizer.from_file(
        str(TOKENIZER_FILE)
    )

    text = "Bonjour, je suis un modèle de langage."

    encoding = tokenizer.encode(text)

    print("\n" + "=" * 60)
    print("TEST DU TOKENIZER")
    print("=" * 60)

    print("\nTexte :")
    print(text)

    print("\nTokens :")
    print(encoding.tokens)

    print("\nIDs :")
    print(encoding.ids)

    print("\nNombre de tokens :")
    print(len(encoding.ids))

    print("\nDécodage :")
    print(tokenizer.decode(encoding.ids))


def test_utf8(tokenizer):
    """
    Vérifie que les caractères UTF-8 sont
    correctement tokenisés puis reconstruits.
    """

    texts = [
        "Bonjour, je suis un modèle.",
        "J'aime l'éléphant.",
        "Ça va très bien.",
        "École, français, développement.",
        "à â ä é è ê ë î ï ô ö ù û ü ç",
    ]

    print("\n" + "=" * 60)
    print("TEST UTF-8")
    print("=" * 60)

    for text in texts:

        encoding = tokenizer.encode(text)

        decoded = tokenizer.decode(
            encoding.ids
        )

        print("\nOriginal  :", text)
        print("Tokens    :", encoding.tokens)
        print("IDs       :", encoding.ids)
        print("Décodé    :", decoded)
        print("Identique :", text == decoded)


def test_special_tokens(tokenizer):
    """
    Vérifie la présence des tokens spéciaux.
    """

    print("\n" + "=" * 60)
    print("TEST DES TOKENS SPÉCIAUX")
    print("=" * 60)

    for token in SPECIAL_TOKENS:

        token_id = tokenizer.token_to_id(token)

        print(
            f"{token:6} → ID {token_id}"
        )


def main():

    print("=" * 60)
    print("ENTRAÎNEMENT DU TOKENIZER BPE")
    print("=" * 60)

    # Vérification du dataset
    if not TRAIN_FILE.exists():

        raise FileNotFoundError(
            f"Dataset introuvable : {TRAIN_FILE}"
        )

    print(f"\nDataset : {TRAIN_FILE}")
    print(f"Vocabulaire cible : {VOCAB_SIZE}")
    print(f"Fréquence minimale : {MIN_FREQUENCY}")

    # Création
    print("\nCréation du tokenizer...")

    tokenizer = create_tokenizer()

    # Entraînement
    print("Entraînement...\n")

    train_tokenizer(tokenizer)

    # Sauvegarde
    print("\nSauvegarde...")

    save_tokenizer(tokenizer)

    print("\nTokenizer entraîné avec succès.")

    print(
        f"Nombre de tokens : "
        f"{tokenizer.get_vocab_size()}"
    )

    print("\nFichiers créés :")

    print(f"  → {TOKENIZER_FILE}")
    print(f"  → {VOCAB_FILE}")
    print(f"  → {CONFIG_FILE}")

    # Tests
    test_tokenizer()

    test_utf8(tokenizer)

    test_special_tokens(tokenizer)


if __name__ == "__main__":
    main()