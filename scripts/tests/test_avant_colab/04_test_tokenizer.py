"""
Test 4 — Entraîner le tokenizer.

Objectif :
- Vérifier que le tokenizer s'entraîne correctement
- Produire tokenizer/tokenizer.json, vocab.json, config.json
- Tester l'encodage/décodage sur un texte

Usage :
    poetry run python -m scripts.tests.test_avant_colab.04_test_tokenizer
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer


# ============================================================
# CONFIGURATION
# ============================================================

# Fichier d'entraînement (produit par 03_test_split.py)
TRAIN_FILE = Path("data/processed/train.jsonl")

# Fichiers de sortie
TOKENIZER_DIR = Path("tokenizer")
TOKENIZER_FILE = TOKENIZER_DIR / "tokenizer.json"
TOKENIZER_VOCAB_FILE = TOKENIZER_DIR / "vocab.json"
TOKENIZER_CONFIG_FILE = TOKENIZER_DIR / "config.json"

# Réduit pour test local
MAX_VOCAB_SIZE = 8000
MIN_FREQUENCY = 2

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TEST 4 — ENTRAÎNEMENT DU TOKENIZER")
    print("=" * 70)

    print(f"\nDataset        : {TRAIN_FILE}")
    print(f"Max vocab      : {MAX_VOCAB_SIZE}")
    print(f"Min frequency  : {MIN_FREQUENCY}")
    print(f"Special tokens : {SPECIAL_TOKENS}")

    # --------------------------------------------------------
    # VÉRIFICATION FICHIER SOURCE
    # --------------------------------------------------------

    if not TRAIN_FILE.exists():

        print(f"\n❌ {TRAIN_FILE} introuvable.")
        print("Lancez d'abord :")
        print("  poetry run python -m scripts.tests.test_avant_colab.03_test_split")
        return False

    # --------------------------------------------------------
    # CRÉATION DU TOKENIZER
    # --------------------------------------------------------

    print("\nCréation du tokenizer BPE...")

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

    # --------------------------------------------------------
    # ENTRAÎNEMENT
    # --------------------------------------------------------

    print("Entraînement...")

    tokenizer.train(
        files=[str(TRAIN_FILE)],
        trainer=trainer,
    )

    # --------------------------------------------------------
    # SAUVEGARDE
    # --------------------------------------------------------

    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer.save(str(TOKENIZER_FILE))

    print(f"\n✅ Sauvegardé : {TOKENIZER_FILE}")

    # --------------------------------------------------------
    # VOCAB.JSON
    # --------------------------------------------------------

    vocabulary = tokenizer.get_vocab()

    with TOKENIZER_VOCAB_FILE.open("w", encoding="utf-8") as f:

        json.dump(
            vocabulary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"✅ Sauvegardé : {TOKENIZER_VOCAB_FILE}")

    # --------------------------------------------------------
    # CONFIG.JSON
    # --------------------------------------------------------

    config = {
        "max_vocab_size": MAX_VOCAB_SIZE,
        "actual_vocab_size": tokenizer.get_vocab_size(),
        "min_frequency": MIN_FREQUENCY,
        "special_tokens": SPECIAL_TOKENS,
    }

    with TOKENIZER_CONFIG_FILE.open("w", encoding="utf-8") as f:

        json.dump(
            config,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"✅ Sauvegardé : {TOKENIZER_CONFIG_FILE}")

    # --------------------------------------------------------
    # TEST D'ENCODAGE
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TEST D'ENCODAGE / DÉCODAGE")
    print("=" * 70)

    test_texts = [
        "Bonjour, ceci est un test.",
        "Vous êtes un agent secret du Centre d'Intelligence.",
        "Considérez la séquence de Collatz modifiée.",
    ]

    for test_text in test_texts:

        encoding = tokenizer.encode(test_text)

        decoded = tokenizer.decode(encoding.ids)

        print(f"\nTexte    : {test_text!r}")
        print(f"Tokens   : {encoding.tokens[:10]}{'...' if len(encoding.tokens) > 10 else ''}")
        print(f"IDs      : {encoding.ids[:10]}{'...' if len(encoding.ids) > 10 else ''}")
        print(f"Longueur : {len(encoding.ids)}")
        print(f"Décodé   : {decoded!r}")

        # Vérification round-trip
        if decoded == test_text:
            print("Round-trip : ✅")
        else:
            print(f"Round-trip : ⚠️ (léger diff)")

    # --------------------------------------------------------
    # VÉRIFICATION DES FICHIERS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VÉRIFICATION DES FICHIERS")
    print("=" * 70)

    for path in [
        TOKENIZER_FILE,
        TOKENIZER_VOCAB_FILE,
        TOKENIZER_CONFIG_FILE,
    ]:

        if path.exists():

            size_kb = path.stat().st_size / 1024

            print(f"  ✅ {path} ({size_kb:.2f} Ko)")

        else:

            print(f"  ❌ {path} absent")

    # --------------------------------------------------------
    # RÉCAPITULATIF
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TEST 4 TERMINÉ")
    print("=" * 70)

    print(f"\nVocab demandé : {MAX_VOCAB_SIZE}")
    print(f"Vocab réel    : {tokenizer.get_vocab_size()}")

    print(f"\nTokenizer : {TOKENIZER_FILE}")
    print(f"Vocab     : {TOKENIZER_VOCAB_FILE}")
    print(f"Config    : {TOKENIZER_CONFIG_FILE}")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)