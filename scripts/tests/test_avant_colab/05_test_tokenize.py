"""
Test 5 — Tokeniser le dataset.

Objectif :
- Vérifier que la tokenisation fonctionne
- Produire data/tokenized/train.jsonl et validation.jsonl
- Vérifier que chaque séquence a exactement BLOCK_SIZE tokens
- Vérifier le décalage input_ids / labels

Usage :
    poetry run python -m scripts.tests.test_avant_colab.05_test_tokenize
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tokenizers import Tokenizer


# ============================================================
# CONFIGURATION
# ============================================================

# Fichiers source (produits par 03_test_split.py)
TRAIN_FILE = Path("data/processed/train.jsonl")
VALIDATION_FILE = Path("data/processed/validation.jsonl")

# Fichiers de sortie
TRAIN_OUTPUT = Path("data/tokenized/train.jsonl")
VALIDATION_OUTPUT = Path("data/tokenized/validation.jsonl")

# Tokenizer
TOKENIZER_FILE = Path("tokenizer/tokenizer.json")

# Taille des séquences
BLOCK_SIZE = 128


# ============================================================
# LECTURE
# ============================================================

def read_texts(file_path):
    """
    Lit les textes depuis un fichier JSONL.
    """

    texts = []

    with file_path.open("r", encoding="utf-8") as f:

        for line_num, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                doc = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"⚠️ Ligne {line_num} invalide : {e}")
                continue

            text = doc.get("text")

            if not isinstance(text, str):
                continue

            text = text.strip()

            if not text:
                continue

            texts.append(text)

    return texts


# ============================================================
# CRÉATION DES SÉQUENCES
# ============================================================

def create_sequences(token_ids):
    """
    Découpe les tokens en séquences de (BLOCK_SIZE + 1) :

        input_ids = sequence[:-1]   → BLOCK_SIZE tokens
        labels    = sequence[1:]    → BLOCK_SIZE tokens

    Les séquences incomplètes sont ignorées.
    """

    sequence_length = BLOCK_SIZE + 1

    sequences = []

    for start in range(
        0,
        len(token_ids) - sequence_length + 1,
        BLOCK_SIZE,
    ):

        seq = token_ids[start : start + sequence_length]

        if len(seq) != sequence_length:
            continue

        sequences.append({
            "input_ids": seq[:-1],
            "labels": seq[1:],
        })

    return sequences


# ============================================================
# TRAITEMENT D'UN FICHIER
# ============================================================

def process_dataset(tokenizer, input_file, output_file):

    print(f"\nTraitement : {input_file}")

    texts = read_texts(input_file)

    print(f"  Textes lus : {len(texts)}")

    total_tokens = 0
    total_seqs = 0

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as out:

        for i, text in enumerate(texts, start=1):

            ids = tokenizer.encode(text).ids

            total_tokens += len(ids)

            sequences = create_sequences(ids)

            for seq in sequences:
                out.write(json.dumps(seq, ensure_ascii=False) + "\n")
                total_seqs += 1

            if i % 20 == 0:
                print(f"  Traités : {i}/{len(texts)}")

    print(f"  Tokens    : {total_tokens:,}")
    print(f"  Séquences : {total_seqs:,}")
    print(f"  Sortie    : {output_file}")

    return total_seqs


# ============================================================
# VÉRIFICATION
# ============================================================

def verify_dataset(file_path):

    print(f"\nVérification : {file_path}")

    if not file_path.exists():
        print("  ❌ Fichier absent")
        return False

    total = 0
    invalid_len = 0
    invalid_shift = 0

    with file_path.open("r", encoding="utf-8") as f:

        for line_num, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                invalid_len += 1
                continue

            input_ids = sample.get("input_ids")
            labels = sample.get("labels")

            # Vérifier la longueur
            if (
                not isinstance(input_ids, list)
                or len(input_ids) != BLOCK_SIZE
                or not isinstance(labels, list)
                or len(labels) != BLOCK_SIZE
            ):
                invalid_len += 1
                if invalid_len <= 3:
                    print(f"  ⚠️ Ligne {line_num} : longueur incorrecte")
                continue

            # Vérifier le décalage (input_ids[1:] == labels[:-1])
            if input_ids[1:] != labels[:-1]:
                invalid_shift += 1
                if invalid_shift <= 3:
                    print(f"  ⚠️ Ligne {line_num} : décalage incorrect")

            total += 1

    # Résumé
    print(f"  Séquences : {total}")

    if invalid_len > 0:
        print(f"  ❌ Longueurs incorrectes : {invalid_len}")
    else:
        print(f"  ✅ Longueurs correctes ({BLOCK_SIZE} input_ids + {BLOCK_SIZE} labels)")

    if invalid_shift > 0:
        print(f"  ❌ Décalages incorrects : {invalid_shift}")
    else:
        print(f"  ✅ Décalage input_ids/labels correct")

    return invalid_len == 0 and invalid_shift == 0


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TEST 5 — TOKENISATION")
    print("=" * 70)

    # --------------------------------------------------------
    # VÉRIFICATION DES FICHIERS SOURCES
    # --------------------------------------------------------

    if not TRAIN_FILE.exists():
        print(f"\n❌ {TRAIN_FILE} introuvable.")
        print("Lancez d'abord : 03_test_split")
        return False

    if not VALIDATION_FILE.exists():
        print(f"\n❌ {VALIDATION_FILE} introuvable.")
        print("Lancez d'abord : 03_test_split")
        return False

    if not TOKENIZER_FILE.exists():
        print(f"\n❌ {TOKENIZER_FILE} introuvable.")
        print("Lancez d'abord : 04_test_tokenizer")
        return False

    # --------------------------------------------------------
    # CHARGEMENT DU TOKENIZER
    # --------------------------------------------------------

    print(f"\nTokenizer : {TOKENIZER_FILE}")

    tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))

    print(f"Vocab      : {tokenizer.get_vocab_size()}")
    print(f"Block size : {BLOCK_SIZE}")

    # --------------------------------------------------------
    # TRAITEMENT TRAIN
    # --------------------------------------------------------

    train_seqs = process_dataset(
        tokenizer=tokenizer,
        input_file=TRAIN_FILE,
        output_file=TRAIN_OUTPUT,
    )

    # --------------------------------------------------------
    # TRAITEMENT VALIDATION
    # --------------------------------------------------------

    val_seqs = process_dataset(
        tokenizer=tokenizer,
        input_file=VALIDATION_FILE,
        output_file=VALIDATION_OUTPUT,
    )

    # --------------------------------------------------------
    # VÉRIFICATION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VÉRIFICATION DES SORTIES")
    print("=" * 70)

    train_ok = verify_dataset(TRAIN_OUTPUT)
    val_ok = verify_dataset(VALIDATION_OUTPUT)

    # --------------------------------------------------------
    # RÉSUMÉ
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TEST 5 TERMINÉ")
    print("=" * 70)

    print(f"\nVocab      : {tokenizer.get_vocab_size()}")
    print(f"Block size : {BLOCK_SIZE}")
    print(f"\nTrain      : {train_seqs:,} séquences")
    print(f"Validation : {val_seqs:,} séquences")

    print(f"\n→ {TRAIN_OUTPUT}")
    print(f"→ {VALIDATION_OUTPUT}")

    if train_ok and val_ok:
        print("\n✅ Toutes les vérifications sont passées")
    else:
        print("\n⚠️ Certaines vérifications ont échoué")

    return train_ok and val_ok and train_seqs > 0 and val_seqs > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)