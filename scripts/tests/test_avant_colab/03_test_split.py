"""
Test 3 — Split train/validation.

Objectif :
- Vérifier que le split fonctionne
- Vérifier les proportions train/validation
- Produire data/processed/train.jsonl et validation.jsonl

Usage :
    poetry run python -m scripts.tests.test_avant_colab.03_test_split
"""

import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# CONFIGURATION
# ============================================================

# Fichier source (produit par 02_test_download.py)
CLEAN_DATASET_FILE = Path("data/processed/c4_clean.jsonl")

# Fichiers de sortie
TRAIN_FILE = Path("data/processed/train.jsonl")
VALIDATION_FILE = Path("data/processed/validation.jsonl")

# Proportions
TRAIN_RATIO = 0.8
RANDOM_SEED = 42


# ============================================================
# LECTURE
# ============================================================

def read_documents(file_path):
    """
    Lit les documents JSONL et retourne uniquement
    les documents avec un champ 'text' valide.
    """

    documents = []

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

            documents.append({"text": text})

    return documents


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TEST 3 — SPLIT TRAIN/VALIDATION")
    print("=" * 70)

    print(f"\nSource    : {CLEAN_DATASET_FILE}")
    print(f"Ratio     : {TRAIN_RATIO}")
    print(f"Seed      : {RANDOM_SEED}")

    # --------------------------------------------------------
    # VÉRIFICATION FICHIER SOURCE
    # --------------------------------------------------------

    if not CLEAN_DATASET_FILE.exists():

        print(f"\n❌ {CLEAN_DATASET_FILE} introuvable.")
        print("Lancez d'abord :")
        print("  poetry run python -m scripts.tests.test_avant_colab.02_test_download")
        return False

    # --------------------------------------------------------
    # LECTURE
    # --------------------------------------------------------

    print(f"\nLecture de {CLEAN_DATASET_FILE}...")

    documents = read_documents(CLEAN_DATASET_FILE)

    print(f"Documents lus : {len(documents)}")

    if len(documents) == 0:

        print("\n❌ Aucun document à traiter.")
        return False

    # --------------------------------------------------------
    # SHUFFLE REPRODUCTIBLE
    # --------------------------------------------------------

    random.seed(RANDOM_SEED)
    random.shuffle(documents)

    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

    train_size = int(len(documents) * TRAIN_RATIO)

    train_docs = documents[:train_size]
    validation_docs = documents[train_size:]

    # --------------------------------------------------------
    # ÉCRITURE
    # --------------------------------------------------------

    TRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)

    with TRAIN_FILE.open("w", encoding="utf-8") as f:

        for doc in train_docs:
            f.write(
                json.dumps(doc, ensure_ascii=False) + "\n"
            )

    with VALIDATION_FILE.open("w", encoding="utf-8") as f:

        for doc in validation_docs:
            f.write(
                json.dumps(doc, ensure_ascii=False) + "\n"
            )

    # --------------------------------------------------------
    # VÉRIFICATION
    # --------------------------------------------------------

    train_size_actual = train_size
    validation_size_actual = len(documents) - train_size

    print("\n" + "=" * 70)
    print("VÉRIFICATION")
    print("=" * 70)

    print(f"\n{TRAIN_FILE} :")

    if TRAIN_FILE.exists():

        with TRAIN_FILE.open("r", encoding="utf-8") as f:
            actual_train_lines = sum(1 for _ in f)

        size_kb = TRAIN_FILE.stat().st_size / 1024

        print(f"  Lignes : {actual_train_lines}")
        print(f"  Taille : {size_kb:.2f} Ko")

        if actual_train_lines == train_size_actual:
            print(f"  ✅ Correspond au split attendu")
        else:
            print(f"  ❌ Attendu {train_size_actual}, obtenu {actual_train_lines}")

    else:

        print(f"  ❌ Fichier absent")

    print(f"\n{VALIDATION_FILE} :")

    if VALIDATION_FILE.exists():

        with VALIDATION_FILE.open("r", encoding="utf-8") as f:
            actual_val_lines = sum(1 for _ in f)

        size_kb = VALIDATION_FILE.stat().st_size / 1024

        print(f"  Lignes : {actual_val_lines}")
        print(f"  Taille : {size_kb:.2f} Ko")

        if actual_val_lines == validation_size_actual:
            print(f"  ✅ Correspond au split attendu")
        else:
            print(f"  ❌ Attendu {validation_size_actual}, obtenu {actual_val_lines}")

    else:

        print(f"  ❌ Fichier absent")

    # --------------------------------------------------------
    # RÉCAPITULATIF
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TEST 3 TERMINÉ")
    print("=" * 70)

    print(f"\nTotal      : {len(documents)}")
    print(f"Train      : {len(train_docs)}   ({TRAIN_RATIO * 100:.0f}%)")
    print(f"Validation : {len(validation_docs)}   ({(1 - TRAIN_RATIO) * 100:.0f}%)")
    print(f"\n→ {TRAIN_FILE}")
    print(f"→ {VALIDATION_FILE}")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)