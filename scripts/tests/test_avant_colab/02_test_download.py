"""
Test 2 — Télécharger quelques documents Nemotron.

Objectif :
- Vérifier que le téléchargement fonctionne
- Produire un petit c4_clean.jsonl (100 docs)

Usage :
    poetry run python -m scripts.tests.test_avant_colab.02_test_download
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_dataset


# ============================================================
# CONFIGURATION
# ============================================================

HF_DATASET_ID = "nvidia/Nemotron-SFT-Multilingual-v1"

# Le split est le sous-ensemble à utiliser :
#   code_de, code_es, code_fr, code_it, code_ja, code_zh
#   math_de, math_es, math_fr, math_it, math_ja, math_zh
#   stem_de, stem_es, stem_fr, stem_it, stem_ja, stem_zh
TARGET_SPLIT = "code_fr"

# Nombre de documents à écrire
NB_DOCUMENTS = 100

# Fichier de sortie
OUTPUT_FILE = Path("data/processed/c4_clean.jsonl")


# ============================================================
# FORMATAGE DES MESSAGES
# ============================================================

def format_messages(messages):
    """
    Transforme une liste de messages en texte brut.
    """

    parts = []

    for msg in messages:

        role = msg.get("role", "")
        content = msg.get("content", "")

        if not content:
            continue

        parts.append(f"<|{role}|>\n{content}")

    return "\n\n".join(parts)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TEST 2 — TÉLÉCHARGEMENT NEMOTRON")
    print("=" * 70)

    print(f"\nDataset      : {HF_DATASET_ID}")
    print(f"Split        : {TARGET_SPLIT}")
    print(f"Nb documents : {NB_DOCUMENTS}")
    print(f"Sortie       : {OUTPUT_FILE}")

    # --------------------------------------------------------
    # CHARGEMENT
    # --------------------------------------------------------

    print("\nChargement en streaming...")

    ds = load_dataset(
        HF_DATASET_ID,
        split=TARGET_SPLIT,
        streaming=True,
    )

    # --------------------------------------------------------
    # DOSSIER
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # ÉCRITURE
    # --------------------------------------------------------

    written = 0
    skipped = 0

    print(f"\nÉcriture vers {OUTPUT_FILE}...\n")

    with OUTPUT_FILE.open("w", encoding="utf-8") as output:

        for example in ds:

            if written >= NB_DOCUMENTS:
                break

            # ------------------------------------------------
            # Extraction
            # ------------------------------------------------

            messages = example.get("messages", [])

            if not messages:
                skipped += 1
                continue

            text = format_messages(messages)

            if not text.strip():
                skipped += 1
                continue

            # ------------------------------------------------
            # Écriture
            # ------------------------------------------------

            output.write(
                json.dumps(
                    {"text": text},
                    ensure_ascii=False,
                )
                + "\n"
            )

            written += 1

            if written % 25 == 0:
                print(f"  Écrits : {written}/{NB_DOCUMENTS}")

    # --------------------------------------------------------
    # RÉSUMÉ
    # --------------------------------------------------------

    if not OUTPUT_FILE.exists() or OUTPUT_FILE.stat().st_size == 0:
        print("\n❌ Aucun document écrit.")
        return False

    size_kb = OUTPUT_FILE.stat().st_size / 1024

    print("\n" + "=" * 70)
    print("TEST 2 TERMINÉ")
    print("=" * 70)
    print(f"\nFichier   : {OUTPUT_FILE}")
    print(f"Taille    : {size_kb:.2f} Ko")
    print(f"Documents : {written}")
    print(f"Ignorés   : {skipped}")

    return written > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)