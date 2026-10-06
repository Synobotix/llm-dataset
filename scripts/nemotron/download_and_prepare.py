"""
Télécharge Nemotron en streaming et produit
un fichier compatible avec scripts/split_dataset.py.

Format de sortie :
    {"text": "..."}
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_dataset
from tqdm import tqdm

from llm.config.parameters import (
    CLEAN_DATASET_FILE,
)


# ============================================================
# CONFIGURATION
# ============================================================

HF_DATASET_ID = "nvidia/Nemotron-SFT-Multilingual-v1"

# Le dataset n'a PAS de configs — il a une seule config "default"
# On filtre par colonnes
HF_SPLIT = "train"

# Langue cible (parmi: de, es, fr, it, ja, zh)
TARGET_LANGUAGE = "fr"

# Domaine cible (parmi: multilingual_code, multilingual_math, multilingual_stem)
# ⚠️ Vérifiez les valeurs exactes avec la cellule de diagnostic ci-dessous
TARGET_DOMAIN = None   # None = pas de filtre sur le domaine

# Nombre de documents à ÉCRIRE dans le fichier
MAX_DOCUMENTS = 5000

# Limite de sécurité : nombre max d'exemples à PARCOURIR
# (car on filtre, il faut parcourir plus que ce qu'on écrit)
MAX_SCAN = 500_000


# ============================================================
# FORMATAGE DES MESSAGES
# ============================================================

def format_messages(messages):

    parts = []

    for msg in messages:

        role = msg.get("role", "")
        content = msg.get("content", "")

        if not content:
            continue

        parts.append(f"<|{role}|>\n{content}")

    return "\n\n".join(parts)


# ============================================================
# TÉLÉCHARGEMENT + CONVERSION
# ============================================================

def download_and_prepare():

    print("=" * 70)
    print("PRÉPARATION NEMOTRON")
    print("=" * 70)
    print(f"\nDataset : {HF_DATASET_ID}")
    print(f"Langue  : {TARGET_LANGUAGE}")
    print(f"Domaine : {TARGET_DOMAIN or 'tous'}")
    print(f"Sortie  : {CLEAN_DATASET_FILE}")
    print(f"Max doc : {MAX_DOCUMENTS}")

    # --------------------------------------------------------
    # CHARGEMENT EN STREAMING (sans config)
    # --------------------------------------------------------

    print("\nChargement en streaming...")

    ds = load_dataset(
        HF_DATASET_ID,
        split=HF_SPLIT,
        streaming=True,
    )

    # --------------------------------------------------------
    # DIAGNOSTIC : afficher les colonnes disponibles
    # --------------------------------------------------------

    print("\nColonnes disponibles :")

    for example in ds.take(1):
        for key in example.keys():
            print(f"  - {key}")

    # --------------------------------------------------------
    # DOSSIER DE SORTIE
    # --------------------------------------------------------

    CLEAN_DATASET_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # ÉCRITURE
    # --------------------------------------------------------

    written = 0
    scanned = 0
    skipped = 0

    print(f"\nÉcriture vers : {CLEAN_DATASET_FILE}\n")

    with CLEAN_DATASET_FILE.open("w", encoding="utf-8") as output:

        for example in ds:

            scanned += 1

            if scanned > MAX_SCAN:
                print(f"\n⚠️ Limite de scan atteinte ({MAX_SCAN})")
                break

            if written >= MAX_DOCUMENTS:
                break

            if scanned % 1000 == 0:
                print(f"  Scanné: {scanned} | Écrit: {written} | Ignoré: {skipped}")

            # ------------------------------------------------
            # Filtre langue
            # ------------------------------------------------

            if TARGET_LANGUAGE:

                lang = example.get("language")

                if lang != TARGET_LANGUAGE:
                    skipped += 1
                    continue

            # ------------------------------------------------
            # Filtre domaine
            # ------------------------------------------------

            if TARGET_DOMAIN:

                domain = example.get("domain")

                if domain != TARGET_DOMAIN:
                    skipped += 1
                    continue

            # ------------------------------------------------
            # Extraction des messages
            # ------------------------------------------------

            messages = example.get("messages", [])

            if not messages:
                skipped += 1
                continue

            text = format_messages(messages)

            if not text.strip():
                skipped += 1
                continue

            output.write(
                json.dumps(
                    {"text": text},
                    ensure_ascii=False,
                )
                + "\n"
            )

            written += 1

    # --------------------------------------------------------
    # RÉSUMÉ
    # --------------------------------------------------------

    size_mb = CLEAN_DATASET_FILE.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 70)
    print("PRÉPARATION TERMINÉE")
    print("=" * 70)
    print(f"\nFichier   : {CLEAN_DATASET_FILE}")
    print(f"Taille    : {size_mb:.2f} Mo")
    print(f"Documents : {written}")
    print(f"Scannés   : {scanned}")
    print(f"Ignorés   : {skipped}")

    return CLEAN_DATASET_FILE


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    download_and_prepare()