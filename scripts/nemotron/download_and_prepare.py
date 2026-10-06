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

# Config parmi les 18 disponibles :
#   code_de, code_es, code_fr, code_it, code_ja, code_zh
#   math_de, math_es, math_fr, math_it, math_ja, math_zh
#   stem_de, stem_es, stem_fr, stem_it, stem_ja, stem_zh
HF_CONFIG = "code_fr"

HF_SPLIT = "train"

# Nombre de documents à extraire (None = tout)
MAX_DOCUMENTS = 5000


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

        parts.append(
            f"<|{role}|>\n{content}"
        )

    return "\n\n".join(parts)


# ============================================================
# TÉLÉCHARGEMENT + CONVERSION
# ============================================================

def download_and_prepare():

    print("=" * 70)
    print("PRÉPARATION NEMOTRON")
    print("=" * 70)
    print(f"\nDataset : {HF_DATASET_ID}")
    print(f"Config  : {HF_CONFIG}")
    print(f"Split   : {HF_SPLIT}")
    print(f"Sortie  : {CLEAN_DATASET_FILE}")
    print(f"Max doc : {MAX_DOCUMENTS}")

    # --------------------------------------------------------
    # CHARGEMENT EN STREAMING
    # --------------------------------------------------------

    print("\nChargement en streaming...")

    ds = load_dataset(
        HF_DATASET_ID,
        HF_CONFIG,
        split=HF_SPLIT,
        streaming=True,
    )

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
    skipped = 0

    print(f"\nÉcriture vers : {CLEAN_DATASET_FILE}\n")

    with CLEAN_DATASET_FILE.open(
        "w",
        encoding="utf-8",
    ) as output:

        for i, example in enumerate(
            tqdm(
                ds,
                total=MAX_DOCUMENTS,
                desc="Traitement",
            )
        ):

            if (
                MAX_DOCUMENTS is not None
                and i >= MAX_DOCUMENTS
            ):
                break

            # ------------------------------------------------
            # Extraction des messages
            # ------------------------------------------------

            messages = example.get("messages", [])

            if not messages:
                skipped += 1
                continue

            # ------------------------------------------------
            # Conversion en texte
            # ------------------------------------------------

            text = format_messages(messages)

            if not text.strip():
                skipped += 1
                continue

            # ------------------------------------------------
            # Écriture au format attendu par split_dataset.py
            # ------------------------------------------------

            record = {"text": text}

            output.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            written += 1

    # --------------------------------------------------------
    # RÉSUMÉ
    # --------------------------------------------------------

    size_mb = (
        CLEAN_DATASET_FILE.stat().st_size
        / (1024 * 1024)
    )

    print("\n" + "=" * 70)
    print("PRÉPARATION TERMINÉE")
    print("=" * 70)
    print(f"\nFichier  : {CLEAN_DATASET_FILE}")
    print(f"Taille   : {size_mb:.2f} Mo")
    print(f"Documents: {written}")
    print(f"Ignorés  : {skipped}")

    # ========================================================
    # UPLOAD VERS HUGGING FACE
    # ========================================================

    try:

        from scripts.hub.hub_sync import (
            upload_file_to_hub,
        )

        print()
        print("=" * 70)
        print("UPLOAD VERS HUGGING FACE")
        print("=" * 70)

        upload_file_to_hub(
            CLEAN_DATASET_FILE,
            remote_path=(
                f"data/processed/{CLEAN_DATASET_FILE.name}"
            ),
        )

    except Exception as error:

        print(
            f"\n⚠️ Upload Nemotron brut échoué : "
            f"{error}"
        )

    return CLEAN_DATASET_FILE


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    download_and_prepare()