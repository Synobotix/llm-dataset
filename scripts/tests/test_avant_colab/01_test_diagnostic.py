"""
Test 1 — Diagnostic du dataset Nemotron.

Affiche :
- Les colonnes disponibles
- Les vraies valeurs de 'domain' et 'language'
- La structure des messages

Usage :
    poetry run python -m scripts.tests.test_avant_colab.01_test_diagnostic
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_dataset, get_dataset_split_names


# ============================================================
# CONFIGURATION
# ============================================================

HF_DATASET_ID = "nvidia/Nemotron-SFT-Multilingual-v1"

# Les "splits" du dataset sont :
#   code_de, code_es, code_fr, code_it, code_ja, code_zh
#   math_de, math_es, math_fr, math_it, math_ja, math_zh
#   stem_de, stem_es, stem_fr, stem_it, stem_ja, stem_zh
TARGET_SPLIT = "code_fr"

NB_EXAMPLES_STRUCTURE = 5
NB_EXAMPLES_STATS = 200


# ============================================================
# DIAGNOSTIC
# ============================================================

def main():

    print("=" * 70)
    print("TEST 1 — DIAGNOSTIC NEMOTRON")
    print("=" * 70)

    # --------------------------------------------------------
    # LISTER LES SPLITS DISPONIBLES
    # --------------------------------------------------------

    print(f"\nDataset : {HF_DATASET_ID}")

    print("\nSplits disponibles :")

    try:

        splits = get_dataset_split_names(HF_DATASET_ID)

        for s in sorted(splits):
            marker = " ←" if s == TARGET_SPLIT else ""
            print(f"  - {s}{marker}")

    except Exception as e:

        print(f"  ⚠️ Impossible de lister : {e}")

    # --------------------------------------------------------
    # CHARGEMENT DU SPLIT CIBLE
    # --------------------------------------------------------

    print(f"\nChargement du split : {TARGET_SPLIT}")

    ds = load_dataset(
        HF_DATASET_ID,
        split=TARGET_SPLIT,
        streaming=True,
    )

    # --------------------------------------------------------
    # STRUCTURE
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(f"STRUCTURE ({NB_EXAMPLES_STRUCTURE} premiers exemples)")
    print("=" * 70)

    for i, example in enumerate(ds.take(NB_EXAMPLES_STRUCTURE)):

        print(f"\n--- Exemple {i + 1} ---")
        print(f"  domain   : {example.get('domain')!r}")
        print(f"  language : {example.get('language')!r}")
        print(f"  license  : {example.get('license')!r}")
        print(f"  keys     : {list(example.keys())}")

        messages = example.get("messages", [])
        print(f"  messages : {len(messages)} message(s)")

        if messages:
            first = messages[0]
            content = first.get("content", "")
            print(f"    role    : {first.get('role')!r}")
            print(f"    content : {content[:100]!r}...")

    # --------------------------------------------------------
    # STATISTIQUES
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(f"STATISTIQUES ({NB_EXAMPLES_STATS} premiers exemples)")
    print("=" * 70)

    domains = {}
    languages = {}

    for i, example in enumerate(ds.take(NB_EXAMPLES_STATS)):

        d = example.get("domain")
        l = example.get("language")

        domains[d] = domains.get(d, 0) + 1
        languages[l] = languages.get(l, 0) + 1

    print("\nDomaines :")

    for d, count in sorted(domains.items(), key=lambda x: -x[1]):
        print(f"  {d!r:<40} : {count}")

    print("\nLangues :")

    for l, count in sorted(languages.items(), key=lambda x: -x[1]):
        print(f"  {l!r:<40} : {count}")

    # --------------------------------------------------------
    # RÉCAPITULATIF
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RÉCAPITULATIF")
    print("=" * 70)
    print(f"\n✅ Split utilisé : {TARGET_SPLIT}")
    print("\n➡️ Notez ces valeurs pour configurer 02_test_download.py :")
    print(f"   TARGET_SPLIT = {TARGET_SPLIT!r}")

    return True


if __name__ == "__main__":
    main()