"""
Test 6 — Entraîner BitNet sur 1 époque.

Objectif :
- Vérifier que l'entraînement se lance
- Produire un checkpoint et un résultat .md
- NE PAS uploader sur Hugging Face

Usage :
    poetry run python -m scripts.tests.test_avant_colab.06_test_train
"""

import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_SCRIPT = "scripts.bitnet.train_bitnet"

# ⚠️ Désactiver l'upload HF pour ce test
os.environ["DISABLE_HF_UPLOAD"] = "1"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TEST 6 — ENTRAÎNEMENT BITNET (1 époque)")
    print("=" * 70)

    print("\n⚠️ Ce test va lancer l'entraînement.")
    print("   Il NE va PAS uploader sur Hugging Face.")

    # --------------------------------------------------------
    # LANCEMENT
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DÉMARRAGE DE L'ENTRAÎNEMENT")
    print("=" * 70 + "\n")

    result = subprocess.run(
        [sys.executable, "-m", TRAIN_SCRIPT],
        env=os.environ.copy(),
        text=True,
    )

    if result.returncode != 0:
        print(f"\n❌ Entraînement échoué (code {result.returncode})")
        return False

    # --------------------------------------------------------
    # VÉRIFICATION DES SORTIES
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VÉRIFICATION DES SORTIES")
    print("=" * 70)

    checkpoint_dir = Path("checkpoints/bitnet")
    result_dir = Path("documentation/result/bitnet")

    # Checkpoints
    print(f"\n{checkpoint_dir} :")

    if checkpoint_dir.exists():

        checkpoints = sorted(checkpoint_dir.glob("*.pt"))

        if checkpoints:
            for c in checkpoints:
                size_mb = c.stat().st_size / (1024 * 1024)
                print(f"  ✅ {c.name} ({size_mb:.2f} Mo)")
        else:
            print("  ⚠️ Aucun checkpoint")

    else:
        print("  ❌ Dossier introuvable")

    # Résultats
    print(f"\n{result_dir} :")

    if result_dir.exists():

        results = sorted(result_dir.glob("*.md"))

        if results:
            for r in results:
                print(f"  ✅ {r.name}")
        else:
            print("  ⚠️ Aucun résultat")

    else:
        print("  ❌ Dossier introuvable")

    # --------------------------------------------------------
    # RÉSUMÉ
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TEST 6 TERMINÉ")
    print("=" * 70)

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)