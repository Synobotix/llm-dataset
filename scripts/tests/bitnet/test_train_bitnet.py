import sys
from pathlib import Path


# Ajouter la racine du projet au PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.bitnet.train_bitnet import main


# ============================================================
# TEST TRAIN BITNET
# ============================================================

def test_train_bitnet():

    print("=" * 70)
    print("TEST DE train_bitnet.py")
    print("=" * 70)

    try:

        main()

        print("\n" + "=" * 70)
        print("TEST RÉUSSI")
        print("=" * 70)

    except Exception as error:

        print("\n" + "=" * 70)
        print("TEST ÉCHOUÉ")
        print("=" * 70)

        print(
            f"\nErreur : "
            f"{type(error).__name__}"
        )

        print(
            f"Message : "
            f"{error}"
        )

        raise


if __name__ == "__main__":
    test_train_bitnet()