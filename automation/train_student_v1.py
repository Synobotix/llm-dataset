
import subprocess
from pathlib import Path


PROJECT_PATH = Path("/content/llm-dataset")


def train_student_v1() -> None:
    """
    Lance l'entraînement complet du Student v1.
    """

    print("=" * 60)
    print("ENTRAINEMENT STUDENT V1")
    print("=" * 60)

    if not PROJECT_PATH.exists():
        raise FileNotFoundError(
            f"Projet introuvable : {PROJECT_PATH}"
        )

    subprocess.run(
        [
            "poetry",
            "run",
            "python",
            "scripts/train.py",
        ],
        cwd=PROJECT_PATH,
        check=True,
        timeout=3600,
    )

    print()
    print("ENTRAINEMENT STUDENT V1 : OK")
