
import subprocess
from pathlib import Path


REPOSITORY_URL = "https://github.com/Synobotix/llm-dataset.git"
PROJECT_PATH = Path("/content/llm-dataset")


def clone_repository() -> None:
    """
    Clone le dépôt GitHub dans l'environnement Colab.
    """

    if PROJECT_PATH.exists():
        print(f"Dépôt déjà présent : {PROJECT_PATH}")
        return

    print("=" * 60)
    print("CLONAGE DU DÉPÔT")
    print("=" * 60)

    subprocess.run(
        [
            "git",
            "clone",
            REPOSITORY_URL,
            str(PROJECT_PATH),
        ],
        check=True,
    )

    print()
    print(f"Dépôt cloné : {PROJECT_PATH}")
