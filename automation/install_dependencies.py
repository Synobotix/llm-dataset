
import subprocess
from pathlib import Path


PROJECT_PATH = Path("/content/llm-dataset")


def install_dependencies() -> None:
    """
    Installe les dépendances du projet avec Poetry
    dans l'environnement Colab.
    """

    print("=" * 60)
    print("INSTALLATION DES DEPENDANCES")
    print("=" * 60)

    if not PROJECT_PATH.exists():
        raise FileNotFoundError(
            f"Projet introuvable : {PROJECT_PATH}"
        )

    subprocess.run(
        [
            "poetry",
            "install",
        ],
        cwd=PROJECT_PATH,
        check=True,
        timeout=1800,
    )

    print()
    print("Installation des dépendances terminée.")
    print("Poetry install : OK")
