
import subprocess
from pathlib import Path


PROJECT_PATH = Path("/content/llm-dataset")
LOCK_PATH = PROJECT_PATH / "poetry.lock"


def regenerate_lock() -> None:
    """
    Supprime l'ancien poetry.lock et génère un nouveau
    lock correspondant à la configuration actuelle du projet.
    """

    print("=" * 60)
    print("REGENERATION DU POETRY LOCK")
    print("=" * 60)

    if not PROJECT_PATH.exists():
        raise FileNotFoundError(
            f"Projet introuvable : {PROJECT_PATH}"
        )

    if LOCK_PATH.exists():
        LOCK_PATH.unlink()
        print("Ancien poetry.lock supprimé.")
    else:
        print("Aucun poetry.lock existant.")

    subprocess.run(
        [
            "poetry",
            "lock",
        ],
        cwd=PROJECT_PATH,
        check=True,
    )

    if not LOCK_PATH.exists():
        raise RuntimeError(
            "Le nouveau poetry.lock n'a pas été créé."
        )

    print("Nouveau poetry.lock créé.")
    print("Poetry lock : OK")
