
import subprocess
from pathlib import Path


PROJECT_PATH = Path("/content/llm-dataset")


def verify_environment() -> None:
    """
    Vérifie Python, PyTorch, CUDA et le GPU
    dans l'environnement Poetry du projet.
    """

    print("=" * 60)
    print("VERIFICATION DE L'ENVIRONNEMENT")
    print("=" * 60)

    if not PROJECT_PATH.exists():
        raise FileNotFoundError(
            f"Projet introuvable : {PROJECT_PATH}"
        )

    command = [
        "poetry",
        "run",
        "python",
        "-c",
        (
            "import sys; "
            "import torch; "
            "print('Python :', sys.version.split()[0]); "
            "print('PyTorch :', torch.__version__); "
            "print('CUDA disponible :', torch.cuda.is_available()); "
            "print('CUDA version :', torch.version.cuda); "
            "print('GPU :', "
            "torch.cuda.get_device_name(0) "
            "if torch.cuda.is_available() else 'Aucun GPU')"
        ),
    ]

    result = subprocess.run(
        command,
        cwd=PROJECT_PATH,
        check=True,
        capture_output=True,
        text=True,
    )

    print(result.stdout)

    if not result.stdout:
        raise RuntimeError(
            "Aucune information d'environnement retournée."
        )

    print("ENVIRONNEMENT : OK")
