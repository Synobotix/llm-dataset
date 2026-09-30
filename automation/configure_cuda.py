
from pathlib import Path


PROJECT_PATH = Path("/content/llm-dataset")
PYPROJECT_PATH = PROJECT_PATH / "pyproject.toml"

CPU_PYTORCH_URL = "https://download.pytorch.org/whl/cpu"
CUDA_PYTORCH_URL = "https://download.pytorch.org/whl/cu126"


def configure_cuda() -> None:
    """
    Configure la source PyTorch CUDA 12.6 dans le
    pyproject.toml de la copie Colab.
    """

    print("=" * 60)
    print("CONFIGURATION PYTORCH CUDA")
    print("=" * 60)

    if not PYPROJECT_PATH.exists():
        raise FileNotFoundError(
            f"pyproject.toml introuvable : {PYPROJECT_PATH}"
        )

    content = PYPROJECT_PATH.read_text(encoding="utf-8")

    if CUDA_PYTORCH_URL in content:
        print("PyTorch CUDA 12.6 est déjà configuré.")
        return

    if CPU_PYTORCH_URL not in content:
        raise RuntimeError(
            "La source PyTorch CPU n'a pas été trouvée "
            "dans pyproject.toml."
        )

    content = content.replace(
        CPU_PYTORCH_URL,
        CUDA_PYTORCH_URL,
    )

    PYPROJECT_PATH.write_text(
        content,
        encoding="utf-8",
    )

    print("Source PyTorch configurée : CUDA 12.6")
