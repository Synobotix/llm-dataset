
import subprocess
from pathlib import Path


PROJECT_PATH = Path("/content/llm-dataset")


def inference_student_v1() -> None:
    print("=" * 60)
    print("INFERENCE STUDENT V1")
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
            "scripts/inference.py",
        ],
        cwd=PROJECT_PATH,
        check=True,
    )

    print()
    print("INFERENCE STUDENT V1 : OK")
