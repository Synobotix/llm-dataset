
from pathlib import Path


PROJECT_PATH = Path("/content/llm-dataset")

TRAIN_DATA_PATH = PROJECT_PATH / "data/processed/train_tokens.jsonl"
VALIDATION_DATA_PATH = PROJECT_PATH / "data/processed/validation_tokens.jsonl"


def verify_dataset() -> None:
    """
    Vérifie la présence des datasets tokenisés
    nécessaires à l'entraînement Student v1.
    """

    print("=" * 60)
    print("VERIFICATION DU DATASET")
    print("=" * 60)

    datasets = [
        TRAIN_DATA_PATH,
        VALIDATION_DATA_PATH,
    ]

    for dataset_path in datasets:

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset introuvable : {dataset_path}"
            )

        if not dataset_path.is_file():
            raise RuntimeError(
                f"Le chemin n'est pas un fichier : {dataset_path}"
            )

        size_mb = dataset_path.stat().st_size / (1024 * 1024)

        print(
            f"{dataset_path.name} : "
            f"{size_mb:.2f} MB"
        )

    print()
    print("DATASET : OK")
