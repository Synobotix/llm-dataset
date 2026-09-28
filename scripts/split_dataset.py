import json
import random
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = Path("data/processed/c4_clean.jsonl")

TRAIN_FILE = Path("data/processed/train.jsonl")
VALIDATION_FILE = Path("data/processed/validation.jsonl")

TRAIN_RATIO = 0.90
SEED = 42


# ============================================================
# Lecture du dataset
# ============================================================

def load_dataset(file_path: Path):
    dataset = []

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                document = json.loads(line)
                dataset.append(document)

            except json.JSONDecodeError as error:
                print(
                    f"[WARNING] Ligne {line_number} ignorée : "
                    f"{error}"
                )

    return dataset


# ============================================================
# Split
# ============================================================

def split_dataset(dataset):
    random_generator = random.Random(SEED)

    # Copie pour ne pas modifier le dataset original
    shuffled_dataset = dataset.copy()

    # Mélange reproductible
    random_generator.shuffle(shuffled_dataset)

    train_size = int(len(shuffled_dataset) * TRAIN_RATIO)

    train_dataset = shuffled_dataset[:train_size]
    validation_dataset = shuffled_dataset[train_size:]

    return train_dataset, validation_dataset


# ============================================================
# Sauvegarde
# ============================================================

def save_dataset(dataset, file_path: Path):
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as file:
        for document in dataset:
            file.write(
                json.dumps(
                    document,
                    ensure_ascii=False
                ) + "\n"
            )


# ============================================================
# Programme principal
# ============================================================

def main():
    print("=" * 60)
    print("SPLIT DU DATASET C4")
    print("=" * 60)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {INPUT_FILE}"
        )

    print(f"\nDataset source : {INPUT_FILE}")

    # Chargement
    dataset = load_dataset(INPUT_FILE)

    print(f"Documents chargés : {len(dataset)}")

    if len(dataset) < 2:
        raise ValueError(
            "Le dataset doit contenir au moins 2 documents."
        )

    # Split
    train_dataset, validation_dataset = split_dataset(dataset)

    # Sauvegarde
    save_dataset(train_dataset, TRAIN_FILE)
    save_dataset(validation_dataset, VALIDATION_FILE)

    # Résultats
    print("\nRésultat du split :")
    print(f"  Train      : {len(train_dataset)} documents")
    print(f"  Validation : {len(validation_dataset)} documents")

    print("\nFichiers créés :")
    print(f"  → {TRAIN_FILE}")
    print(f"  → {VALIDATION_FILE}")

    print("\nConfiguration :")
    print(f"  Train ratio : {TRAIN_RATIO:.0%}")
    print(f"  Validation  : {(1 - TRAIN_RATIO):.0%}")
    print(f"  Seed        : {SEED}")

    print("\n" + "=" * 60)
    print("SPLIT TERMINÉ")
    print("=" * 60)


if __name__ == "__main__":
    main()