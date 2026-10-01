import json
import random
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("data/processed/c4_clean.jsonl")

TRAIN_FILE = Path("data/processed/train.jsonl")
VALIDATION_FILE = Path("data/processed/validation.jsonl")

# Premier palier
TOTAL_DOCUMENTS = 100

# Pour le premier test :
# 80 % entraînement / 20 % validation
VALIDATION_RATIO = 0.2

# Reproductibilité
SEED = 42


# ============================================================
# LECTURE DU DATASET
# ============================================================

def load_jsonl(path: Path) -> list[dict]:
    documents = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                document = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSON invalide à la ligne {line_number}"
                ) from error

            documents.append(document)

    return documents


# ============================================================
# SAUVEGARDE JSONL
# ============================================================

def save_jsonl(path: Path, documents: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for document in documents:
            file.write(
                json.dumps(
                    document,
                    ensure_ascii=False
                ) + "\n"
            )


# ============================================================
# SPLIT
# ============================================================

def split_dataset(
    documents: list[dict],
    total_documents: int,
    validation_ratio: float,
    seed: int
) -> tuple[list[dict], list[dict]]:

    if total_documents > len(documents):
        raise ValueError(
            f"Le dataset contient seulement {len(documents)} documents, "
            f"mais {total_documents} sont demandés."
        )

    # On prend un sous-ensemble
    selected_documents = documents[:total_documents].copy()

    # Mélange reproductible
    random_generator = random.Random(seed)
    random_generator.shuffle(selected_documents)

    validation_size = max(
        1,
        round(total_documents * validation_ratio)
    )

    validation_documents = selected_documents[:validation_size]
    train_documents = selected_documents[validation_size:]

    return train_documents, validation_documents


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("SPLIT DU DATASET")
    print("=" * 60)

    print(f"Dataset source : {INPUT_FILE}")
    print(f"Documents demandés : {TOTAL_DOCUMENTS}")

    documents = load_jsonl(INPUT_FILE)

    print(f"Documents disponibles : {len(documents)}")

    train_documents, validation_documents = split_dataset(
        documents=documents,
        total_documents=TOTAL_DOCUMENTS,
        validation_ratio=VALIDATION_RATIO,
        seed=SEED
    )

    save_jsonl(TRAIN_FILE, train_documents)
    save_jsonl(VALIDATION_FILE, validation_documents)

    print()
    print("Résultat :")
    print(f"  Train       : {len(train_documents)} documents")
    print(f"  Validation  : {len(validation_documents)} documents")

    print()
    print(f"Train sauvegardé       : {TRAIN_FILE}")
    print(f"Validation sauvegardée : {VALIDATION_FILE}")

    print("=" * 60)


if __name__ == "__main__":
    main()