import json
import random
from pathlib import Path

from llm.tokenizer.tokenizer import load_tokenizer
from llm.config.parameters import (
    BLOCK_SIZE,
    TRAIN_RATIO,
    RANDOM_SEED,
)

DISTILLED_FILE = Path("data/distilled/distilled.jsonl")
DISTILLED_TRAIN_FILE = Path("data/distilled/train.jsonl")
DISTILLED_VALIDATION_FILE = Path("data/distilled/validation.jsonl")

DISTILLED_TOKENIZED_TRAIN = Path("data/tokenized/distilled_train.jsonl")
DISTILLED_TOKENIZED_VALIDATION = Path("data/tokenized/distilled_validation.jsonl")


def read_distilled_responses(file_path: Path):
    """
    Lit le fichier distilled.jsonl et extrait les réponses.
    """
    responses = []

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                example = json.loads(line)
            except json.JSONDecodeError as error:
                print(f"[WARNING] Ligne {line_number} ignorée : {error}")
                continue

            response = example.get("response")
            if not isinstance(response, str):
                continue

            response = response.strip()
            if not response:
                continue

            responses.append(response)

    return responses


def split_data(responses, train_ratio=TRAIN_RATIO, seed=RANDOM_SEED):
    """
    Sépare les données en train/validation de façon reproductible.
    """
    random.seed(seed)
    random.shuffle(responses)

    split_index = int(len(responses) * train_ratio)
    train_data = responses[:split_index]
    validation_data = responses[split_index:]

    return train_data, validation_data


def save_jsonl(data, output_file: Path):
    """
    Sauvegarde les données au format JSONL (un texte par ligne).
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as file:
        for text in data:
            file.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")


def tokenize_text(tokenizer, text: str):
    encoding = tokenizer.encode(text)
    return encoding.ids


def create_sequences(token_ids):
    sequence_length = BLOCK_SIZE + 1
    sequences = []

    for start in range(0, len(token_ids) - sequence_length + 1, BLOCK_SIZE):
        sequence = token_ids[start : start + sequence_length]
        if len(sequence) != sequence_length:
            continue

        input_ids = sequence[:-1]
        labels = sequence[1:]

        if len(input_ids) != BLOCK_SIZE or len(labels) != BLOCK_SIZE:
            continue

        sequences.append({"input_ids": input_ids, "labels": labels})

    return sequences


def process_dataset(tokenizer, input_file: Path, output_file: Path):
    print(f"\n{'=' * 60}")
    print(f"TRAITEMENT : {input_file}")
    print(f"{'=' * 60}")

    with input_file.open("r", encoding="utf-8") as file:
        documents = []
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
                text = doc.get("text")
                if isinstance(text, str) and text.strip():
                    documents.append(text.strip())
            except json.JSONDecodeError:
                continue

    print(f"Documents lus : {len(documents)}")

    total_tokens = 0
    total_sequences = 0

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as output:
        for text in documents:
            token_ids = tokenize_text(tokenizer, text)
            total_tokens += len(token_ids)
            sequences = create_sequences(token_ids)

            for sequence in sequences:
                output.write(json.dumps(sequence, ensure_ascii=False) + "\n")
                total_sequences += 1

    print(f"Tokens totaux      : {total_tokens:,}")
    print(f"Séquences créées   : {total_sequences:,}")
    print(f"Block size          : {BLOCK_SIZE}")
    print(f"Longueur input_ids  : {BLOCK_SIZE}")
    print(f"Longueur labels     : {BLOCK_SIZE}")
    print(f"Fichier             : {output_file}")

    return total_sequences


def verify_dataset(file_path: Path):
    print(f"\n{'=' * 60}")
    print(f"VÉRIFICATION : {file_path}")
    print(f"{'=' * 60}")

    number_of_sequences = 0
    invalid_sequences = 0

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                print(f"[ERROR] Ligne {line_number} invalide.")
                invalid_sequences += 1
                continue

            input_ids = sample.get("input_ids")
            labels = sample.get("labels")
            number_of_sequences += 1

            if not isinstance(input_ids, list) or len(input_ids) != BLOCK_SIZE:
                print(f"[ERROR] Ligne {line_number} : input_ids invalide")
                invalid_sequences += 1

            if not isinstance(labels, list) or len(labels) != BLOCK_SIZE:
                print(f"[ERROR] Ligne {line_number} : labels invalide")
                invalid_sequences += 1

            if (
                isinstance(input_ids, list)
                and isinstance(labels, list)
                and len(input_ids) == BLOCK_SIZE
                and len(labels) == BLOCK_SIZE
            ):
                if input_ids[1:] != labels[:-1]:
                    print(f"[ERROR] Ligne {line_number} : input_ids / labels mal décalés")
                    invalid_sequences += 1

    print(f"\nSéquences vérifiées : {number_of_sequences:,}")
    print(f"Erreurs              : {invalid_sequences:,}")

    if invalid_sequences == 0:
        print("✓ Dataset valide")
    else:
        raise ValueError("Le dataset contient des séquences invalides.")


def main():
    print("=" * 60)
    print("PRÉPARATION DES DONNÉES DE DISTILLATION")
    print("=" * 60)

    if not DISTILLED_FILE.exists():
        raise FileNotFoundError(f"Fichier distillé introuvable : {DISTILLED_FILE}")

    print("\n1. Extraction des réponses du dataset distillé...")
    responses = read_distilled_responses(DISTILLED_FILE)
    print(f"   Réponses extraites : {len(responses)}")

    print("\n2. Split train/validation...")
    train_data, validation_data = split_data(responses)
    print(f"   Train      : {len(train_data)}")
    print(f"   Validation : {len(validation_data)}")

    print("\n3. Sauvegarde des fichiers bruts...")
    save_jsonl(train_data, DISTILLED_TRAIN_FILE)
    save_jsonl(validation_data, DISTILLED_VALIDATION_FILE)
    print(f"   → {DISTILLED_TRAIN_FILE}")
    print(f"   → {DISTILLED_VALIDATION_FILE}")

    print("\n4. Chargement du tokenizer...")
    tokenizer = load_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    print(f"   Vocabulaire : {vocab_size}")

    print("\n5. Tokenisation train...")
    train_sequences = process_dataset(
        tokenizer=tokenizer,
        input_file=DISTILLED_TRAIN_FILE,
        output_file=DISTILLED_TOKENIZED_TRAIN,
    )

    print("\n6. Tokenisation validation...")
    validation_sequences = process_dataset(
        tokenizer=tokenizer,
        input_file=DISTILLED_VALIDATION_FILE,
        output_file=DISTILLED_TOKENIZED_VALIDATION,
    )

    print("\n7. Vérification train...")
    verify_dataset(DISTILLED_TOKENIZED_TRAIN)

    print("\n8. Vérification validation...")
    verify_dataset(DISTILLED_TOKENIZED_VALIDATION)

    print("\n" + "=" * 60)
    print("PRÉPARATION TERMINÉE")
    print("=" * 60)
    print(f"\nVocabulaire : {vocab_size}")
    print(f"Train      : {train_sequences:,} séquences")
    print(f"Validation : {validation_sequences:,} séquences")
    print(f"\nFichiers générés :")
    print(f"  → {DISTILLED_TOKENIZED_TRAIN}")
    print(f"  → {DISTILLED_TOKENIZED_VALIDATION}")


if __name__ == "__main__":
    main()