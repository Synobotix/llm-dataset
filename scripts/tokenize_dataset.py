import json
from pathlib import Path

from tokenizers import Tokenizer


# ============================================================
# CONFIGURATION
# ============================================================

TOKENIZER_FILE = Path("tokenizer/tokenizer.json")

TRAIN_FILE = Path("data/processed/train.jsonl")
VALIDATION_FILE = Path("data/processed/validation.jsonl")

TRAIN_OUTPUT = Path("data/processed/train_tokens.jsonl")
VALIDATION_OUTPUT = Path("data/processed/validation_tokens.jsonl")

SEQ_LENGTH = 256


# ============================================================
# CHARGEMENT DU TOKENIZER
# ============================================================

def load_tokenizer():
    if not TOKENIZER_FILE.exists():
        raise FileNotFoundError(
            f"Tokenizer introuvable : {TOKENIZER_FILE}"
        )

    return Tokenizer.from_file(str(TOKENIZER_FILE))


# ============================================================
# LECTURE DES TEXTES
# ============================================================

def read_texts(file_path):
    with file_path.open("r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                document = json.loads(line)

            except json.JSONDecodeError:
                print(
                    f"Ligne JSON invalide ignorée : "
                    f"{file_path}:{line_number}"
                )
                continue

            text = document.get("text")

            if not text:
                continue

            yield text


# ============================================================
# TOKENISATION
# ============================================================

def tokenize_texts(tokenizer, texts):

    for text in texts:

        encoding = tokenizer.encode(text)

        yield encoding.ids


# ============================================================
# CREATION DES SEQUENCES
# ============================================================

def create_sequences(token_ids, seq_length):

    sequences = []

    for i in range(0, len(token_ids) - seq_length, seq_length):

        sequence = token_ids[i:i + seq_length + 1]

        if len(sequence) == seq_length + 1:
            sequences.append(sequence)

    return sequences


# ============================================================
# TRAITEMENT DU DATASET
# ============================================================

def process_dataset(tokenizer, input_file, output_file):

    print()
    print("=" * 60)
    print(f"Traitement : {input_file}")
    print("=" * 60)

    total_documents = 0
    total_tokens = 0
    total_sequences = 0

    with output_file.open("w", encoding="utf-8") as output:

        for token_ids in tokenize_texts(
            tokenizer,
            read_texts(input_file)
        ):

            total_documents += 1
            total_tokens += len(token_ids)

            sequences = create_sequences(
                token_ids,
                SEQ_LENGTH
            )

            for sequence in sequences:

                item = {
                    "input_ids": sequence[:-1],
                    "labels": sequence[1:]
                }

                output.write(
                    json.dumps(item)
                    + "\n"
                )

                total_sequences += 1

    print(f"Documents tokenisés : {total_documents}")
    print(f"Tokens produits     : {total_tokens}")
    print(f"Séquences créées    : {total_sequences}")
    print(f"Sortie              : {output_file}")


# ============================================================
# TEST D'UNE SEQUENCE
# ============================================================

def test_sequence(tokenizer, output_file):

    print()
    print("=" * 60)
    print("TEST D'UNE SEQUENCE")
    print("=" * 60)

    with output_file.open("r", encoding="utf-8") as file:

        line = file.readline()

        if not line:
            print("Aucune séquence trouvée.")
            return

        item = json.loads(line)

    input_ids = item["input_ids"]
    labels = item["labels"]

    print(f"Nombre d'input IDs : {len(input_ids)}")
    print(f"Nombre de labels   : {len(labels)}")

    print()
    print("Premiers input IDs :")
    print(input_ids[:20])

    print()
    print("Premiers labels :")
    print(labels[:20])

    print()
    print("Décodage input :")
    print(tokenizer.decode(input_ids[:20]))

    print()
    print("Décodage labels :")
    print(tokenizer.decode(labels[:20]))

    print()
    print("Vérification du décalage :")
    print(f"Dernier input  : {input_ids[-1]}")
    print(f"Premier label  : {labels[0]}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("TOKENISATION DU DATASET")
    print("=" * 60)

    tokenizer = load_tokenizer()

    print()
    print(f"Tokenizer chargé : {TOKENIZER_FILE}")
    print(f"Vocabulaire      : {tokenizer.get_vocab_size()}")
    print(f"Sequence length  : {SEQ_LENGTH}")

    process_dataset(
        tokenizer,
        TRAIN_FILE,
        TRAIN_OUTPUT
    )

    process_dataset(
        tokenizer,
        VALIDATION_FILE,
        VALIDATION_OUTPUT
    )

    test_sequence(
        tokenizer,
        TRAIN_OUTPUT
    )

    print()
    print("=" * 60)
    print("TOKENISATION TERMINÉE")
    print("=" * 60)


if __name__ == "__main__":
    main()