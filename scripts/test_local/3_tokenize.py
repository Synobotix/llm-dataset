"""
3_tokenize.py

Tokenise train.jsonl/validation.jsonl en séquences de 32 tokens (réduit
pour le test ; sur le vrai corpus on garde BLOCK_SIZE=256).

Usage : poetry run python scripts/test_local/3_tokenize.py
"""
import json
from tokenizers import Tokenizer

BLOCK_SIZE = 32
CHUNK_SIZE = BLOCK_SIZE + 1


def process(tokenizer, input_path, output_path):
    all_ids = []
    n_docs = 0
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            ids = tokenizer.encode(json.loads(line)["text"]).ids
            all_ids.extend(ids)
            n_docs += 1

    n_seq = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(0, len(all_ids) - CHUNK_SIZE + 1, BLOCK_SIZE):
            chunk = all_ids[i : i + CHUNK_SIZE]
            if len(chunk) < CHUNK_SIZE:
                break
            f.write(json.dumps({"input_ids": chunk[:-1], "labels": chunk[1:]}) + "\n")
            n_seq += 1
    return n_docs, len(all_ids), n_seq


def main():
    tokenizer = Tokenizer.from_file("tokenizer/tokenizer_test.json")

    for name, inp, out in [
        ("Train", "data/processed/train.jsonl", "data/processed/train_tokens_test.jsonl"),
        ("Validation", "data/processed/validation.jsonl", "data/processed/validation_tokens_test.jsonl"),
    ]:
        n_docs, n_tokens, n_seq = process(tokenizer, inp, out)
        print(f"{name}: {n_docs} docs, {n_tokens} tokens, {n_seq} séquences")


if __name__ == "__main__":
    main()
