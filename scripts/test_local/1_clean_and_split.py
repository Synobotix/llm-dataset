"""
1_clean_and_split.py

Nettoie data/raw/sample_10docs.jsonl et split en train/validation (8/2).
Usage : poetry run python scripts/test_local/1_clean_and_split.py
"""
import json
import os
import re

URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
HTML_TAG_RE = re.compile(r"<[^>]+>")
MULTI_SPACE_RE = re.compile(r"[ \t]+")


def clean_text(text):
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


def main():
    os.makedirs("data/processed", exist_ok=True)

    docs = []
    with open("data/raw/sample_10docs.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            docs.append({"text": clean_text(r["text"])})

    with open("data/processed/c4_clean.jsonl", "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    train_docs, val_docs = docs[:8], docs[8:]

    with open("data/processed/train.jsonl", "w", encoding="utf-8") as f:
        for d in train_docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    with open("data/processed/validation.jsonl", "w", encoding="utf-8") as f:
        for d in val_docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    total_words = sum(len(d["text"].split()) for d in docs)
    print(f"Train: {len(train_docs)} docs, Validation: {len(val_docs)} docs")
    print(f"Total mots: {total_words}")


if __name__ == "__main__":
    main()
