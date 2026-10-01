"""
2_train_tokenizer.py

Entraîne un tokenizer BPE avec un vocab réduit (300), adapté à un corpus
de test de 10 documents. Sur le vrai corpus (5000 docs), on garde VOCAB_SIZE=16000.

Usage : poetry run python scripts/test_local/2_train_tokenizer.py
"""
import json
import os
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

VOCAB_SIZE = 300
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def iter_texts():
    with open("data/processed/train.jsonl", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)["text"]


def main():
    os.makedirs("tokenizer", exist_ok=True)

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE, special_tokens=SPECIAL_TOKENS, show_progress=True
    )
    tokenizer.train_from_iterator(iter_texts(), trainer=trainer)

    tokenizer.save("tokenizer/tokenizer_test.json")
    print("Taille vocab réelle:", tokenizer.get_vocab_size())


if __name__ == "__main__":
    main()
