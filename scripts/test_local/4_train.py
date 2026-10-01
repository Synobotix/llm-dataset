"""
4_train.py

Entraîne le CausalTransformer sur le corpus de test (10 documents).
Config réduite : embedding_dim=64, 2 couches, vocab=300, séquences de 32.

Usage : poetry run python scripts/test_local/4_train.py
"""
import json
import sys
sys.path.insert(0, "src")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from llm.model.transformer import CausalTransformer

VOCAB_SIZE = 300
EMBEDDING_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 2
FF_HIDDEN_DIM = 128
MAX_SEQ_LEN = 32

BATCH_SIZE = 4
EPOCHS = 300
LR = 3e-4


class TokenDataset(Dataset):
    def __init__(self, path):
        self.examples = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                self.examples.append(json.loads(line))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return (
            torch.tensor(ex["input_ids"], dtype=torch.long),
            torch.tensor(ex["labels"], dtype=torch.long),
        )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_ds = TokenDataset("data/processed/train_tokens_test.jsonl")
    val_ds = TokenDataset("data/processed/validation_tokens_test.jsonl")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = CausalTransformer(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ff_hidden_dim=FF_HIDDEN_DIM,
        max_sequence_length=MAX_SEQ_LEN,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Nombre de paramètres : {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for input_ids, labels in train_loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            logits = model(input_ids)
            loss = criterion(logits.view(-1, VOCAB_SIZE), labels.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 50 == 0 or epoch == 1:
            avg_train_loss = total_loss / len(train_loader)
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for input_ids, labels in val_loader:
                    input_ids, labels = input_ids.to(device), labels.to(device)
                    logits = model(input_ids)
                    loss = criterion(logits.view(-1, VOCAB_SIZE), labels.view(-1))
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch:4d} | train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f}")

    torch.save(model.state_dict(), "checkpoints/student_test.pt")
    print("\nCheckpoint sauvegardé : checkpoints/student_test.pt")


if __name__ == "__main__":
    main()
