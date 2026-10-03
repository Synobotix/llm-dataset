# Pipeline de données C4

Flux complet depuis C4 brut jusqu'aux séquences tokenisées.

```text
data/raw/
  └─ (C4 brut)

  ↓  scripts/prepare_C4.py
data/processed/ (pré-traité)

  ↓  scripts/clean_C4.py  (cleaning.py, filtering.py)
data/processed/clean_*.jsonl (nettoyé/filtré)

  ↓  scripts/split_dataset.py
data/processed/train.jsonl
data/processed/validation.jsonl

  ↓  tokenizer/train_tokenizer.py
tokenizer/tokenizer.json (+ vocab)

  ↓  scripts/tokenize_dataset.py
data/tokenized/train.jsonl
data/tokenized/validation.jsonl
```

## Modules

- `src/llm/data/cleaning.py` : nettoyage texte
- `src/llm/data/filtering.py` : filtres (longueur, qualité...)
- `src/llm/data/deduplication.py` : déduplication
- `src/llm/data/dataset.py` : Dataset PyTorch (séquences)
- `src/llm/data/dataloader.py` : DataLoaders train/val

## Format

Fichiers `.jsonl` : un document/exemple par ligne. Après tokenisation, séquences prêtes pour next-token prediction (avec `MAX_SEQUENCE_LENGTH`).
