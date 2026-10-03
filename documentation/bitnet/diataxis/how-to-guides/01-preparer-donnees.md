# How-to : Préparer les données C4

## 1. Préparer C4 brut

```bash
poetry run python scripts/prepare_C4.py
```

Génère/traite les données brutes C4 dans `data/raw/` (selon script).

## 2. Nettoyer C4

```bash
poetry run python scripts/clean_C4.py
```

Applique nettoyage (ponctuation, espaces, filtres) → `data/processed/`.

Inspecter si besoin :
```bash
poetry run python scripts/inspect_clean_C4.py
```

## 3. Split train/validation

```bash
poetry run python scripts/split_dataset.py
```

Crée `data/processed/train.jsonl` et `data/processed/validation.jsonl`.

## 4. Tokeniser

```bash
poetry run python scripts/tokenize_dataset.py
```

Produit `data/tokenized/train.jsonl` et `data/tokenized/validation.jsonl` + fichiers tokenizer dans `tokenizer/`.

## Vérification

Vérifiez existence :
```bash
ls data/tokenized/
ls tokenizer/
```
