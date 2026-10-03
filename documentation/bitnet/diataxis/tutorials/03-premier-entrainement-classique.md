# Tutoriel : Premier entraînement du Transformer classique

## Objectif

Lancer un premier entraînement du modèle classique sur le dataset tokenisé.

## Prérequis

- Données tokenisées : `data/tokenized/train.jsonl`, `data/tokenized/validation.jsonl`
- Tokenizer : `tokenizer/tokenizer.json`

## Étapes

### 1. Vérifier config minimale

Dans `scripts/train.py`, vérifiez `EPOCHS`, `BATCH_SIZE`, `DEVICE`. Pour un premier test, `EPOCHS=2-5` suffit.

### 2. Lancer

```bash
poetry run python scripts/train.py
```

### 3. Suivre les logs

Vous verrez : création dataloaders, nb paramètres, chargement checkpoint (si présent), puis par epoch : train/val loss, perplexité, gradient norm, learning rate.

### 4. Résultat

Checkpoint sauvegardé selon logique trainer/script (ex. `student_v1_100docs.pt` configuré). Vérifiez dans `checkpoints/student_v1/` ou dossier indiqué.

### 5. Tester l'inférence

```bash
poetry run python scripts/inference.py
```
