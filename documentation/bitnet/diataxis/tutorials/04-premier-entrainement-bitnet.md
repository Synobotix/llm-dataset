# Tutoriel : Premier entraînement du BitNet

## Objectif

Lancer un premier entraînement BitNet.

## Prérequis

- Données tokenisées OK
- Tokenizer OK

## Étapes

### 1. Configurer le checkpoint

Dans `scripts/train_bitnet.py`, décidez de reprendre ou non.

Pour un premier run from-scratch : commentez ou adaptez le chargement de `PREVIOUS_CHECKPOINT` (vérifiez s'il existe). Définissez `NEW_CHECKPOINT` (ex. `checkpoint/optiminisation_bitnet/bitnet_test.pt`).

### 2. Réduire la taille pour test

Pour un test rapide : baissez `EPOCHS` (ex. 2-3), `MAX_DOCUMENT`/batch si besoin (via config/parameters).

### 3. Lancer

```bash
poetry run python scripts/train_bitnet.py
```

### 4. Observer

Logs incluent loss/train-val, PPL, gradient norm. À la fin, statistiques des poids BitLinear (-1/0/+1).

### 5. Tester l'inférence

```bash
poetry run python scripts/bitnet/inference.py
```

Vérifiez que le `CHECKPOINT_FILE` pointe vers votre nouveau checkpoint.
