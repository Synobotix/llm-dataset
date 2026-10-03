# How-to : Entraîner le Transformer classique

## 1. Lancer l'entraînement

```bash
poetry run python scripts/train.py
```

## 2. Configuration

`scripts/train.py` définit :
- Dimensions (D_MODEL, NUM_HEADS, NUM_LAYERS, FFN_HIDDEN_DIM, MAX_SEQ_LEN)
- Hyperparams (EPOCHS, LEARNING_RATE, WEIGHT_DECAY, GRADIENT_CLIP)
- Checkpoints :
  ```python
  CHECKPOINT_TO_LOAD = "checkpoints/student_v1/student_v1_25docs.pt"
  CHECKPOINT_TO_SAVE = "student_v1_100docs.pt"
  ```
- DEVICE, SEED

## 3. Composants utilisés

- `Transformer` (`src/llm/model/transformer.py`)
- `create_train_dataloader()`, `create_validation_dataloader()` (`src/llm/data/dataloader.py`)
- `Trainer` (`src/llm/training/trainer.py`)
- `load_checkpoint()` (`src/llm/training/checkpoint.py`)

## 4. Reprendre vs from-scratch

- Si `CHECKPOINT_TO_LOAD` existe → reprise (restaure model + optimizer + scheduler state)
- Sinon → erreur/à adapter selon besoin

## 5. Sauvegarde

Le Trainer gère les checkpoints (selon logique dans `trainer.py`/`checkpoint.py`). Le script configure aussi le nom de sauvegarde.
