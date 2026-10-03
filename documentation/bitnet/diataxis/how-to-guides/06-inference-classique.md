# How-to : Lancer l'inférence (Transformer classique)

## 1. Lancer

```bash
poetry run python scripts/inference.py
```

## 2. Configurer

Dans `scripts/inference.py` :
- `CHECKPOINT_PATH = Path("checkpoints/student_v1/student_v1_100docs.pt")`
- `MAX_NEW_TOKENS = 50`
- `VOCAB_SIZE`, dimensions modèle (doivent correspondre au checkpoint)
- `TRAIN_PROMPTS`, `RANDOM_PROMPTS`

## 3. Vérifier cohérence checkpoint

Le checkpoint doit contenir `model_state_dict` (et éventuellement `global_step`). Les dimensions (EMBEDDING_DIM, NUM_HEADS, NUM_LAYERS, FFN_HIDDEN_DIM, MAX_SEQUENCE_LENGTH) doivent matcher le modèle sauvé.

## 4. Dépannage

**Checkpoint introuvable** : entraîner ou copier un `.pt` dans `checkpoints/student_v1/`.

**Mauvais device** : script auto-détecte CUDA/CPU via `DEVICE`.
