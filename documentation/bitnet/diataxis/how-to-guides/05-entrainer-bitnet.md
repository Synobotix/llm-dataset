# How-to : Entraîner le modèle BitNet

Ce guide explique comment entraîner (ou reprendre l'entraînement) du **Transformer BitNet**.

## 1. Pré-requis

- Données tokenisées prêtes : `data/tokenized/train.jsonl` et `data/tokenized/validation.jsonl` (créées via les étapes data)
- Tokenizer existant : `tokenizer/tokenizer.json`
- Environnement prêt

## 2. Vérifier les chemins

`scripts/train_bitnet.py` définit :

```python
PREVIOUS_CHECKPOINT = Path("checkpoint/optiminisation_bitnet/bitnet_50docs1.pt")
NEW_CHECKPOINT = Path("checkpoint/optiminisation_bitnet/bitnet_200docs1.pt")
TRAIN_DATASET_FILE = Path("data/tokenized/train.jsonl")
```

Adaptez selon votre besoin (reprise vs entraînement from-scratch). Pour partir de zéro, mettez `PREVIOUS_CHECKPOINT = None` ou commentez le chargement (le script tente de charger s'il existe).

## 3. Lancer l'entraînement

Depuis la racine du projet :

```bash
poetry run python scripts/train_bitnet.py
```

## 4. Ce que fait le script

1. Charge le tokenizer (`tokenizers.Tokenizer`)
2. Charge/instancie `BitTransformer` + `LMHead`
3. Tente de charger `PREVIOUS_CHECKPOINT` (state_dict transformer + lm_head + méta)
4. Crée DataLoader pour train/validation
5. Boucle d'entraînement (AdamW, CrossEntropy, gradient clipping, scheduler)
6. Calcule/train loss, validation loss, perplexité (PPL)
7. Sauvegarde `NEW_CHECKPOINT` à la fin/points de sauvegarde

## 5. Sur GPU/CPU

Le script détecte CUDA :

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## 6. Configurer les hyperparamètres

Importés depuis `llm.config.parameters` (ex. `D_MODEL`, `NUM_HEADS`, `HIDDEN_DIM`, `NUM_BLOCKS`, `EPOCHS`, `LEARNING_RATE`, `BATCH_SIZE`, `MAX_SEQUENCE_LENGTH`, ...). Modifiez ces valeurs ou surchargez-les dans `train_bitnet.py`.

## 7. Checkpoints

Les checkpoints BitNet sont sauvegardés sous `checkpoint/optiminisation_bitnet/`. Un checkpoint contient généralement :
- `transformer_state_dict`
- `lm_head_state_dict`
- métadonnées (epoch, global_step, loss, config...)

## 8. Astuces

- Pour reprendre : laissez `PREVIOUS_CHECKPOINT` pointant vers un `.pt` existant.
- Pour partir de zéro : supprimez/ignorez le chargement du checkpoint précédent.
- Sur petit dataset (tests), réduisez `EPOCHS`/augmentez logs pour vérifier la cohérence.

## Prochaines étapes

- [Lancer l'inférence (BitNet)](./07-inference-bitnet.md)
- [Évaluer un modèle](./09-evaluer-modele.md)
