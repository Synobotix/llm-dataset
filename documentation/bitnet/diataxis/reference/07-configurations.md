# Référence : Configurations & paramètres

## Fichier config

`src/llm/config/parameters.py` (si présent) + overrides dans scripts.

## Paramètres communs

| Paramètre | Description | Où |
|---|---|---|
| `BATCH_SIZE` | Taille batch | config/scripts |
| `EPOCHS` | Nb époques | scripts |
| `LEARNING_RATE` | Taux apprentissage | scripts/config |
| `WEIGHT_DECAY` | Weight decay AdamW | scripts/config |
| `GRADIENT_CLIP` | Clipping gradient norm | scripts/config |
| `MAX_SEQUENCE_LENGTH` | Longueur max séquences | scripts/config |
| `TOKENIZER_FILE` | Chemin tokenizer.json | config |
| `MAX_DOCUMENT` | Limite documents (tests) | config |

## Modèle classique

`scripts/train.py` : `VOCAB_SIZE`, `D_MODEL`, `NUM_HEADS`, `NUM_LAYERS`, `FFN_HIDDEN_DIM`, `MAX_SEQ_LEN`

## Modèle BitNet

`scripts/train_bitnet.py` utilise `D_MODEL`, `NUM_HEADS`, `HIDDEN_DIM`, `NUM_BLOCKS` depuis config + `MAX_SEQUENCE_LENGTH`.

## Checkpoints

Classique : `checkpoints/student_v1/`
BitNet : `checkpoint/optiminisation_bitnet/`
