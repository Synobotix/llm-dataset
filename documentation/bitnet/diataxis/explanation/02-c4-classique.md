# Architecture C4 - Transformer classique

Description de l'architecture du **Transformer classique** (GPT-like) selon le modèle C4.

## C1 - Context

Transformer causal from-scratch utilisant `nn.Linear` standard. Destiné à l'apprentissage/expérimentation du mécanisme Transformer.

**Acteurs** : développeurs, apprenants
**Données** : C4 (`data/*`), tokenizer `tokenizer/tokenizer.json`
**Artefacts** : checkpoints `checkpoints/student_v1/*.pt`
**Technos** : PyTorch

## C2 - Containers

```text
User
├─ scripts/train.py           (Entraînement)
├─ scripts/inference.py       (Inférence)
├─ src/llm/model/             (Modèle classique)
├─ src/llm/training/          (Trainer, checkpoint)
├─ src/llm/inference/generate.py (Helpers)
└─ Artifacts (tokenizer, data, checkpoints)
```

## C3 - Components

### model/

| Composant | Rôle | Fichier |
|---|---|---|
| `TokenEmbedding` | Embedding token → vecteur | `embedding.py` |
| `PositionalEncoding` | Encodage positionnel | `positional_encoding.py` |
| `ScaledDotProductAttention` | Attention scalaire | `attention.py` |
| `MultiHeadAttention` | MHA | `attention.py` |
| `FeedForward` | MLP 2 couches (Linear→ReLU/GELU→Linear) | `feed_forward.py` |
| `TransformerBlock` | Bloc : Norm + MHA (résidu) + Norm + FFN (résidu) | `transformer_block.py` |
| `Transformer` | Empilement blocks + Embedding + PE + Norm final | `transformer.py` |

### training/

| Composant | Rôle |
|---|---|
| `Trainer` | Boucle d'entraînement (train/val, loss, grad clip, scheduler, checkpoints) |
| `checkpoint.py` | Sauvegarde/chargement state_dict + méta |
| `loss.py`, `optimizer.py`, `scheduler.py` | Helpers |

## C4 - Code (flux)

```text
input_ids (B,T)
  → TokenEmbedding + PositionalEncoding → x (B,T,d_model)
  → TransformerBlock×N
      Pre-norm + MHA (causal mask) + Residual
      Pre-norm + FFN + Residual
  → LayerNorm final → hidden (B,T,d_model)
  → Linear (projection vocab) → logits (B,T,vocab_size)
```

MHA : Q,K,V = Linear(X), split heads, SDPA + causal mask, concat, Linear(O).
