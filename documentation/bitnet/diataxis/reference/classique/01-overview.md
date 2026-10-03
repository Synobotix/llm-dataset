# Référence : Transformer classique - Vue d'ensemble

## Composants

| Module | Fichier | Rôle |
|---|---|---|
| TokenEmbedding | `src/llm/model/embedding.py` | Map token IDs → vecteurs d_model |
| PositionalEncoding | `src/llm/model/positional_encoding.py` | Ajoute info positionnelle |
| Attention | `src/llm/model/attention.py` | ScaledDotProductAttention + MultiHeadAttention |
| FeedForward | `src/llm/model/feed_forward.py` | MLP position-wise |
| TransformerBlock | `src/llm/model/transformer_block.py` | Bloc avec MHA + FFN + résidus + Norm |
| Transformer | `src/llm/model/transformer.py` | Modèle complet (empilement blocks) |

## Paramètres typiques (train.py)

- `vocab_size=994`
- `d_model=120` (EMBEDDING_DIM)
- `num_heads=3`
- `num_layers=2`
- `ffn_hidden_dim=480`
- `max_sequence_length=128`

## Points d'entrée

- Entraînement : `scripts/train.py`
- Inférence : `scripts/inference.py`
- Checkpoints : `checkpoints/student_v1/*.pt`
