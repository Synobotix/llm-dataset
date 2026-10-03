# Référence : BitNet - Vue d'ensemble

## Composants

| Module | Fichier | Rôle |
|---|---|---|
| BitLinear | `src/llm/bitnet_model/bitlinear.py` | Couche linéaire à poids ternaires {-1,0,1} |
| BitAttention | `src/llm/bitnet_model/bit_attention.py` | MHA avec projections BitLinear |
| BitMLP | `src/llm/bitnet_model/bit_mlp.py` | Feed-forward avec BitLinear |
| BitTransformerBlock | `src/llm/bitnet_model/bit_transformer_block.py` | Bloc pre-norm (Attention + MLP) |
| BitTransformer | `src/llm/bitnet_model/bit_transformer.py` | Empilement blocks + embedding + norm final |
| LMHead | `src/llm/bitnet_model/lm_head.py` | Projection vocabulaire (classique) |

## Config typique (train_bitnet.py via parameters)

- `vocab_size` (via tokenizer)
- `d_model = D_MODEL`
- `num_heads = NUM_HEADS`
- `hidden_dim = HIDDEN_DIM`
- `num_blocks = NUM_BLOCKS`
- `max_sequence_length = MAX_SEQUENCE_LENGTH`

## Points d'entrée

- Entraînement : `scripts/train_bitnet.py`
- Inférence : `scripts/bitnet/inference.py`
- Checkpoints : `checkpoint/optiminisation_bitnet/*.pt`

## Remarques

- Embedding classique (non ternarisé)
- LMHead classique (non ternarisé)
- QKV/O + MLP projections ternarisées
