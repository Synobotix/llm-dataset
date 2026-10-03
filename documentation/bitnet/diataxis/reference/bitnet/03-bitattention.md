# Référence : BitAttention

## Localisation

`src/llm/bitnet_model/bit_attention.py`

## Rôle

Implémente le **Multi-Head Self-Attention** causal avec projections Q,K,V,O via `BitLinear`.

## Paramètres

- `d_model` : dimension cachée
- `num_heads` : nombre de têtes
- `dropout` (optionnel)
- `max_sequence_length` : utilisé pour mask causal
- `bias` (projections)

## Architecture

```text
x (B,T,d_model)
  ├─ Q = BitLinear(x)  (B,T,d_model)
  ├─ K = BitLinear(x)
  └─ V = BitLinear(x)

reshape to (B,h,T,dk), dk = d_model/h

attention_scores = Q @ K^T / sqrt(dk)  (B,h,T,T)
apply causal mask (upper triangle → -inf)
softmax → attention_weights
context = attention_weights @ V (B,h,T,dk)

concat heads → (B,T,d_model)
out = BitLinear(concat) → (B,T,d_model)
+ dropout (si activé)
```

## Mask causal

Pour un token à la position `t`, il ne doit voir que les positions `<= t`. Le mask remplit `scores[:, :, i, j] = -∞` pour `j > i`.

## Particularités BitNet

- **Q,K,V,O** utilisent `BitLinear` au lieu de `nn.Linear`.
- **Scaled dot-product** inchangé (opérations matmul/softmax standard).
- **Rescale sqrt(dk)** identique au Transformer classique.

## Tests

`scripts/tests/bitnet/test_bit_attention.py`
