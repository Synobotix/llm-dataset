# Référence : BitMLP

## Localisation

`src/llm/bitnet_model/bit_mlp.py`

## Structure

MLP position-wise classique, mais avec `BitLinear` :

```text
x (B,T,d_model)
  → BitLinear(d_model → hidden_dim)
  → activation (ReLU/GELU selon implémentation)
  → BitLinear(hidden_dim → d_model)
  → dropout (optionnel)
return x
```

## Paramètres

- `d_model`
- `hidden_dim` (FFN hidden size)
- `dropout`
- `activation` (selon code)

## Objectif

Introduire non-linéarité et augmenter capacité de représentation entre blocks. L'utilisation de BitLinear ici est la différence vs `FeedForward` classique.
