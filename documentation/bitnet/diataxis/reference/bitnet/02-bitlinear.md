# Référence : BitLinear

## Objectif

`BitLinear` remplace `nn.Linear` dans BitNet. Au lieu d'utiliser des poids FP32 directement, il maintient des **poids latents entraînables** en FP32 et quantifie ces poids en valeurs ternaires `{ -1, 0, +1 }` lors du forward.

## Localisation

`src/llm/bitnet_model/bitlinear.py`

## Signature (conceptuelle)

```python
class BitLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        ...
    def quantize_weights(self) -> torch.Tensor:
        ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
```

## Comportement

### Initialisation
- Crée `self.weight` (poids latents, FP32, entraînable) de forme cohérente avec `nn.Linear` (typiquement `(out_features, in_features)`)
- Optionnellement `self.bias`

### Quantification des poids
Au forward (ou via `quantize_weights()`), on calcule :

1. **Échelle par absmax** (approche commune BitNet)
   - `s = max(|W_latent|)` ou moyenne/absmax selon implémentation précise
2. **Normalisation + clamp**
   - `W_scaled = clamp(W_latent / (s + eps), -1, 1)`
3. **Arrondi** (round-to-nearest) → `W_q ∈ {-1,0,1}`
4. **Démagnétisation / ajustements** éventuels selon variantes (non visible ici)

Le code du projet utilise cette logique au forward pour obtenir `W_b` quantifié.

### Forward
```text
x ──► matmul(x, W_b^T) + bias (si présent)
```

`W_b` est le tenseur ternaire calculé à partir des poids latents.

## Points importants

- **Poids latents entraînables** : seule cette copie FP32 reçoit les gradients via `backward()`. L'opération d'arrondi est non-différentiable → **straight-through estimator (STE)** est implicitement utilisé (grad passe à travers l'arrondi).
- **Weights ternaires** : `{-1,0,1}` permet des gains mémoire/calcul selon implémentations (même si CPU/GPU ici, c'est l'esprit BitNet).
- **Embedding/LMHead** : dans cette implémentation, **TokenEmbedding** et **LMHead** restent classiques (pas ternarisés). Seules les projections dans Attention/MLP passent par BitLinear.
- **dtype** : conserver cohérence FP32 pour stabilité.

## Utilisation

Dans `BitAttention` et `BitMLP` :

- Q,K,V,O projections : `BitLinear(d_model, d_model)` (avec head-splitting)
- MLP up/down projections : `BitLinear` dans `bit_mlp.py`

## Tests associés

`scripts/tests/bitnet/test_bitlinear.py` vérifie comportement/forme attendue.
