# BitNet : principes & motivations

## Qu'est-ce que BitNet ?

**BitNet** (BitNet b1.58, etc.) est une approche qui remplace les poids des projections linéaires par des poids à **précision ultra-faible** (ternaires). Dans cette implémentation, les poids sont quantifiés dans `{-1, 0, +1}`.

L'idée : **réduire la précision des poids (et/ou activations)** tout en préservant les performances du modèle, pour diminuer coût mémoire/calcul et faciliter déploiement.

## Principe clé : BitLinear

Au lieu de `y = x @ W^T + b` avec `W` FP32, BitNet utilise :

1. **Poids latents FP32** (entraînables) — nécessaires pour la différentiabilité
2. **Quantification au forward** : `W_q = round(clamp(W / s, -1, 1)) ∈ {-1,0,1}` où `s = max(|W|)` (absmax)
3. **Forward ternaire** : `y = x @ W_q^T + b`

La quantification est non-différentiable → **Straight-Through Estimator (STE)** : gradient passe à travers l'arrondi (`∂L/∂W ≈ ∂L/∂W_q`).

## Différences vs Transformer classique

| Aspect | Classique | BitNet |
|---|---|---|
| Projections | `nn.Linear` (FP32/FP16/BF16) | `BitLinear` (poids ternaires au forward) |
| Précision poids | Continue | Discrète {-1,0,1} |
| Embedding | Classique | Classique (généralement) |
| LM Head | Classique | Classique (généralement) |
| Attention | Softmax + matmuls FP | Matmuls avec poids ternaires, softmax inchangé |
| Complexité calcul | FLOPs standards | Potentiellement réduite (opérations entières/ternaires) |

## Avantages attendus

- **Efficacité mémoire** : poids ternaires ~1.58 bits (vs 16/32). Moins de mémoire pour stocker poids.
- **Efficacité calcul** : remplace multiplications flottantes par additions/soustractions (pour valeurs {-1,0,1}).
- **Déploiement edge** : intéressant pour modèles plus légers/efficaces.
- **Recherche** : étudier compromis précision/efficacité.

## Compromis

- **Stabilité d'entraînement** : quantification introduit bruit, nécessite ajustements (lr, init, clipping)
- **Représentativité** : poids ternaires moins expressifs → besoin parfois d'échelle/architecte adaptée
- **Compatibilité checkpoints** : différences de structure (QKV fusionné vs séparé) à gérer

## Dans ce projet

- Implémentation **from-scratch** pédagogique (BitLinear + BitAttention + BitMLP + BitTransformer)
- **Q,K,V,O** ternarisés via BitLinear
- **MLP** via BitLinear
- **Embedding + LMHead** restent classiques
- Tests dédiés (`tests/bitnet/`) pour valider chaque composant
