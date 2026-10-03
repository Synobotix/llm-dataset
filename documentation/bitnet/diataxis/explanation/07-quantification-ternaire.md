# Quantification ternaire (-1,0,1)

## Définition

La quantification ternaire transforme des poids continus FP32 en 3 valeurs : `{-1, 0, +1}`.

Forme générale (absmax) :

```text
s = max(|W|)
W_scaled = clamp(W / s, -1, 1)
W_q = round(W_scaled)  ∈ {-1,0,1}
```

On obtient souvent une distribution ~33%/33%/-1,0,1 (dépend des poids). Le projet affiche ce comptage après entraînement (ex. dans logs d'entraînement BitNet).

## Pourquoi ternaire ?

- **Densité/sparsité contrôlée** : le `0` permet de mettre certains poids à zéro (sparsité partielle)
- **Efficacité arithmétique** : `y = x @ W_q^T` se décompose en accumulations de +x_i, -x_i, 0 (pas de multiplications flottantes coûteuses)
- **Compromis précision/efficacité** : 1.58 bits effectifs (entropie), entre binaire (1-bit) et 2-bit

## Straight-Through Estimator (STE)

L'arrondi `round()` est non-différentiable. Pendant le backprop :

```text
∂L/∂W_q  (flux arrière sur poids quantifiés)
  → propagé directement à ∂L/∂W (STE)
```

Concrètement : on utilise `W_q` au forward, mais on met à jour `W` (poids latents) avec les gradients calculés comme si c'était linéaire. C'est la pratique standard pour quantification apprentissage (Quantization-Aware Training).

## Impact sur l'entraînement

- **Bruit d'approximation** : quantification ajoute du bruit → peut ralentir/converger différemment
- **Gradient clipping** utile (présent dans train_bitnet)
- **Learning rate** souvent à ajuster vs classique
- **Stabilité** : absmax + clamp aident

## Vérification pratique

Les tests BitNet (`test_bitlinear.py`, `test_loss_backprop.py`) vérifient formes/comportements. Les logs affichent répartition -1/0/+1 des poids BitLinear finaux — indicateur utile de la quantification effective.
