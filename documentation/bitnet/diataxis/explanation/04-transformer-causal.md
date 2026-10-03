# Transformer causal : explication

## Principe

Un Transformer causal (autoregressif) prédit le prochain token en se basant **uniquement sur le passé**. À l'entraînement comme à l'inférence, le modèle ne doit pas voir les tokens futurs.

## Masque causal

Dans l'Attention, pour calculer les scores entre query à pos i et key à pos j :

- Si `j <= i` : autorisé (contexte gauche)
- Si `j > i` : masqué (futur) → scores mis à -∞ avant softmax

Résultat : `attention_weights[i,j] = 0` pour `j>i`.

## Next-token prediction

Cible = `input_ids[1:]`, prédiction = `logits[:-1]`. Le modèle apprend `P(x_{t} | x_{1}..x_{t-1})`.

## Génération autoregressive

À l'inférence :
1. Prompt tokenisé → contexte initial
2. Pour k pas : calcul logits du dernier token → choisir prochain token (greedy ici) → l'ajouter au contexte
3. Répéter jusqu'à `MAX_NEW_TOKENS`
4. Décoder

Contexte tronqué à `max_sequence_length` pour limiter mémoire.
