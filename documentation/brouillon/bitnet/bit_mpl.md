Ce qui se passe réellement

Supposons :

d_model = 256
hidden_dim = 680

et une entrée :

[batch, sequence, 256]
Première BitLinear
[batch, sequence, 256]
              │
              ▼
          BitLinear
              │
              ▼
[batch, sequence, 680]

Les poids de cette couche sont ternarisés :

W1 ∈ {-1, 0, +1} × scale
GELU

Ensuite :

[batch, sequence, 680]
              │
              ▼
             GELU
              │
              ▼
[batch, sequence, 680]

GELU n'est pas une BitLinear.

Elle reste donc une opération classique.

Deuxième BitLinear

Enfin :

[batch, sequence, 680]
              │
              ▼
          BitLinear
              │
              ▼
[batch, sequence, 256]

Donc le MLP conserve la dimension du Transformer :

256 → 680 → 256
4. Pourquoi deux BitLinear ?

C'est le principe classique du MLP Transformer.

Le premier agrandit la représentation :

256 → 680

Cela permet au réseau de faire davantage de transformations internes.

Puis le deuxième la ramène :

680 → 256

Ainsi le bloc suivant reçoit toujours :

[batch, sequence, 256]