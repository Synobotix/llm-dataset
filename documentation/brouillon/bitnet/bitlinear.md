L’objectif est de construire une vraie couche BitLinear qui pourra ensuite remplacer les nn.Linear principaux de notre Transformer BitNet.

2.1 Rôle de BitLinear

Notre couche aura cette logique :

Entrée X
   │
   ▼
Normalisation / quantification de X
   │
   ▼
Poids flottants entraînables W
   │
   ▼
Quantification de W
   │
   ▼
W ∈ {-1, 0, +1}
   │
   ▼
Calcul linéaire
   │
   ▼
Sortie

Un point très important : on ne va pas stocker les poids entraînables uniquement sous forme -1/0/+1.

On conserve des poids latents flottants pour que backward() et optimizer.step() puissent fonctionner. Pendant le forward, on construit les poids quantifiés.