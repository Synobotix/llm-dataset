Donc je proposerais 3 niveaux

Niveau 1 — ce qu'on vient de faire

BitLinear Q/K/V
       ↓
SDPA
       ↓
BitLinear Output

Optimisation du calcul de l'attention.

Niveau 2 — optimisation BitNet de l'architecture

                 BitLinear
                256 → 768
                     ↓
                  split
               ↙    ↓    ↘
              Q     K     V
               \    |    /
                SDPA causal
                     ↓
                 BitLinear
                256 → 256

C'est celui que je testerais ensuite.

Niveau 3 — vraie accélération matérielle

Là on quitte la simple optimisation PyTorch :

poids ternaires
      ↓
packing réel
      ↓
kernel spécialisé
      ↓
calcul ternaire / int8
      ↓
GPU