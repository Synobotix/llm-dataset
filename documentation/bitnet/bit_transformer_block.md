Ce bloc fait maintenant

Pour une entrée :

[B, T, 256]

par exemple :

[2, 8, 256]

on a :

                    [2, 8, 256]
                          │
                          ▼
                       RMSNorm
                          │
                          ▼
                    BitAttention
                          │
                          ▼
                    + résiduel ◄──── entrée
                          │
                          ▼
                       RMSNorm
                          │
                          ▼
                       BitMLP
                          │
                          ▼
                    + résiduel ◄──── précédent
                          │
                          ▼
                    [2, 8, 256]

Les poids des projections de l'attention et du MLP sont ternaires, grâce à BitLinear.

En revanche, RMSNorm, les connexions résiduelles et les opérations d'attention restent des opérations numériques classiques.