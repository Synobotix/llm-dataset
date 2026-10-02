Le LM Head est la dernière couche qui transforme ce que le Transformer a compris en scores pour les tokens possibles.

Exemple simple

Supposons que ton tokenizer donne :

"Le chat mange"

comme :

[38, 120, 451]

Le BitTransformer reçoit ces IDs et produit des représentations :

[38, 120, 451]
        ↓
BitTransformer
        ↓
[3, 256]

Chaque token possède maintenant un vecteur de 256 valeurs.

Le problème est que ton vocabulaire contient 994 tokens.

Le LM Head fait donc :

256 dimensions
      ↓
   LM Head
      ↓
994 scores

Donc :

[3, 256]
    ↓
[3, 994]
Que représentent les 994 valeurs ?

Pour chaque position, chaque valeur correspond au score d'un token du vocabulaire.

Par exemple, pour prédire le token suivant après "Le chat mange" :

Token       Score
────────────────────
"une"        2.31
"la"         1.87
"du"         0.92
"maison"     0.41
"de"        -0.15
...

Le modèle peut ensuite prendre le token ayant le score le plus élevé :

"une"

Donc :

"Le chat mange" → "une"
Pourquoi BitLinear dans notre LM Head ?

Notre LM Head contient :

self.projection = BitLinear(
    in_features=d_model,
    out_features=vocab_size,
)

Donc chez nous :

256 → 994

avec des poids quantifiés en :

-1 × scale
 0 × scale
+1 × scale

C'est pourquoi ton test affichait :

tensor([-0.0312, 0.0000, 0.0312])
Important : le LM Head ne comprend pas le texte

Le LM Head ne fait pas :

"Je comprends que le prochain mot est..."

Il effectue simplement une transformation mathématique :

hidden state
     ↓
BitLinear
     ↓
994 logits

C'est ensuite la fonction de perte pendant l'entraînement qui va indiquer au modèle si le score donné au bon token était suffisamment élevé.

Notre architecture complète devient donc :

Tokens
   ↓
Embedding
   ↓
BitTransformer
   ↓
Hidden states [256]
   ↓
LM Head
   ↓
Logits [994]
   ↓
CrossEntropyLoss
   ↓
Apprentissage

Et c'est justement CrossEntropyLoss + Backpropagation + Optimizer qui seront les prochaines pièces nécessaires pour entraîner ton modèle.