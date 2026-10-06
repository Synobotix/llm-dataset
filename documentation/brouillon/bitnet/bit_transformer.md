BitTransformer est le modèle complet qui empile plusieurs BitTransformerBlock.

Jusqu'ici, nous avons construit les pièces séparément :

BitLinear
   ↓
BitAttention
   ↓
BitMLP
   ↓
BitTransformerBlock
   ↓
BitTransformer
Concrètement, il fait quoi ?

Supposons que ton tokenizer transforme :

"Je vais au marché"

en :

[38, 125, 72, 291]

Le BitTransformer reçoit ces IDs de tokens :

input_ids
[38, 125, 72, 291]
1. Embedding

Les IDs deviennent des vecteurs :

[38, 125, 72, 291]
        ↓
[256 dimensions]
[256 dimensions]
[256 dimensions]
[256 dimensions]

Donc avec une séquence de 4 tokens :

[4] → [4, 256]

L'embedding reste classique, il n'est pas ternarisé.

2. Passage dans les blocks

Ensuite :

Embedding
    ↓
Block 1
    ↓
Block 2
    ↓
Block 3
    ↓
...
    ↓
Block N

Chaque BitTransformerBlock fait :

RMSNorm
   ↓
BitAttention
   ↓
résidu
   ↓
RMSNorm
   ↓
BitMLP
   ↓
résidu

Et à l'intérieur de BitAttention et BitMLP, les projections utilisent nos BitLinear.

Donc c'est là que la partie BitNet intervient principalement.

3. Pourquoi plusieurs blocks ?

Chaque block permet au modèle de transformer progressivement la représentation des tokens.

Très simplifié :

Embedding
   ↓
Block 1
"quels tokens sont liés ?"
   ↓
Block 2
"quelles relations sont importantes ?"
   ↓
Block 3
"quel contexte se dégage ?"
   ↓
...
   ↓
Block N
représentation finale

Ce n'est pas que chaque block réalise exactement ces opérations conscientes ; c'est une manière simple de comprendre que les représentations sont progressivement transformées.

4. RMSNorm final

Après tous les blocks :

Block N
   ↓
RMSNorm

On obtient les hidden states finaux.

Par exemple :

[batch, sequence, 256]

Avec notre test :

[2, 8, 256]
5. Et ensuite ?

C'est important : BitTransformer ne prédit pas encore directement les tokens.

Pour l'instant :

input_ids
    ↓
Embedding
    ↓
BitTransformerBlock × N
    ↓
RMSNorm
    ↓
hidden states

Il manque :

                    LM Head
                       ↓
hidden states → logits du vocabulaire
                       ↓
              probabilités
                       ↓
                prochain token

Par exemple, si ton vocabulaire contient 994 tokens, le LM Head transformera :

[256]

en :

[994]

Chaque valeur représente le score d'un token possible.

Donc le pipeline complet sera :

Tokens
   ↓
Embedding
   ↓
BitTransformer
   ↓
Hidden states
   ↓
LM Head
   ↓
Logits
   ↓
Softmax
   ↓
Prédiction du prochain token

C'est justement le LM Head qui sera notre prochaine grande étape après avoir validé BitTransformer.