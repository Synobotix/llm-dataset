1. Le rôle général

Supposons que le texte soit :

"Je vais au marché"

Après le tokenizer puis l'embedding, on obtient quelque chose comme :

Je       → vecteur
vais     → vecteur
au       → vecteur
marché   → vecteur

Donc BitAttention reçoit une matrice :

X = [batch, séquence, d_model]

Dans notre test :

X = [2, 8, 256]

Cela signifie :

2 séquences dans le batch
8 tokens par séquence
chaque token est représenté par 256 nombres
2. Première chose : fabriquer Q, K et V

BitAttention possède trois BitLinear :

self.q_proj = BitLinear(...)
self.k_proj = BitLinear(...)
self.v_proj = BitLinear(...)

On fait donc :

X
│
├── BitLinear → Q
├── BitLinear → K
└── BitLinear → V

Mathématiquement :

Q = X × WQ
K = X × WK
V = X × WV

Mais contrairement à un Transformer classique, WQ, WK et WV sont quantifiés par notre BitLinear :

WQ → {-1, 0, +1} × scale
WK → {-1, 0, +1} × scale
WV → {-1, 0, +1} × scale

C'est là que BitNet intervient directement dans l'attention.

3. Pourquoi Q, K et V ?

C'est le cœur du mécanisme d'attention.

On peut les comprendre ainsi :

Query — Q

« Qu'est-ce que ce token cherche ? »

Key — K

« Qu'est-ce que ce token peut fournir comme information ? »

Value — V

« Quelle information ce token va réellement transmettre ? »

Par exemple, dans :

"Le chat mange la souris"

lorsque le modèle traite mange, son Query peut avoir une forte correspondance avec les Keys de chat et souris.

L'attention permet donc au modèle de déterminer quels tokens sont importants pour chaque token.

4. Plusieurs têtes

Nous avons :

d_model = 256
num_heads = 4

On divise donc les 256 dimensions en 4 groupes :

256 / 4 = 64

Chaque tête possède donc :

head_dim = 64

Avant séparation :

Q
[2, 8, 256]

Après séparation :

[2, 4, 8, 64]

Donc :

batch
 ↓
2 séquences

heads
 ↓
4 têtes

sequence
 ↓
8 tokens

head_dim
 ↓
64 dimensions

Chaque tête peut apprendre différentes relations entre les tokens.

5. Q × Kᵀ

Ensuite on compare Q avec K.

Le code fait :

attention_scores = torch.matmul(
    q,
    k.transpose(-2, -1),
)

C'est :

Q × Kᵀ

Pour une tête :

Q       [8, 64]
Kᵀ      [64, 8]

             ↓

scores  [8, 8]

Donc chaque token est comparé aux 8 tokens.

On obtient une matrice :

          token
        1  2  3  4
     ┌─────────────
  1  │ ?  ?  ?  ?
  2  │ ?  ?  ?  ?
  3  │ ?  ?  ?  ?
  4  │ ?  ?  ?  ?

Chaque ? représente un score indiquant à quel point un token porte attention à un autre.

6. Pourquoi diviser par √64 ?

On fait :

attention_scores / math.sqrt(self.head_dim)

Donc ici :

√64 = 8

On fait :

scores / 8

Cela évite que les scores deviennent trop grands avant le softmax.

Sans cette normalisation, le softmax pourrait devenir trop extrême et rendre l'apprentissage plus difficile.

7. Le causal mask

C'est indispensable pour notre GPT.

Supposons :

Je vais au marché

Lorsque le modèle est en train de traiter :

au

il peut regarder :

Je
vais
au

mais il ne doit pas regarder :

marché

car marché est un token futur.

Le masque crée donc :

        Je  vais  au  marché

Je      ✓    ✗    ✗     ✗
vais    ✓    ✓    ✗     ✗
au      ✓    ✓    ✓     ✗
marché  ✓    ✓    ✓     ✓

Dans le code :

attention_scores = attention_scores.masked_fill(
    causal_mask,
    float("-inf"),
)

Les positions interdites deviennent :

-inf

Puis le softmax leur donne pratiquement :

0

Donc elles ne contribuent pas à l'attention.

8. Softmax

Après le masque :

scores
   ↓
softmax
   ↓
attention_weights

Imaginons qu'un token ait les scores :

[2.0, 1.0, 0.5, -inf]

Après softmax, on obtient quelque chose comme :

[0.63, 0.23, 0.14, 0.00]

Le modèle dit alors essentiellement :

Token 1 → importance 63 %
Token 2 → importance 23 %
Token 3 → importance 14 %
Token 4 → interdit

C'est cette matrice de poids qui indique où regarder.

9. Attention × V

Ensuite :

attention_output = torch.matmul(
    attention_weights,
    v,
)

Donc :

Attention = Softmax(QKᵀ / √d) × V

Le modèle prend les informations contenues dans V et les mélange selon les poids d'attention.

Par exemple :

V1 × 0.63
+
V2 × 0.23
+
V3 × 0.14

Cela produit une nouvelle représentation du token.

10. On rassemble les 4 têtes

Nous avions :

[batch, 4, sequence, 64]

On rassemble les 4 têtes :

4 × 64 = 256

On revient donc à :

[batch, sequence, 256]
11. Projection finale

Enfin :

output = self.out_proj(attention_output)

out_proj est également une :

BitLinear(256 → 256)

Donc les quatre projections de notre attention utilisent BitLinear :

Q       → BitLinear
K       → BitLinear
V       → BitLinear
Output  → BitLinear
12. Où est réellement BitNet ?

C'est probablement le point le plus important.

Notre BitAttention ne fait pas :

tout → -1 / 0 / +1

Elle fait plutôt :

                 BitNet
                   ↓
X ──→ BitLinear ──→ Q
X ──→ BitLinear ──→ K
X ──→ BitLinear ──→ V
          │
          ▼
       Attention
          │
       Softmax
          │
          ▼
      BitLinear
          │
          ▼
        Output

Les opérations comme :

Q × Kᵀ
Softmax
Attention × V

restent des opérations numériques classiques dans notre implémentation.

13. Et pourquoi c'est intéressant ?

Dans un Transformer classique, les grosses projections utilisent des poids flottants :

W =

 0.127
-0.843
 0.291
 0.512
...

Dans notre BitLinear, le forward utilise :

Wq = scale ×

+1
-1
 0
+1
...

Donc nous réduisons fortement la précision des poids des projections.

Mais nous conservons :

W_float

pour l'apprentissage.

C'est cette combinaison qui permet d'avoir :

poids latents flottants
        ↓
quantification
        ↓
poids ternaires
        ↓
forward
        ↓
loss
        ↓
STE
        ↓
gradient
        ↓
poids latents mis à jour
En résumé

BitAttention fait donc exactement le travail d'une Multi-Head Self-Attention causale, mais ses projections sont construites avec notre BitLinear :

X
│
├── Q = BitLinear(X)
├── K = BitLinear(X)
└── V = BitLinear(X)
       │
       ▼
    QKᵀ / √d
       │
       ▼
  Causal Mask
       │
       ▼
    Softmax
       │
       ▼
     × V
       │
       ▼
   4 têtes fusionnées
       │
       ▼
 BitLinear Output
       │
       ▼
      X'

C'est donc le deuxième gros composant BitNet après BitMLP.

Maintenant, le test de BitAttention doit vérifier séparément les dimensions, le masque causal, le forward et le backward, avant de l'intégrer dans bit_transformer_block.py.