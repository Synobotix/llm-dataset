1. Architecture générale

Nous allons construire un Transformer causal de type GPT, mais avec les projections linéaires principales remplacées par des opérations BitNet.

                    input_ids
                        │
                        ▼
                 Token Embedding
                        │
                        ▼
              Position / contexte
                        │
                        ▼
              ┌──────────────────┐
              │  BitNet Block 1  │
              └────────┬─────────┘
                       │
              ┌────────▼─────────┐
              │  BitNet Block 2  │
              └────────┬─────────┘
                       │
                      ...
                       │
              ┌────────▼─────────┐
              │  BitNet Block N  │
              └────────┬─────────┘
                       │
                       ▼
                   Final Norm
                       │
                       ▼
                    LM Head
                       │
                       ▼
                     logits
                       │
                       ▼
                      loss
2. Un BitNet Block

Chaque bloc aura cette structure :

                         X
                         │
                         ▼
                       Norm
                         │
                         ▼
                  BitNet Attention
                         │
                         ▼
                    + Residual
                         │
                         ▼
                        X'
                         │
                         ▼
                       Norm
                         │
                         ▼
                     BitNet MLP
                         │
                         ▼
                    + Residual
                         │
                         ▼
                        X''

Donc mathématiquement, avec une architecture pre-norm :

X' = X + Attention(Norm(X)) 

puis :

X'' = X' + MLP(Norm(X')) 

C'est X'' qui entre dans le bloc suivant.

3. BitNet Attention

L'attention sera composée de quatre projections principales :

                  X
                  │
       ┌──────────┼──────────┐
       ▼          ▼          ▼
   BitLinear   BitLinear   BitLinear
       │          │          │
       ▼          ▼          ▼
       Q          K          V
       │          │          │
       └─────┬────┴─────┬────┘
             ▼          │
        Q × Kᵀ         │
             │          │
             ▼          │
       Scaling +        │
       causal mask      │
             │          │
             ▼          │
          Softmax       │
             │          │
             └────┬─────┘
                  ▼
              Attention × V
                  │
                  ▼
             BitLinear
                  │
                  ▼
               Output

Les projections :

Q = BitLinear(X)
K = BitLinear(X)
V = BitLinear(X)

et la projection finale sera également BitLinear.

4. Où intervient réellement le BitNet ?

C'est important.

Nous ne ferons pas :

tout → 1 bit

Nous aurons plutôt :

             BitNet Transformer
                     │
        ┌────────────┴────────────┐
        │                         │
   Bit Attention              Bit MLP
        │                         │
        ▼                         ▼
   BitLinear                  BitLinear
        │                         │
        ▼                         ▼
 poids ternaires             poids ternaires

Les poids des BitLinear seront quantifiés vers :

W \in \{-1,0,+1\} 

C'est le cœur de BitNet b1.58.

5. BitNet MLP

Le MLP sera :

                 X
                 │
                 ▼
              BitLinear
                 │
                 ▼
             Activation
                 │
                 ▼
              BitLinear
                 │
                 ▼
               Output

Donc :

MLP(X)=W_2\,Activation(W_1X) 

mais W₁ et W₂ seront gérés par BitLinear.

6. Les paramètres que nous allons définir

Notre modèle aura une configuration centralisée.

Par exemple :

vocab_size
d_model
n_layers
n_heads
d_ff
max_seq_len

Avec :

vocab_size
    ↓
nombre de tokens du tokenizer

d_model
    ↓
dimension des représentations

n_layers
    ↓
nombre de BitNet Blocks

n_heads
    ↓
nombre de têtes d'attention

d_ff
    ↓
dimension interne du MLP

max_seq_len
    ↓
longueur maximale d'une séquence

Nous ne fixerons pas encore arbitrairement les valeurs finales : elles dépendront de la taille du Student que tu veux entraîner.

7. Embedding

L'entrée sera :

input_ids

par exemple :

[38, 125, 72, 914]

L'embedding transforme chaque token en vecteur :

38  → [ ... d_model valeurs ... ]
125 → [ ... d_model valeurs ... ]
72  → [ ... d_model valeurs ... ]
914 → [ ... d_model valeurs ... ]

Donc :

input_ids
    ↓
Embedding
    ↓
[B, T, d_model]

où :

B = batch size
T = longueur de séquence.

Nous ne quantifions pas automatiquement l'embedding en ternary.

8. Normalisation

Avant Attention et MLP, nous utiliserons une normalisation de type RMSNorm plutôt qu'une architecture qui ajoute inutilement des opérations.

Donc :

X
 ↓
RMSNorm
 ↓
BitNet Attention

puis :

X'
 ↓
RMSNorm
 ↓
BitNet MLP

Le choix exact pourra être fixé dans notre configuration.

9. Activations

Le MLP aura une fonction d'activation.

Nous pourrons utiliser une activation adaptée aux architectures GPT modernes, par exemple :

SiLU / SwiGLU

Si nous choisissons SwiGLU, le MLP sera légèrement différent :

                 X
              /     \
             /       \
       BitLinear   BitLinear
          │           │
        SiLU          │
          │           │
          └───── × ───┘
                │
                ▼
            BitLinear

C'est une possibilité intéressante, mais nous devons choisir explicitement cette variante avant d'écrire bit_mlp.py.

10. La partie la plus importante : BitLinear

Chaque BitLinear aura conceptuellement :

                 W_float
                    │
                    ▼
              quantification
                    │
                    ▼
               W_ternaire
               {-1,0,+1}
                    │
                    ▼
                   ×
                    ▲
                    │
              X quantifié
                    │
                    ▼
                 Output

Pendant le training :

W_float
   ↓
quantification
   ↓
W_ternaire
   ↓
forward
   ↓
loss
   ↓
backward / STE
   ↓
gradient
   ↓
AdamW
   ↓
W_float mis à jour

C'est cette partie que nous développerons ensuite dans bitlinear.py.

11. Causalité

Comme notre modèle est un GPT, l'attention doit rester causale.

Pour :

Le chat mange

au moment de prédire mange, le modèle peut utiliser :

Le
chat

mais pas les tokens futurs.

La matrice sera conceptuellement :

       1  2  3  4
1      ✓  ✗  ✗  ✗
2      ✓  ✓  ✗  ✗
3      ✓  ✓  ✓  ✗
4      ✓  ✓  ✓  ✓

Cela fera partie de bit_attention.py.

12. Pipeline complet de notre architecture

Finalement :

                         input_ids
                             │
                             ▼
                        Embedding
                             │
                             ▼
                         Position
                             │
                             ▼
                ╔════════════════════════╗
                ║     BitNet Block       ║
                ║                        ║
                ║       RMSNorm          ║
                ║          ↓             ║
                ║   BitLinear Q/K/V      ║
                ║          ↓             ║
                ║  Causal Attention      ║
                ║          ↓             ║
                ║    BitLinear Output    ║
                ║          ↓             ║
                ║      Residual          ║
                ║          ↓             ║
                ║       RMSNorm          ║
                ║          ↓             ║
                ║      BitNet MLP        ║
                ║          ↓             ║
                ║      Residual          ║
                ╚══════════╤═════════════╝
                           │
                           ▼
                       × N Blocks
                           │
                           ▼
                       Final RMSNorm
                           │
                           ▼
                         LM Head
                           │
                           ▼
                         Logits
                           │
                           ▼
                    Cross Entropy Loss