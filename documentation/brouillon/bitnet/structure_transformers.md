On va refaire totalement le transformers pour que tous le calcul soit adatpé:

-structure code:

src/llm/bitnet_model/
│
│
├── bitlinear.py
│
├── bit_attention.py
│
├── bit_mlp.py
│
├── bit_transformer_block.py
│
├── bit_transformer.py
│
└── lm_head.py

training:

scripts/train_bitnet.py

les étapes à faire:

1. Définir l'architecture BitNet
        ↓
2. bitlinear.py
        ↓
3. Tester BitLinear seul
        ↓
4. bit_mlp.py
        ↓
5. Tester BitNet MLP
        ↓
6. bit_attention.py
        ↓
7. Tester BitNet Attention
        ↓
8. bit_transformer_block.py
        ↓
9. Tester un Transformer Block
        ↓
10. bit_transformer.py
        ↓
11. Tester le Transformer complet
        ↓
12. LM Head
        ↓
13. Tester la prédiction des tokens
        ↓
14. Loss + Backpropagation
        ↓
15. Tester l'entraînement sur quelques batchs
        ↓
16. Vérifier les checkpoints
        ↓
17. Entraînement sur ton dataset tokenisé