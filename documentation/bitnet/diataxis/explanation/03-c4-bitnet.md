# Architecture C4 - BitNet Transformer

Ce document décrit l'architecture du **BitNet Transformer** selon le modèle **C4** (Context, Containers, Components, Code).

## C1 - Context

Le BitNet Transformer est un modèle de langage causal (type GPT) entraîné from-scratch. Il remplace les projections linéaires classiques (`nn.Linear`) par **BitLinear** qui quantifient les poids en valeurs ternaires `{ -1, 0, +1 }`.

**Acteurs/Utilisateurs**
- Développeurs ML : entraînent, expérimentent, testent
- Chercheurs/Apprenants : explorent l'implémentation BitNet
- Utilisateurs : génèrent du texte via inférence

**Systèmes externes**
- **Dataset C4** (Common Crawl) : données d'entraînement (`data/raw/`, `data/processed/`, `data/tokenized/`)
- **Tokenizer** : `tokenizers` (JSON) stocké dans `tokenizer/tokenizer.json`
- **Fichiers de checkpoint** : `checkpoint/optiminisation_bitnet/*.pt`
- **PyTorch** : backend d'entraînement/inférence

## C2 - Containers

Vue de haut niveau des conteneurs (processus, modules, stockage).

```text
User
│
├─ Scripts CLI
│  ├─ scripts/train_bitnet.py        (Entraînement BitNet)
│  ├─ scripts/bitnet/inference.py    (Inférence BitNet)
│  └─ scripts/tokenize_dataset.py    (Tokenisation)
│
├─ Core Library (src/llm)
│  ├─ bitnet_model/                  (Implémentation modèle BitNet)
│  ├─ inference/generate_bitnet.py   (Helpers chargement/génération)
│  ├─ data/                          (Pipeline données)
│  ├─ tokenizer/                     (Wrapper tokenizer)
│  └─ training/                      (Utils entraînement/checkpoints)
│
├─ Artifacts
│  ├─ tokenizer/*.json               (Tokenizer)
│  ├─ data/{raw,processed,tokenized} (Datasets)
│  └─ checkpoint/optiminisation_bitnet/*.pt (Poids entraînés)
│
└─ Tests
   └─ scripts/tests/bitnet/          (Tests unitaires BitNet)
```

**Frontières** : les scripts CLI orchestrent ; `src/llm/bitnet_model` contient le cœur du modèle.

## C3 - Components

Détail des composants internes du BitNet.

### 3.1 bitnet_model/

| Composant | Rôle | Fichier |
|---|---|---|
| **BitLinear** | Couche linéaire à poids ternaires {-1,0,1}. Poids latents FP32 pour backward, quantifiés au forward (absmax + rounding/clamp). | `bitlinear.py` |
| **BitAttention** | Self-attention multi-têtes. Projections Q,K,V,O via BitLinear. Support mask causal. | `bit_attention.py` |
| **BitMLP** | Feed-forward (MLP) utilisant BitLinear. | `bit_mlp.py` |
| **BitTransformerBlock** | Bloc Transformer : Pre-norm (RMSNorm/LayerNorm) + BitAttention (résidu) + Pre-norm + BitMLP (résidu). | `bit_transformer_block.py` |
| **BitTransformer** | Empile N BitTransformerBlock. Embedding classique + blocks + RMSNorm final. Renvoie hidden states. | `bit_transformer.py` |
| **LMHead** | Projection linéaire (classique) des hidden states vers vocab_size pour prédire logits. | `lm_head.py` |

### 3.2 inference/generate_bitnet.py

| Fonction | Rôle |
|---|---|
| `load_tokenizer()` | Charge tokenizer JSON |
| `load_checkpoint()` | Charge checkpoint PyTorch (map_location device) |
| `create_bitnet_model()` | Instancie BitTransformer + LMHead, charge state_dict avec **remapping QKV** (qkv_proj → q_proj/k_proj/v_proj) si nécessaire, déplace sur device |
| `generate()` | Génération autoregressive (greedy). Boucle max_new_tokens, coupe à max_sequence_length, concatène tokens |

### 3.3 train_bitnet.py

Orchestrateur d'entraînement BitNet :
- Charge tokenizer + dataset tokenisé (JSONL)
- Instancie `BitTransformer` + `LMHead`
- Charge éventuel `PREVIOUS_CHECKPOINT` (reprise)
- Optimizer AdamW, CrossEntropyLoss (ignore_index si défini), gradient clipping
- Boucle epochs/batches, logs (loss, PPL), sauvegarde `NEW_CHECKPOINT` (contient `transformer_state_dict`, `lm_head_state_dict`, méta)

## C4 - Code (vue de détail)

### BitLinear (concept clé)

```text
X (FP32/BFloat) ──► Normalisation/échelle (absmax sur poids/activations selon impl.)
                    │
                    ▼
Poids latents W_fp32 (entraînables)
                    │
                    ▼
Quantification : W_q = round( clamp(W_fp32 / s, -1, 1) )  ∈ {-1,0,1}
                    │
                    ▼
Y = X @ W_q.T  (ou équivalent matmul) + bias (si présent)
```

> Important : seuls les poids latents sont mis à jour par backprop. La quantification est non différentiable (straight-through est utilisée dans implémentations BitNet classiques). L'implémentation ici suit l'approche du projet (poids latents FP32 + quantif forward).

### Flux de données (forward complet)

```text
input_ids (B, T)
  └─► TokenEmbedding (classique) → x0 (B, T, d_model)
       └─► BitTransformer.blocks[0..N-1]
            ├─ Pre-norm (RMS/LayerNorm)
            ├─ BitAttention(Q,K,V,O via BitLinear) + Residual → x'
            └─ Pre-norm + BitMLP (BitLinear up/down) + Residual → x''
       └─► Final RMSNorm → hidden (B, T, d_model)
            └─► LMHead (Linear classique) → logits (B, T, vocab_size)
                 └─► CrossEntropy (next-token prediction, causal)
```

### Remapping QKV (checkpoint compatibility)

Lorsque le checkpoint contient `blocks.N.attention.qkv_proj.weight` (fusionné) et le modèle attend `q_proj/k_proj/v_proj` (séparés), `create_bitnet_model()` effectue un **chunk** sur la dimension appropriée (généralement dim=0 pour `[3*D_out?, ...]` concaténé en sortie) pour splitter Q,K,V.

### Remarques

- **Embedding reste classique** (non ternarisé) — pratique/usuel dans BitNet.
- **LMHead classique** (projection finale) — souvent non ternarisée.
- **Poids ternaires** concernent principalement projections linéaires dans Attention/MLP.
