# Pipeline LLM — Documentation technique

Vue d'ensemble du pipeline complet, de la collecte des données au modèle final.

----------------------------------------------------------------------

## 0. Flux global du projet

```
Étape 1  → C4 brut
Étape 2  → Nettoyage
Étape 3  → C4 propre
Étape 4  → Split train/validation
Étape 5  → Tokenizer
Étape 6  → Tokenisation + séquences
Étape 7  → Transformer causal
Étape 8  → Pré-entraînement Colab/PC
Étape 9  → Student v1
Étape 10 → Préparation du dataset distillé
Étape 11 → Distillation
Étape 12 → Student v2
Étape 13 → Évaluation
Étape 14 → Modèle final
```

----------------------------------------------------------------------

## 1. Nettoyage du corpus (étapes 1-4)

### a. Pipeline

1. `scripts/test_C4.py` — vérifie l'accès au dataset C4 (français) en streaming, sans tout télécharger.
2. `scripts/prepare_C4.py` — récupère N documents bruts, sauvegardés dans `data/raw/c4_raw.jsonl`.
3. `scripts/clean_C4.py` — nettoie le HTML, les URLs, les emails, les espaces, et élimine les documents manifestement mauvais (trop courts, trop répétitifs, ratio alphabétique trop faible) sans supprimer trop de contenu utile. Produit `data/processed/c4_clean.jsonl`, puis split 90/10 en `train.jsonl` / `validation.jsonl`.
4. `scripts/inspect_clean_C4.py` — vérifie la qualité du corpus après nettoyage (longueur moyenne, échantillons).

### c. Résultat réel obtenu (second run, après correction du split)

Le premier run (913 documents) a été écrasé par erreur lors d'un test local, puis régénéré avec des seuils de nettoyage légèrement différents :

```
Documents analysés     : 1000
Documents conservés    : 644
  - Vides               : 0
  - Trop courts          : 41
  - Ratio alphabétique faible : 9
  - Répétition excessive : 306
  - Doublons              : 0

Train      : 579 documents
Validation : 65 documents
```

----------------------------------------------------------------------

## 9. Entraînement réel en cours (corpus 579/65 documents)

### a. Tokenizer et tokenisation (config complète, réellement exécutées)

```
Tokenizer BPE : vocab_size = 16 000 (atteint, corpus suffisant cette fois)

Train      : 579 docs → 291 470 tokens → 1 138 séquences de 256
Validation : 65 docs  → 33 208 tokens  → 129 séquences de 256
```

### b. Modèle

```
Paramètres totaux : 11 433 088
Config : embedding_dim=256, num_heads=8, num_layers=4, ff_hidden_dim=1024, max_seq_len=256
```

### c. Courbe de loss (en cours, mise à jour au fil de l'entraînement)

```
Epoch 1 | train_loss=8.26 | val_loss=7.70
Epoch 2 | train_loss=7.58 | val_loss=7.54
Epoch 3 | train_loss=7.31 | val_loss=7.37
Epoch 4 | train_loss=7.05 | val_loss=7.21
Epoch 5 | train_loss=6.82 | val_loss=7.11
```

### d. Interprétation

Contrairement au test mécanique à 10 documents (section 6), ici **train_loss et val_loss descendent ensemble** — pas de signe d'overfitting à ce stade. C'est le comportement attendu avec un corpus de taille réaliste : le modèle commence à apprendre des régularités généralisables plutôt que de mémoriser.

**Contrainte observée** : ~4 minutes par epoch sur CPU avec cette configuration (11,4M paramètres, séquences de 256) — l'entraînement complet nécessitera soit beaucoup de temps CPU, soit un passage sur GPU (Colab) pour converger dans un délai raisonnable.

----------------------------------------------------------------------

## 2. Tokenizer BPE (étape 5)

### a. Définition

Le tokenizer transforme le texte en nombres que le Transformer peut comprendre. Un Transformer ne travaille qu'avec des vecteurs/nombres — le tokenizer fait le pont entre le texte brut et les IDs numériques.

### b. Mode d'utilisation

Le texte est découpé en tokens, puis chaque token reçoit un ID issu d'un vocabulaire appris sur le corpus d'entraînement (`train.jsonl`).

```
"Les artisans travaillent rapidement."
       ↓ (BPE)
["Les", " artisans", " travaillent", " rapidement", "."]
       ↓ (vocabulaire)
[125, 842, 3912, 7281, 13]
```

Un mot rare peut être découpé en plusieurs sous-unités :
```
"rapidement" → [" rapide", "ment"]
```

### c. Configuration cible (corpus réel, 913 documents)

```
VOCAB_SIZE     = 16 000
SPECIAL_TOKENS = <pad>, <unk>, <bos>, <eos>
```

Script : `tokenizer/train_tokenizer.py`
Sorties : `tokenizer.json`, `vocab.json`, `config.json`

### d. Note — test mécanique à petite échelle

Un test du pipeline a été réalisé sur 10 documents synthétiques (hors C4 réel) pour valider la mécanique de bout en bout avant de lancer le run complet. Sur un corpus aussi petit, `VOCAB_SIZE` a dû être réduit à 300 (16 000 n'a de sens qu'avec un corpus de taille réelle). Voir section 6 pour le détail de ce test.

----------------------------------------------------------------------

## 3. Tokenisation + création des séquences (étape 6)

### a. Rôle

Transformer le texte tokenisé en séquences numériques exploitables pour l'entraînement : découpage en blocs de taille fixe, avec décalage d'un token entre `input_ids` et `labels` (prédiction du prochain token).

### b. Vue globale

```
train.jsonl
     ↓
Tokenizer BPE
     ↓
Tokens / IDs
     ↓
Découpage en blocs de 256 tokens
     ↓
Séquence de 257 tokens
     ↓ (décalage de 1)
Input (256 IDs) / Labels (256 IDs)
```

### c. Script

`scripts/tokenize_dataset.py` — charge `tokenizer.json`, lit `train.jsonl`/`validation.jsonl`, produit `data/processed/train_tokens.jsonl` et `validation_tokens.jsonl`.

### d. Configuration cible

```
BLOCK_SIZE = 256
```

----------------------------------------------------------------------

## 4. Architecture du modèle — Transformer causal (étape 7)

### a. Structure

```
Token Embedding
      +
Position Embedding
      ↓
Transformer Block × 4
      ↓
Final LayerNorm
      ↓
Linear (LM head)
      ↓
Logits
```

### b. Configuration cible (modèle réel)

```python
vocab_size         = 16_000
embedding_dim      = 256
num_heads          = 8
num_layers         = 4
ff_hidden_dim      = 1024
max_sequence_length = 256
```

### c. Fichiers (`src/llm/model/`)

| Fichier | Rôle |
|---|---|
| `embedding.py` | Token embedding (IDs → vecteurs denses) |
| `positional_encoding.py` | Embeddings de position appris, ajoutés aux tokens |
| `attention.py` | Multi-head self-attention causale (masque triangulaire) |
| `feed_forward.py` | MLP position-wise (Linear → GELU → Linear) |
| `transformer_block.py` | Bloc complet pre-norm : `x + Attn(LN(x))`, puis `x + FF(LN(x))` |
| `transformer.py` | Assemblage complet (`CausalTransformer`) |

----------------------------------------------------------------------

## 5. Entraînement (étape 8 → Student v1, étape 9)

### a. Script

`scripts/train.py` — boucle d'entraînement AdamW + CrossEntropyLoss, sauvegarde du meilleur checkpoint selon la validation loss.

### b. Configuration cible

```python
BATCH_SIZE = 16
EPOCHS     = 20  (à ajuster selon la convergence réelle)
LR         = 3e-4
```

Checkpoint : `checkpoints/student_v1/model.pt`

----------------------------------------------------------------------

## 6. Test mécanique du pipeline (10 documents synthétiques)

Avant de lancer l'entraînement réel sur le corpus C4 complet, un test de bout en bout a été réalisé avec 10 documents français écrits manuellement (pas du vrai C4), pour valider que chaque étape du pipeline fonctionne sans erreur.

### Configuration réduite utilisée pour ce test

```python
VOCAB_SIZE     = 300    # au lieu de 16 000
BLOCK_SIZE     = 32     # au lieu de 256
EMBEDDING_DIM  = 64     # au lieu de 256
NUM_HEADS      = 4      # au lieu de 8
NUM_LAYERS     = 2      # au lieu de 4
FF_HIDDEN_DIM  = 128    # au lieu de 1024
```

### Résultats obtenus

```
Documents   : 8 train / 2 validation (378 mots au total)
Tokens      : 844 (train) / 249 (validation)
Séquences   : 26 (train) / 7 (validation)
Paramètres  : 107 820

Epoch   1 | train_loss=5.90 | val_loss=5.82
Epoch  50 | train_loss=3.16 | val_loss=5.19
Epoch 100 | train_loss=0.49 | val_loss=5.62
Epoch 300 | train_loss=0.02 | val_loss=6.30
```

### Interprétation

- Le `train_loss` s'effondre jusqu'à quasi 0 : le modèle **mémorise** les 8 documents d'entraînement.
- Le `val_loss` remonte au fil de l'entraînement : **overfitting net**, attendu avec un corpus aussi petit (378 mots).
- En génération, un prompt correspondant au début exact d'un document mémorisé produit une suite cohérente (récitation). Les autres prompts produisent du texte incohérent, faute de données suffisantes pour généraliser.

**Conclusion** : le pipeline mécanique (nettoyage → tokenizer → tokenisation → training → génération) fonctionne de bout en bout sans bug. La cohérence linguistique réelle nécessite un passage à l'échelle (corpus de 913+ documents, configuration complète).

----------------------------------------------------------------------

## 7. Distillation (étapes 10-11)

### a. Architecture

```
C4 nettoyé
    ↓
generate_distillation.py
    ├──► prompts.py       → construit le prompt d'instruction
    ├──► teachers.py      → sélectionne un groupe de teachers (rotation)
    ├──► openrouter.py    → appelle l'API OpenRouter
    ├──► validator.py     → valide la réponse (longueur min/max)
    └──► storage.py       → sauvegarde en JSONL
         ↓
data/distilled/distilled.jsonl
```

### b. Teachers (21 modèles, rotation par groupes de 3)

OpenRouter limite le fallback à 3 modèles par requête. Les 21 teachers sont donc répartis en 7 groupes de 3, choisis par rotation round-robin selon l'index du document traité — ce qui permet d'exploiter l'ensemble des 21 modèles sur le corpus tout en respectant la contrainte de l'API.

### c. Format d'un exemple distillé

```json
{
  "source": "...",
  "prompt": "...",
  "response": "...",
  "teacher": "deepseek/deepseek-v4.1-flash"
}
```

----------------------------------------------------------------------

## 8. État d'avancement

| Étape | Statut |
|---|---|
| 1-4. Collecte + nettoyage | ✅ Fait (644 documents conservés sur 1000 ; 579 train / 65 validation) |
| 5. Tokenizer BPE | ✅ Fait sur le corpus réel (vocab 16 000 atteint) |
| 6. Tokenisation | ✅ Fait (1 138 séquences train, 129 validation, blocs de 256) |
| 7. Architecture Transformer | ✅ Fait (tous les fichiers du modèle écrits) |
| 8-9. Entraînement Student v1 | 🔄 En cours (5 epochs faites, train/val loss descendent ensemble, pas d'overfitting constaté) |
| Test mécanique (10 docs) | ✅ Fait (pipeline validé de bout en bout, overfitting attendu confirmé) |
| 10-11. Distillation | ✅ Code prêt (`teachers.py`, `prompts.py`, `validator.py`, `storage.py`, `generate_distillation.py`), pas encore exécuté à grande échelle |
| 12-14. Student v2, évaluation, modèle final | ⬜ À venir |
