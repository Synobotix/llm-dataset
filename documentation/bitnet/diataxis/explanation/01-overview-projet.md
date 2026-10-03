# Vue d'ensemble du projet

Ce projet implémente **deux versions d'un Transformer causal** à partir de zéro, en PyTorch, dans le cadre d'un parcours d'apprentissage/apprentissage pratique.

## Objectifs

- Comprendre l'architecture Transformer (embeddings, attention, MLP, blocs)
- Implémenter un pipeline complet : données → nettoyage → tokenization → entraînement → inférence
- Expérimenter **BitNet** : remplacer les projections linéaires par des **BitLinear** à poids ternaires (-1,0,1)
- Comparer approches classique vs BitNet (structure, comportement, poids)

## Deux modèles

### 1) Transformer classique (`src/llm/model/`)

Architecture GPT-like standard :

- `Transformer` : empilement de `TransformerBlock`
- `MultiHeadAttention` / `ScaledDotProductAttention`
- `FeedForward` (MLP classique)
- `nn.Linear` pour les projections Q/K/V/O et MLP
- `TokenEmbedding` + encodage positionnel
- `LayerNorm` (selon implémentation)

Points d'entrée : `scripts/train.py`, `scripts/inference.py`.

### 2) Transformer BitNet (`src/llm/bitnet_model/`)

Même structure globale, mais avec BitNet :

- `BitLinear` : poids latents FP32, quantifiés en {-1,0,1} au forward
- `BitAttention` : projections Q/K/V/O via `BitLinear`
- `BitMLP` : couches via `BitLinear`
- `BitTransformerBlock` + `BitTransformer`
- `LMHead` (projection finale vers vocabulaire)
- Support RMSNorm/normalisation selon implémentation

Points d'entrée : `scripts/train_bitnet.py`, `scripts/bitnet/inference.py`.

## Pipeline de données

Le dataset utilisé est basé sur C4 (Common Crawl). Étapes clés :

1. **Préparation** (`scripts/prepare_C4.py`) – chargement/formatage
2. **Nettoyage/filtrage** (`scripts/clean_C4.py`, `src/llm/data/cleaning.py`, `filtering.py`)
3. **Déduplication** (`src/llm/data/deduplication.py`)
4. **Split train/validation** (`scripts/split_dataset.py`)
5. **Tokenization** (`tokenizer/train_tokenizer.py`, `src/llm/tokenizer/`)
6. **Création de séquences** (`scripts/tokenize_dataset.py`, `src/llm/data/dataset.py`)

Données dans `data/raw/`, `data/processed/`, `data/tokenized/`.

## Entraînement & inférence

- **Classique** : `Trainer` dans `src/llm/training/trainer.py`, checkpoints gérés via `checkpoint.py`. Config via `scripts/train.py`.
- **BitNet** : entraînement manuel/structuré dans `scripts/train_bitnet.py`. Sauvegarde/chargement dans `checkpoint/optiminisation_bitnet/`.
- **Inférence** : génération autoregressive greedy (par défaut) dans les scripts respectifs.

## Distillation

Un volet **teacher-student** est présent (`src/llm/distillation/`, `scripts/generate_distillation.py`). L'objectif est de distiller un teacher (potentiellement via OpenRouter) vers le student.

## Organisation du code

```text
src/llm/
├── model/              # Transformer classique
├── bitnet_model/       # Transformer BitNet
├── data/               # Nettoyage, filtering, dedup, dataset, dataloader
├── tokenizer/          # Tokenizer wrapper
├── training/           # Trainer, loss, optimizer, scheduler, checkpoint
├── distillation/       # Teacher/student, prompts, validator, storage
├── evaluation/         # Metrics, evaluation
├── inference/          # Helpers génération (generate.py, generate_bitnet.py)
└── config/             # Paramètres
```
