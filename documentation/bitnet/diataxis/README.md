# Documentation Diátaxis - LLM Dataset

Cette documentation est organisée selon le [framework Diátaxis](https://diataxis.fr/) : **Tutorials**, **How-to guides**, **Reference**, **Explanation**.

Le projet contient deux modèles distincts de Transformer :

| Modèle | Implémentation | Scripts | Dossier doc |
|---|---|---|---|
| **Transformer classique** | `src/llm/model/` | `scripts/train.py`, `scripts/inference.py` | Ce dossier (séparé par sections) |
| **BitNet Transformer** | `src/llm/bitnet_model/` | `scripts/train_bitnet.py`, `scripts/bitnet/inference.py` | Ce dossier (séparé par sections) |

## Structure Diátaxis

### [Tutorials](./tutorials/) - Apprendre par la pratique
Destinés aux débutants. Permettent d'apprendre en faisant, étape par étape, sans présumer de connaissances préalables.

- [Quickstart - Inférence (classique)](./tutorials/01-quickstart-classique.md)
- [Quickstart - Inférence (BitNet)](./tutorials/02-quickstart-bitnet.md)
- [Premier entraînement (classique)](./tutorials/03-premier-entrainement-classique.md)
- [Premier entraînement (BitNet)](./tutorials/04-premier-entrainement-bitnet.md)

### [How-to guides](./how-to-guides/) - Résoudre une tâche
Guides orientés tâche pour atteindre un objectif précis. Supposent une certaine connaissance du système.

- [Préparer les données C4](./how-to-guides/01-preparer-donnees.md)
- [Nettoyer et filtrer le dataset](./how-to-guides/02-nettoyer-filtrer.md)
- [Splitter train/validation](./how-to-guides/03-split-dataset.md)
- [Entraîner le modèle classique](./how-to-guides/04-entrainer-classique.md)
- [Entraîner le modèle BitNet](./how-to-guides/05-entrainer-bitnet.md)
- [Lancer l'inférence (classique)](./how-to-guides/06-inference-classique.md)
- [Lancer l'inférence (BitNet)](./how-to-guides/07-inference-bitnet.md)
- [Générer des distillations](./how-to-guides/08-distillation.md)
- [Évaluer un modèle](./how-to-guides/09-evaluer-modele.md)

### [Reference](./reference/) - Information technique précise
Description factuelle, structurée et exhaustive (API, paramètres, configurations).

- [Transformer classique - Vue d'ensemble](./reference/classique/01-overview.md)
- [Transformer classique - Embedding & PE](./reference/classique/02-embedding-pe.md)
- [Transformer classique - Attention](./reference/classique/03-attention.md)
- [Transformer classique - Feed Forward](./reference/classique/04-feedforward.md)
- [Transformer classique - Blocks & Model](./reference/classique/05-transformer.md)
- [BitNet - Vue d'ensemble](./reference/bitnet/01-overview.md)
- [BitNet - BitLinear](./reference/bitnet/02-bitlinear.md)
- [BitNet - BitAttention](./reference/bitnet/03-bitattention.md)
- [BitNet - BitMLP](./reference/bitnet/04-bitmlp.md)
- [BitNet - BitTransformerBlock & BitTransformer](./reference/bitnet/05-bittransformer.md)
- [BitNet - LM Head](./reference/bitnet/06-lmhead.md)
- [Configurations & paramètres](./reference/07-configurations.md)
- [DataLoaders & Datasets](./reference/08-datasets-dataloaders.md)
- [Tokenizer](./reference/09-tokenizer.md)

### [Explanation](./explanation/) - Compréhension & architecture
Explore le *pourquoi* : concepts, choix de conception, architecture (C4).

- [Vue d'ensemble du projet](./explanation/01-overview-projet.md)
- [Architecture C4 - Transformer classique](./explanation/02-c4-classique.md)
- [Architecture C4 - BitNet](./explanation/03-c4-bitnet.md)
- [Transformer causal : explication](./explanation/04-transformer-causal.md)
- [Attention multi-têtes](./explanation/05-attention-mha.md)
- [BitNet : principes & motivations](./explanation/06-bitnet-principes.md)
- [Quantification ternaire (-1,0,1)](./explanation/07-quantification-ternaire.md)
- [Distillation teacher-student](./explanation/08-distillation.md)
- [Pipeline de données C4](./explanation/09-pipeline-donnees.md)
- [Choix d'entraînement & métriques](./explanation/10-entrainement-metriques.md)
