# Quickstart : Inférence avec BitNet

Ce tutoriel vous permet de générer du texte avec **le Transformer BitNet** (projections ternaires -1,0,1).

## 1. Prérequis

- Environnement virtuel activé
- Un checkpoint BitNet entraîné

## 2. Lancer l'inférence BitNet

Depuis la racine du projet :

```bash
poetry run python scripts/bitnet/inference.py
```

## 3. Que fait ce script ?

1. **Charge le tokenizer** via `llm.inference.generate_bitnet.load_tokenizer`
2. **Charge le checkpoint** depuis `checkpoint/optiminisation_bitnet/bitnet_50docs1.pt` (par défaut)
3. **Construit le BitTransformer** via `create_bitnet_model` (avec remapping QKV si nécessaire)
4. **Génère du texte** à partir du prompt défini dans `scripts/bitnet/prompt.py`

## 4. Modifier le prompt

Éditez `scripts/bitnet/prompt.py` :

```python
PROMPT = (
    "il était une fois un petit garçon qui se promenait dans les bois quand tout à coup il rencontra un monstre"
)
```

Relancez ensuite l'inférence.

## 5. Configuration de génération

Dans `scripts/bitnet/inference.py`, vous pouvez ajuster :

```python
MAX_NEW_TOKENS = 50
TEMPERATURE = 1.0
CHECKPOINT_FILE = Path("checkpoint/optiminisation_bitnet/bitnet_50docs1.pt")
```

## 6. Erreurs fréquentes

**Checkpoint introuvable**

```text
FileNotFoundError: ...
```

Créez ou récupérez un checkpoint BitNet. Voir [Premier entraînement (BitNet)](./04-premier-entrainement-bitnet.md).

**Mismatch QKV**

Si vous rencontrez l'erreur `Missing key(s) in state_dict: ... q_proj/k_proj/v_proj` vs `qkv_proj`, le remapping dans `generate_bitnet.py` doit gérer cela. C'est déjà corrigé dans le code actuel.

## Prochaines étapes

- [Premier entraînement (BitNet)](./04-premier-entrainement-bitnet.md)
- [Lancer l'inférence (BitNet) - How-to](../how-to-guides/07-inference-bitnet.md)
