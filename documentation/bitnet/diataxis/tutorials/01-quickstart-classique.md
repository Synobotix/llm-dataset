# Quickstart : Inférence avec le Transformer classique

Ce tutoriel vous permet de générer du texte avec **le Transformer classique** en moins de 5 minutes.

## 1. Prérequis

- Le projet cloné localement
- Environnement virtuel activé (`poetry shell` ou `.venv` activé)
- Un checkpoint entraîné (si non présent, vous verrez une erreur explicite)

## 2. Vérifier l'environnement

Depuis la racine du projet :

```bash
cd /home/remiboivin/workspace/synobotix/stage/llm-dataset
poetry --version
python --version
```

## 3. Lancer l'inférence classique

Le point d'entrée est `scripts/inference.py`.

```bash
poetry run python scripts/inference.py
```

## 4. Que fait ce script ?

1. **Charge le tokenizer** depuis `tokenizer/tokenizer.json`
2. **Construit le Transformer classique** avec les hyperparamètres définis dans `scripts/inference.py`
3. **Charge le checkpoint** depuis `checkpoints/student_v1/student_v1_100docs.pt` (par défaut)
4. **Génère du texte** pour les prompts de test définis dans le script

Vous verrez un affichage du type :

```text
============================================================
TEST DE GÉNÉRATION - STUDENT V1
============================================================
Chargement du tokenizer...
Vocabulaire : 5885
...
```

## 5. Modifier le prompt

Ouvrez `scripts/inference.py` et cherchez la variable `RANDOM_PROMPTS` ou `TRAIN_PROMPTS`.

```python
RANDOM_PROMPTS = [
    "Le soleil se couche",
]
```

Modifiez-la et relancez le script.

## 6. Erreurs fréquentes

**Checkpoint introuvable**

```text
FileNotFoundError: Checkpoint introuvable : checkpoints/student_v1/student_v1_100docs.pt
```

Cela signifie qu'aucun poids entraîné n'est disponible. Passez au tutoriel [Premier entraînement (classique)](./03-premier-entrainement-classique.md) pour en créer un.

## Prochaines étapes

- [Premier entraînement (classique)](./03-premier-entrainement-classique.md)
- [Lancer l'inférence (classique) - How-to](../how-to-guides/06-inference-classique.md)
