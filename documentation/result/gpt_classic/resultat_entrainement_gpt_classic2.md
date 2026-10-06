# Résultats — GPT Classic V1

## Informations générales

| Information | Valeur |
|---|---|
| Device | `cpu` |
| Checkpoint | `checkpoints/gpt_classic/entrainement_gpt_classic_2.pt` |
| Époque finale | `2` |
| Global step | `84` |

---

## Résultats d'entraînement

| Mesure | Valeur |
|---|---:|
| Train loss | `4.891633453823271` |
| Validation loss | `6.611825466156006` |
| Train PPL | `133.1709250432113` |
| Validation PPL | `743.8396341897918` |
| Gradient norm | `1.4306110297617962` |
| Learning rate | `0.0009999999999999992` |

---

## Durée de l'entraînement

| Mesure | Valeur |
|---|---:|
| Durée totale | `00h 00min 04.10s` |
| Durée en secondes | `4.1034400540002025` |

---

## Configuration du modèle

| Paramètre | Valeur |
|---|---:|
| Vocabulaire | `1308` |
| D model | `256` |
| Num heads | `4` |
| Hidden dim | `680` |
| Num blocks | `2` |
| Max sequence length | `128` |
| Batch size | `2` |
| Learning rate | `0.001` |
| Weight decay | `0.01` |
| Gradient clip | `1.0` |

---

## Paramètres entraînables

| Mesure | Valeur |
|---|---:|
| Tenseurs entraînables | `37` |
| Tenseurs gelés | `0` |
| Paramètres entraînables | `1,898,092` |
| Paramètres gelés | `0` |
| Paramètres totaux | `1,898,092` |
| Ratio entraînable | `100.0000 %` |

### Détail par sous-module

| Sous-module | Total | Entraînable |
|---|---:|---:|
| `token_embedding` | `334,848` | `334,848` |
| `positional_encoding` | `0` | `0` |
| `transformer_blocks` | `1,226,576` | `1,226,576` |
| `final_layer_norm` | `512` | `512` |
| `output_projection` | `336,156` | `336,156` |

---

## Checkpoint

Le checkpoint utilisé pour générer ce rapport est :

```text
checkpoints/gpt_classic/entrainement_gpt_classic_2.pt
```

Le rapport Markdown est enregistré dans :

```text
/home/msspr/liste_projet/stage/llm/documentation/result/gpt_classic/resultat_entrainement_gpt_classic2.md
```

Ce fichier a été généré automatiquement à partir du checkpoint GPT Classic.
