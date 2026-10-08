# Résultats — BitNet

## Informations générales

| Information | Valeur |
|---|---|
| Device | `cpu` |
| Checkpoint | `checkpoints/bitnet/entrainement_bitnet_5.pt` |
| Époque finale | `8` |

---

## Résultats d'entraînement

| Mesure | Valeur |
|---|---:|
| Train loss | `2.243579738813898` |
| Validation loss | `4.572648709280449` |
| Train PPL | `9.427017217487437` |
| Validation PPL | `96.80016601194075` |
| Gradient norm | `1.9570119390288256` |
| Learning rate | `0.001` |

---

## Durée de l'entraînement

| Mesure | Valeur |
|---|---:|
| Durée totale | `00h 10min 28.03s` |
| Durée en secondes | `628.0295706870002` |

---

## Configuration du modèle

| Paramètre | Valeur |
|---|---:|
| Vocabulaire | `1000` |
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
| Tenseurs entraînables | `14` |
| Tenseurs gelés | `0` |
| Paramètres entraînables | `1,477,888` |
| Paramètres gelés | `0` |
| Paramètres totaux | `1,477,888` |
| Ratio entraînable | `100.0000 %` |

### Détail par sous-module

| Sous-module | Total | Entraînable |
|---|---:|---:|
| `token_embedding` | `256,000` | `256,000` |
| `blocks` | `1,221,632` | `1,221,632` |
| `final_norm` | `256` | `256` |

---

## Checkpoint

Le checkpoint utilisé pour générer ce rapport est :

```text
checkpoints/bitnet/entrainement_bitnet_5.pt
```

Le rapport Markdown est enregistré dans :

```text
/home/msspr/liste_projet/stage/llm/documentation/result/bitnet/result_entrainement_bitnet2.md
```

Ce fichier a été généré automatiquement à partir du checkpoint BitNet.
