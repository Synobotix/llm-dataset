# Résultats — BitNet

## Informations générales

| Information | Valeur |
|---|---|
| Device | `cpu` |
| Checkpoint | `checkpoints/bitnet/entrainement_bitnet_3.pt` |
| Époque finale | `5` |

---

## Résultats d'entraînement

| Mesure | Valeur |
|---|---:|
| Train loss | `2.946970970734306` |
| Validation loss | `4.460208114824797` |
| Train PPL | `19.048168795547426` |
| Validation PPL | `86.50551030222607` |
| Gradient norm | `1.923104117909872` |
| Learning rate | `0.001` |

---

## Durée de l'entraînement

| Mesure | Valeur |
|---|---:|
| Durée totale | `00h 07min 16.54s` |
| Durée en secondes | `436.5365336120012` |

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
checkpoints/bitnet/entrainement_bitnet_3.pt
```

Le rapport Markdown est enregistré dans :

```text
/home/msspr/liste_projet/stage/llm/documentation/result/bitnet/result_entrainement_bitnet1.md
```

Ce fichier a été généré automatiquement à partir du checkpoint BitNet.
