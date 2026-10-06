# Résultats — BitNet

## Informations générales

| Information | Valeur |
|---|---|
| Device | `cpu` |
| Checkpoint | `checkpoints/bitnet/entrainement_bitnet_2.pt` |
| Époque finale | `4` |

---

## Résultats d'entraînement

| Mesure | Valeur |
|---|---:|
| Train loss | `4.351648348133739` |
| Validation loss | `6.6595659255981445` |
| Train PPL | `77.60627971937528` |
| Validation PPL | `780.2121934709468` |
| Gradient norm | `1.5114598922917317` |
| Learning rate | `0.001` |

---

## Durée de l'entraînement

| Mesure | Valeur |
|---|---:|
| Durée totale | `00h 00min 04.41s` |
| Durée en secondes | `4.413935022001169` |

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
| Tenseurs entraînables | `14` |
| Tenseurs gelés | `0` |
| Paramètres entraînables | `1,556,736` |
| Paramètres gelés | `0` |
| Paramètres totaux | `1,556,736` |
| Ratio entraînable | `100.0000 %` |

### Détail par sous-module

| Sous-module | Total | Entraînable |
|---|---:|---:|
| `token_embedding` | `334,848` | `334,848` |
| `blocks` | `1,221,632` | `1,221,632` |
| `final_norm` | `256` | `256` |

---

## Checkpoint

Le checkpoint utilisé pour générer ce rapport est :

```text
checkpoints/bitnet/entrainement_bitnet_2.pt
```

Le rapport Markdown est enregistré dans :

```text
/home/msspr/liste_projet/stage/llm/documentation/result/bitnet/result_entrainement_bitnet2.md
```

Ce fichier a été généré automatiquement à partir du checkpoint BitNet.
