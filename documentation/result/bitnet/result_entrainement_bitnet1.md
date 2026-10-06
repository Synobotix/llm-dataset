# Résultats — BitNet

## Informations générales

| Information | Valeur |
|---|---|
| Device | `cpu` |
| Checkpoint | `checkpoints/bitnet/entrainement_bitnet_1.pt` |
| Époque finale | `2` |

---

## Résultats d'entraînement

| Mesure | Valeur |
|---|---:|
| Train loss | `5.654756999597317` |
| Validation loss | `6.8418169021606445` |
| Train PPL | `285.6470619411382` |
| Validation PPL | `936.1885534284512` |
| Gradient norm | `1.7902279749092471` |
| Learning rate | `0.001` |

---

## Durée de l'entraînement

| Mesure | Valeur |
|---|---:|
| Durée totale | `N/A` |
| Durée en secondes | `N/A` |

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
| Batch size | `N/A` |
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
checkpoints/bitnet/entrainement_bitnet_1.pt
```

Le rapport Markdown est enregistré dans :

```text
/home/msspr/liste_projet/stage/llm/documentation/result/bitnet/result_entrainement_bitnet1.md
```

Ce fichier a été généré automatiquement à partir du checkpoint BitNet.
