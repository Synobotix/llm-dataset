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
| Train loss | `6.286843880363133` |
| Validation loss | `6.280571036868626` |
| Train PPL | `537.4543791303756` |
| Validation PPL | `534.0935638789607` |
| Gradient norm | `0.9158418502794505` |
| Learning rate | `3e-05` |

---

## Durée de l'entraînement

| Mesure | Valeur |
|---|---:|
| Durée totale | `00h 01min 57.32s` |
| Durée en secondes | `117.31569955500026` |

---

## Configuration du modèle

| Paramètre | Valeur |
|---|---:|
| Vocabulaire | `500` |
| D model | `256` |
| Num heads | `2` |
| Hidden dim | `500` |
| Num blocks | `2` |
| Max sequence length | `256` |
| Batch size | `2` |
| Learning rate | `3e-05` |
| Weight decay | `0.01` |
| Gradient clip | `1.0` |

---

## Paramètres entraînables

| Mesure | Valeur |
|---|---:|
| Tenseurs entraînables | `14` |
| Tenseurs gelés | `0` |
| Paramètres entraînables | `1,165,568` |
| Paramètres gelés | `0` |
| Paramètres totaux | `1,165,568` |
| Ratio entraînable | `100.0000 %` |

### Détail par sous-module

| Sous-module | Total | Entraînable |
|---|---:|---:|
| `token_embedding` | `128,000` | `128,000` |
| `blocks` | `1,037,312` | `1,037,312` |
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
