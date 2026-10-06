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
| Train loss | `3.9845212475198215` |
| Validation loss | `4.9314943885031015` |
| Train PPL | `53.7595458000716` |
| Validation PPL | `138.58645968361898` |
| Gradient norm | `2.1516026154258485` |
| Learning rate | `0.001` |

---

## Durée de l'entraînement

| Mesure | Valeur |
|---|---:|
| Durée totale | `00h 03min 17.49s` |
| Durée en secondes | `197.4883842229974` |

---

## Configuration du modèle

| Paramètre | Valeur |
|---|---:|
| Vocabulaire | `7310` |
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
| Paramètres entraînables | `3,093,248` |
| Paramètres gelés | `0` |
| Paramètres totaux | `3,093,248` |
| Ratio entraînable | `100.0000 %` |

### Détail par sous-module

| Sous-module | Total | Entraînable |
|---|---:|---:|
| `token_embedding` | `1,871,360` | `1,871,360` |
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
