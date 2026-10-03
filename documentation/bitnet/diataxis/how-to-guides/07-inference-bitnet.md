# How-to : Lancer l'inférence BitNet

## 1. Depuis CLI

```bash
cd /home/remiboivin/workspace/synobotix/stage/llm-dataset
poetry run python scripts/bitnet/inference.py
```

## 2. Configuration

**Prompt** : `scripts/bitnet/prompt.py`
```python
PROMPT = "il était une fois ..."
```

**Génération** : `scripts/bitnet/inference.py`
```python
CHECKPOINT_FILE = Path("checkpoint/optiminisation_bitnet/bitnet_50docs1.pt")
MAX_NEW_TOKENS = 50
TEMPERATURE = 1.0
```

## 3. Détails du chargement

`llm.inference.generate_bitnet.create_bitnet_model()` :
- Instancie `BitTransformer` + `LMHead` selon config checkpoint ou défaut
- Charge `transformer_state_dict` + `lm_head_state_dict`
- **Remappe QKV** si `qkv_proj` présent (checkpoint fusionné) vers `q_proj/k_proj/v_proj` (modèle séparé)
- Déplace sur CPU/CUDA

## 4. Génération

Implémentation greedy par défaut : à chaque pas, prend `argmax` du dernier logit. Contexte tronqué à `max_sequence_length`.

## 5. Avec un autre checkpoint

Changez `CHECKPOINT_FILE` pour pointer vers votre checkpoint (ex. `bitnet_200docs1.pt`).

## 6. Déboguer QKV mismatch

Si erreur persiste, vérifiez shapes dans le checkpoint :
```python
ckpt = torch.load("checkpoint/...pt", map_location="cpu")
sd = ckpt["transformer_state_dict"]
for k,v in sd.items():
    if "qkv" in k:
        print(k, v.shape)
```

Le remapping chunk sur dim=0 est correct dans la majorité des cas (concat sortie).
