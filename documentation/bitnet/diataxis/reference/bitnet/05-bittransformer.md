# Référence : BitTransformerBlock & BitTransformer

## BitTransformerBlock

**Localisation** : `src/llm/bitnet_model/bit_transformer_block.py`

Structure type **pre-norm** :

```text
x
├─ x1 = LayerNorm/RMSNorm(x)
├─ attn = BitAttention(x1)
├─ x = x + attn  (residual)
├─ x2 = LayerNorm/RMSNorm(x)
├─ mlp = BitMLP(x2)
└─ x = x + mlp   (residual)
return x
```

Composants :
- `self.attention = BitAttention(...)`
- `self.mlp = BitMLP(...)`
- `self.norm1`, `self.norm2`

## BitTransformer

**Localisation** : `src/llm/bitnet_model/bit_transformer.py`

```text
input_ids (B,T)
  → token_embedding(input_ids) → (B,T,d_model)
  → blocks: [block(x) for block in blocks] (itératif)
  → final_norm(x) → hidden_states (B,T,d_model)
return hidden_states
```

Attributs clés :
- `vocab_size`, `d_model`, `num_heads`, `hidden_dim`, `num_blocks`, `max_sequence_length`
- `token_embedding` (classique, `nn.Embedding`)
- `blocks` : `nn.ModuleList` de `BitTransformerBlock`
- `final_norm`

**Remarque** : `BitTransformer` renvoie les **hidden states**. La projection vocabulaire est assurée par `LMHead`.

## LMHead

**Localisation** : `src/llm/bitnet_model/lm_head.py`

```python
class LMHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size, bias=False)
    def forward(self, hidden_states):
        return self.linear(hidden_states)
```

Note : `LMHead` est **classique** (`nn.Linear`), non ternarisé.
