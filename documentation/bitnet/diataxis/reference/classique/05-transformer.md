# Référence : TransformerBlock & Transformer (classique)

## TransformerBlock

`src/llm/model/transformer_block.py`

Structure (selon implémentation) :
```text
x
├─ norm1(x) → x1
├─ mha(x1)  → attn_out
├─ x = x + attn_out (residual)
├─ norm2(x) → x2
├─ ffn(x2)  → ffn_out
└─ x = x + ffn_out
return x
```

## Transformer

`src/llm/model/transformer.py`

```text
input_ids
  → embedding → x
  → + positional encoding (si utilisé)
  → for block in blocks: x = block(x)
  → final_norm(x) → hidden
return hidden
```

Projection vocabulaire se fait typiquement via une couche linéaire externe (dans train/inference selon code).
