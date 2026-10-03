# Référence : Attention (classique)

## ScaledDotProductAttention

```python
def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2,-1)) / sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, v), weights
```

## MultiHeadAttention

Étapes :
1. `Q,K,V = Linear(x)` pour chaque tête (ou projeté puis split)
2. Reshape `(B,T,d_model) → (B,h,T,dk)`
3. SDPA avec mask causal
4. Concat → `(B,T,d_model)`
5. `output = Linear(concat)`

Causal mask : matrice triangulaire inférieure, empêche regarder tokens futurs.
