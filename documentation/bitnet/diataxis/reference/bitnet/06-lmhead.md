# Référence : LMHead

## Localisation

`src/llm/bitnet_model/lm_head.py`

## Définition

```python
class LMHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size, bias=False)
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden_states)  # (B,T,vocab_size)
```

## Rôle

Projette les `hidden_states` (sortie finale des blocks + final_norm) vers l'espace du vocabulaire pour produire des **logits** par token.

## Particularité BitNet

Dans cette implémentation, **LMHead reste classique** (`nn.Linear`), non ternarisé. Seuls les modules Attention/MLP utilisent `BitLinear`.

## Usage

Après `BitTransformer` :
```python
hidden = transformer(input_ids)  # (B,T,d_model)
logits = lm_head(hidden)          # (B,T,vocab_size)
```

Perte : `CrossEntropyLoss(logits.view(-1,vocab), targets.view(-1))` (causal, next-token).
