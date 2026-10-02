import math

import torch
import torch.nn as nn

from llm.bitnet_model.bitlinear import BitLinear


class BitAttention(nn.Module):
    """
    Multi-Head Self-Attention adaptée à notre architecture BitNet.

    Les projections Q, K, V et la projection de sortie
    utilisent BitLinear.

    Architecture :

        X
        │
        ├── BitLinear → Q
        ├── BitLinear → K
        └── BitLinear → V
                │
                ▼
          Multi-Head Attention
                │
                ▼
            Causal Mask
                │
                ▼
              Softmax
                │
                ▼
               × V
                │
                ▼
          BitLinear Output
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        bias: bool = False,
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                "d_model doit être divisible par num_heads."
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # --------------------------------------------------
        # Projections Q, K, V
        # --------------------------------------------------

        self.q_proj = BitLinear(
            in_features=d_model,
            out_features=d_model,
            bias=bias,
        )

        self.k_proj = BitLinear(
            in_features=d_model,
            out_features=d_model,
            bias=bias,
        )

        self.v_proj = BitLinear(
            in_features=d_model,
            out_features=d_model,
            bias=bias,
        )

        # --------------------------------------------------
        # Projection de sortie
        # --------------------------------------------------

        self.out_proj = BitLinear(
            in_features=d_model,
            out_features=d_model,
            bias=bias,
        )

    def split_heads(self, x: torch.Tensor):
        """
        Transforme :

            [batch, sequence, d_model]

        en :

            [batch, num_heads, sequence, head_dim]
        """

        batch_size, sequence_length, _ = x.shape

        x = x.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )

        x = x.transpose(1, 2)

        return x

    def merge_heads(self, x: torch.Tensor):
        """
        Transforme :

            [batch, num_heads, sequence, head_dim]

        en :

            [batch, sequence, d_model]
        """

        batch_size, _, sequence_length, _ = x.shape

        x = x.transpose(1, 2)

        x = x.contiguous().view(
            batch_size,
            sequence_length,
            self.d_model,
        )

        return x

    def create_causal_mask(
        self,
        sequence_length: int,
        device: torch.device,
    ):
        """
        Crée le masque causal.

        Exemple pour 4 tokens :

            0  -∞ -∞ -∞
            0   0  -∞ -∞
            0   0   0  -∞
            0   0   0   0

        Un token ne peut donc pas regarder
        les tokens situés après lui.
        """

        mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        return mask

    def forward(self, x: torch.Tensor):
        """
        Forward de l'attention.

        Entrée :

            [batch, sequence, d_model]

        Sortie :

            [batch, sequence, d_model]
        """

        batch_size, sequence_length, _ = x.shape

        # --------------------------------------------------
        # 1. Projections Q, K, V
        # --------------------------------------------------

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # --------------------------------------------------
        # 2. Séparation des têtes
        # --------------------------------------------------

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # --------------------------------------------------
        # 3. Produit Q × Kᵀ
        # --------------------------------------------------

        attention_scores = torch.matmul(
            q,
            k.transpose(-2, -1),
        )

        # --------------------------------------------------
        # 4. Scaling
        # --------------------------------------------------

        attention_scores = (
            attention_scores
            / math.sqrt(self.head_dim)
        )

        # --------------------------------------------------
        # 5. Causal Mask
        # --------------------------------------------------

        causal_mask = self.create_causal_mask(
            sequence_length=sequence_length,
            device=x.device,
        )

        attention_scores = attention_scores.masked_fill(
            causal_mask,
            float("-inf"),
        )

        # --------------------------------------------------
        # 6. Softmax
        # --------------------------------------------------

        attention_weights = torch.softmax(
            attention_scores,
            dim=-1,
        )

        # --------------------------------------------------
        # 7. Attention × V
        # --------------------------------------------------

        attention_output = torch.matmul(
            attention_weights,
            v,
        )

        # --------------------------------------------------
        # 8. Fusion des têtes
        # --------------------------------------------------

        attention_output = self.merge_heads(
            attention_output
        )

        # --------------------------------------------------
        # 9. Projection finale BitLinear
        # --------------------------------------------------

        output = self.out_proj(
            attention_output
        )

        return output