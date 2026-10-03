import torch
import torch.nn as nn
import torch.nn.functional as F

from llm.bitnet_model.bitlinear import BitLinear


class BitAttention(nn.Module):
    """
    Multi-Head Self-Attention optimisée pour notre architecture BitNet.

    Optimisations :

    - projection Q/K/V fusionnée dans une seule BitLinear ;
    - poids de la projection Q/K/V ternaires via BitLinear ;
    - activations quantifiées via BitLinear ;
    - scaled_dot_product_attention de PyTorch ;
    - causalité native avec is_causal=True ;
    - pas de masque causal créé manuellement ;
    - pas de tenseur attention_scores explicite ;
    - pas de tenseur attention_weights explicite ;
    - projection de sortie BitLinear.
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
        # Projection Q / K / V fusionnée
        # --------------------------------------------------
        #
        # Au lieu de :
        #
        # X → BitLinear → Q
        # X → BitLinear → K
        # X → BitLinear → V
        #
        # On fait :
        #
        # X → BitLinear → [Q | K | V]
        #
        # d_model → 3 × d_model
        # --------------------------------------------------

        self.qkv_proj = BitLinear(
            in_features=d_model,
            out_features=3 * d_model,
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

    # ======================================================
    # QKV → MULTI-HEAD
    # ======================================================

    def split_heads(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Transforme :

            [B, S, D]

        en :

            [B, H, S, Hd]
        """

        batch_size, sequence_length, _ = x.shape

        return (
            x.view(
                batch_size,
                sequence_length,
                self.num_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )

    # ======================================================
    # MULTI-HEAD → D_MODEL
    # ======================================================

    def merge_heads(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Transforme :

            [B, H, S, Hd]

        en :

            [B, S, D]
        """

        batch_size, _, sequence_length, _ = x.shape

        return (
            x.transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                sequence_length,
                self.d_model,
            )
        )

    # ======================================================
    # FORWARD
    # ======================================================

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Entrée :

            [B, S, D]

        Sortie :

            [B, S, D]
        """

        # --------------------------------------------------
        # 1. Projection Q / K / V fusionnée
        # --------------------------------------------------

        qkv = self.qkv_proj(x)

        # --------------------------------------------------
        # 2. Séparation Q / K / V
        # --------------------------------------------------

        q, k, v = qkv.chunk(
            3,
            dim=-1,
        )

        # --------------------------------------------------
        # 3. Séparation des têtes
        # --------------------------------------------------

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # --------------------------------------------------
        # 4. Scaled Dot-Product Attention
        # --------------------------------------------------

        attention_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )

        # --------------------------------------------------
        # 5. Fusion des têtes
        # --------------------------------------------------

        attention_output = self.merge_heads(
            attention_output
        )

        # --------------------------------------------------
        # 6. Projection de sortie BitLinear
        # --------------------------------------------------

        return self.out_proj(
            attention_output
        )