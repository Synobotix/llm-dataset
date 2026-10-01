import math

import torch
import torch.nn as nn


def debug_tensor(
    name: str,
    tensor: torch.Tensor,
):
    """ print(f"\n--- {name} ---")
    print("shape :", tensor.shape)
    print("dtype :", tensor.dtype)
    print("min   :", tensor.min().item())
    print("max   :", tensor.max().item())
    print("mean  :", tensor.mean().item())
    print("std   :", tensor.std().item())
    print("NaN   :", torch.isnan(tensor).any().item())
    print("Inf   :", torch.isinf(tensor).any().item()) """


class MultiHeadCausalSelfAttention(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        max_sequence_length: int,
    ):
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim doit être divisible par num_heads."
            )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # Projection Query
        self.query = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        # Projection Key
        self.key = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        # Projection Value
        self.value = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        # Projection finale
        self.output = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        # --------------------------------------------------
        # Masque causal
        # --------------------------------------------------

        mask = torch.tril(
            torch.ones(
                max_sequence_length,
                max_sequence_length,
            )
        )

        self.register_buffer(
            "causal_mask",
            mask,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # --------------------------------------------------
        # Entrée
        # --------------------------------------------------

        debug_tensor(
            "ATTENTION - entrée x",
            x,
        )

        batch_size, sequence_length, _ = x.shape

        print("\n--- DIMENSIONS ---")
        print("batch_size      :", batch_size)
        print("sequence_length :", sequence_length)
        print("embedding_dim   :", self.embedding_dim)
        print("num_heads       :", self.num_heads)
        print("head_dim        :", self.head_dim)

        # --------------------------------------------------
        # Q, K, V
        # --------------------------------------------------

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        debug_tensor(
            "Q après Linear",
            q,
        )

        debug_tensor(
            "K après Linear",
            k,
        )

        debug_tensor(
            "V après Linear",
            v,
        )

        # --------------------------------------------------
        # Séparation des têtes
        # --------------------------------------------------

        q = q.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )

        k = k.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )

        v = v.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )

        print("\n--- APRÈS VIEW ---")
        print("Q :", q.shape)
        print("K :", k.shape)
        print("V :", v.shape)

        # --------------------------------------------------
        # [batch, sequence, heads, head_dim]
        # →
        # [batch, heads, sequence, head_dim]
        # --------------------------------------------------

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        print("\n--- APRÈS TRANSPOSE ---")
        print("Q :", q.shape)
        print("K :", k.shape)
        print("V :", v.shape)

        # --------------------------------------------------
        # Scores d'attention
        # --------------------------------------------------

        scores = torch.matmul(
            q,
            k.transpose(-2, -1),
        )

        debug_tensor(
            "Scores attention",
            scores,
        )

        # --------------------------------------------------
        # Scaling
        # --------------------------------------------------

        scores = scores / math.sqrt(
            self.head_dim
        )

        debug_tensor(
            "Scores après scaling",
            scores,
        )

        # --------------------------------------------------
        # Masque causal
        # --------------------------------------------------

        mask = self.causal_mask[
            :sequence_length,
            :sequence_length,
        ]

        print("\n--- MASQUE CAUSAL ---")
        print("mask shape :", mask.shape)
        print("mask dtype :", mask.dtype)
        print("mask min   :", mask.min().item())
        print("mask max   :", mask.max().item())

        # Affichage du masque pour les petites séquences
        if sequence_length <= 10:
            print("\nMatrice du masque :")
            print(mask.to(torch.int32))

        # --------------------------------------------------
        # Application du masque
        # --------------------------------------------------

        scores = scores.masked_fill(
            mask == 0,
            float("-inf"),
        )

        print("\n--- SCORES APRÈS MASQUE ---")

        print(
            "NaN :",
            torch.isnan(scores).any().item(),
        )

        print(
            "Inf :",
            torch.isinf(scores).any().item(),
        )

        # --------------------------------------------------
        # Vérification des positions futures
        # --------------------------------------------------

        future_positions = (
            mask == 0
        )

        future_scores = scores[
            :,
            :,
            future_positions,
        ]

        print(
            "Nombre de scores futurs :",
            future_scores.numel(),
        )

        if future_scores.numel() > 0:

            all_future_inf = torch.isinf(
                future_scores
            ).all()

            print(
                "Tous les scores futurs sont -inf :",
                all_future_inf.item(),
            )

        # --------------------------------------------------
        # Softmax
        # --------------------------------------------------

        attention_weights = torch.softmax(
            scores,
            dim=-1,
        )

        debug_tensor(
            "Attention weights",
            attention_weights,
        )

        # --------------------------------------------------
        # Vérification causalité après Softmax
        # --------------------------------------------------

        future_weights = attention_weights[
            :,
            :,
            future_positions,
        ]

        print("\n--- VÉRIFICATION CAUSALITÉ ---")

        print(
            "Nombre de poids futurs :",
            future_weights.numel(),
        )

        if future_weights.numel() > 0:

            max_future_weight = (
                future_weights.max().item()
            )

            print(
                "Poids futurs max :",
                max_future_weight,
            )

            all_future_zero = torch.all(
                future_weights == 0
            )

            print(
                "Tous les poids futurs sont zéro :",
                all_future_zero.item(),
            )

        # --------------------------------------------------
        # Vérification somme des poids
        # --------------------------------------------------

        attention_sum = attention_weights.sum(
            dim=-1
        )

        print("\n--- SOMME DES POIDS ---")

        print(
            "Min somme :",
            attention_sum.min().item(),
        )

        print(
            "Max somme :",
            attention_sum.max().item(),
        )

        sums_are_one = torch.allclose(
            attention_sum,
            torch.ones_like(attention_sum),
            atol=1e-6,
        )

        # IMPORTANT :
        # torch.allclose() retourne déjà un bool Python.
        # Il ne faut donc PAS mettre .item() ici.

        print(
            "Sommes proches de 1 :",
            sums_are_one,
        )

        # --------------------------------------------------
        # Affichage des poids pour petite séquence
        # --------------------------------------------------

        if sequence_length <= 10:

            print("\n--- POIDS ATTENTION HEAD 0 ---")

            print(
                attention_weights[0, 0]
            )

        # --------------------------------------------------
        # Application aux valeurs V
        # --------------------------------------------------

        attention_output = torch.matmul(
            attention_weights,
            v,
        )

        debug_tensor(
            "Attention output avant transpose",
            attention_output,
        )

        # --------------------------------------------------
        # Retour à :
        # [batch, sequence, heads, head_dim]
        # --------------------------------------------------

        attention_output = attention_output.transpose(
            1,
            2,
        )

        print(
            "\nAttention output après transpose :",
            attention_output.shape,
        )

        # --------------------------------------------------
        # Fusion des têtes
        # --------------------------------------------------

        attention_output = (
            attention_output
            .contiguous()
            .view(
                batch_size,
                sequence_length,
                self.embedding_dim,
            )
        )

        debug_tensor(
            "Attention output après reshape",
            attention_output,
        )

        # --------------------------------------------------
        # Projection finale
        # --------------------------------------------------

        output = self.output(
            attention_output
        )

        debug_tensor(
            "Sortie attention finale",
            output,
        )

        return output