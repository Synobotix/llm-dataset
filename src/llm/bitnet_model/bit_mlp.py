import torch
import torch.nn as nn

from llm.bitnet_model.bitlinear import BitLinear


class BitMLP(nn.Module):
    """
    MLP adapté à notre architecture BitNet.

    Architecture :

        X
        │
        ▼
    BitLinear
        │
        ▼
      GELU
        │
        ▼
    BitLinear
        │
        ▼
      Output

    Dimensions :

        d_model → hidden_dim → d_model
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        bias: bool = False,
    ):
        super().__init__()

        self.d_model = d_model
        self.hidden_dim = hidden_dim

        # Première projection :
        #
        # d_model → hidden_dim
        self.fc1 = BitLinear(
            in_features=d_model,
            out_features=hidden_dim,
            bias=bias,
        )

        # Fonction d'activation.
        #
        # GELU est couramment utilisée dans les
        # architectures Transformer.
        self.activation = nn.GELU()

        # Deuxième projection :
        #
        # hidden_dim → d_model
        self.fc2 = BitLinear(
            in_features=hidden_dim,
            out_features=d_model,
            bias=bias,
        )

    def forward(self, x: torch.Tensor):
        """
        Forward du BitMLP.

        Entrée :
            [batch, sequence, d_model]

        Sortie :
            [batch, sequence, d_model]
        """

        # Projection vers la dimension cachée.
        x = self.fc1(x)

        # Activation.
        x = self.activation(x)

        # Projection vers d_model.
        x = self.fc2(x)

        return x