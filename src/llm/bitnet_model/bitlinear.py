import torch
import torch.nn as nn
import torch.nn.functional as F


class BitLinear(nn.Module):
    """
    Couche linéaire adaptée à une architecture BitNet.

    Les poids entraînables restent en précision flottante.
    Lors du forward, ils sont quantifiés en {-1, 0, +1}.

    Schéma :

    
        X
        │
        ├── quantification des activations
        │
        ▼
        X_q
        │
        │        W_float
        │           │
        │           ▼
        │      quantification
        │           │
        │           ▼
        │      W_ternary
        │
        └───────────┬───────────
                    │
                    ▼
               Linear
                    │
                    ▼
                  Y
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Poids flottants entraînables.
        #
        # Ils ne sont PAS directement utilisés comme poids
        # pendant le calcul BitNet.
        #
        # Ils servent de "poids latents" à partir desquels
        # nous construisons les poids ternaires.
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialisation des poids.
        """

        nn.init.kaiming_uniform_(
            self.weight,
            a=5 ** 0.5
        )

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @staticmethod
    def quantize_weights(weight: torch.Tensor):
        """
        Quantification des poids flottants vers {-1, 0, +1}.

        On calcule d'abord une échelle basée sur
        la moyenne de la valeur absolue des poids.

        Puis :

            poids > 0  -> +1
            poids < 0  -> -1

        avec une zone proche de zéro qui devient 0.

        La valeur retournée est :

            scale * poids_ternaires
        """

        # Moyenne de la valeur absolue des poids.
        scale = weight.abs().mean()

        # Protection contre le cas où scale == 0.
        scale = torch.clamp(
            scale,
            min=1e-5
        )

        # Normalisation.
        normalized_weight = weight / scale

        # Quantification.
        #
        # Les valeurs proches de zéro deviennent 0.
        # Les autres deviennent -1 ou +1.
        quantized_weight = torch.where(
            normalized_weight > 0.5,
            torch.ones_like(normalized_weight),
            torch.where(
                normalized_weight < -0.5,
                -torch.ones_like(normalized_weight),
                torch.zeros_like(normalized_weight)
            )
        )

        # On conserve l'échelle.
        quantized_weight = quantized_weight * scale

        return quantized_weight

    @staticmethod
    def quantize_activations(x: torch.Tensor):
        """
        Quantification simple des activations.

        Pour cette première implémentation pédagogique,
        nous utilisons une quantification symétrique.

        Les activations sont normalisées par leur maximum
        absolu puis quantifiées sur une plage entière.

        Le nombre de niveaux est volontairement paramétrable.
        """

        # Valeur maximale absolue.
        scale = x.abs().amax(
            dim=-1,
            keepdim=True
        )

        # Protection contre division par zéro.
        scale = torch.clamp(
            scale,
            min=1e-5
        )

        # Normalisation.
        x_normalized = x / scale

        # Quantification sur 8 bits signés.
        qmax = 127

        x_quantized = torch.round(
            x_normalized * qmax
        )

        x_quantized = torch.clamp(
            x_quantized,
            -qmax,
            qmax
        )

        # Retour vers l'échelle originale.
        x_quantized = (
            x_quantized / qmax
        ) * scale

        return x_quantized

    @staticmethod
    def straight_through_estimator(
        original: torch.Tensor,
        quantized: torch.Tensor,
    ):
        """
        Straight-Through Estimator (STE).

        Forward :
            utilise la valeur quantifiée.

        Backward :
            laisse approximativement passer le gradient
            comme si la quantification n'existait pas.

        Formule :

            original + (quantized - original).detach()
        """

        return (
            original
            + (quantized - original).detach()
        )

    def forward(self, x: torch.Tensor):
        """
        Forward de BitLinear.
        """

        # --------------------------------------------------
        # 1. Quantification des activations
        # --------------------------------------------------

        x_quantized = self.quantize_activations(x)

        # --------------------------------------------------
        # 2. Quantification des poids
        # --------------------------------------------------

        weight_quantized = self.quantize_weights(
            self.weight
        )

        # --------------------------------------------------
        # 3. STE
        # --------------------------------------------------

        x_quantized = self.straight_through_estimator(
            x,
            x_quantized
        )

        weight_quantized = self.straight_through_estimator(
            self.weight,
            weight_quantized
        )

        # --------------------------------------------------
        # 4. Calcul linéaire
        # --------------------------------------------------

        output = F.linear(
            x_quantized,
            weight_quantized,
            self.bias
        )

        return output