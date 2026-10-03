import torch
import torch.nn as nn
import torch.nn.functional as F


class BitLinear(nn.Module):
    """
    BitLinear inspiré de BitNet b1.58.

    Principe :

        Poids maître FP32
              │
              ▼
        RoundClip
              │
              ▼
        {-1, 0, +1} × scale
              │
              ▼
             STE
              │
              ▼
           calcul

    Les activations sont quantifiées en 8 bits.

    Optimisations :

    - poids maître conservés en FP32 pour l'entraînement ;
    - ternarisation par RoundClip ;
    - STE pour les poids et activations ;
    - quantification activation absmax ;
    - buffer ternaire INT8 pour l'inférence ;
    - préparation des poids ternaires une seule fois
      pour l'inférence ;
    - réduction des opérations inutiles ;
    - compatible avec F.linear().

    IMPORTANT :

    Cette implémentation reste une version PyTorch
    de référence.

    F.linear() n'utilise pas encore un kernel matériel
    spécialisé INT8 × ternaire.
    """

    ACTIVATION_BITS = 8
    ACTIVATION_QMAX = 128
    MIN_SCALE = 1e-5

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # ==================================================
        # POIDS MAÎTRE FP32
        # ==================================================

        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                dtype=torch.float32,
            )
        )

        # ==================================================
        # BIAS
        # ==================================================

        if bias:

            self.bias = nn.Parameter(
                torch.zeros(
                    out_features,
                    dtype=torch.float32,
                )
            )

        else:

            self.register_parameter(
                "bias",
                None,
            )

        self.reset_parameters()

        # ==================================================
        # BUFFER POIDS TERNAIRES
        # ==================================================

        self.register_buffer(
            "weight_ternary",
            torch.empty(
                0,
                dtype=torch.int8,
            ),
            persistent=False,
        )

        # ==================================================
        # SCALE POIDS
        # ==================================================

        self.register_buffer(
            "weight_scale",
            torch.tensor(
                1.0,
                dtype=torch.float32,
            ),
            persistent=False,
        )

    # ======================================================
    # INITIALISATION
    # ======================================================

    def reset_parameters(self):

        nn.init.kaiming_uniform_(
            self.weight,
            a=5 ** 0.5,
        )

        if self.bias is not None:

            nn.init.zeros_(
                self.bias
            )

    # ======================================================
    # QUANTIFICATION DES POIDS
    # ======================================================

    @classmethod
    def quantize_weights(
        cls,
        weight: torch.Tensor,
    ):
        """
        Quantification :

            scale = mean(abs(weight))

            weight / scale
                ↓
             round
                ↓
            clamp(-1,1)

        Résultat :

            {-1, 0, +1} × scale
        """

        scale = (
            weight.abs()
            .mean()
            .clamp(
                min=cls.MIN_SCALE
            )
        )

        weight_scaled = (
            weight / scale
        )

        weight_ternary = torch.clamp(
            torch.round(
                weight_scaled
            ),
            -1,
            1,
        )

        weight_quantized = (
            weight_ternary
            * scale
        )

        return (
            weight_quantized,
            scale,
        )

    # ======================================================
    # QUANTIFICATION DES ACTIVATIONS
    # ======================================================

    @classmethod
    def quantize_activations(
        cls,
        x: torch.Tensor,
        bits: int = ACTIVATION_BITS,
    ):
        """
        Quantification absmax des activations.

        Pour 8 bits :

            [-127, +127]

        Les valeurs sont ensuite déquantifiées
        en FP32 pour F.linear().
        """

        if bits == 8:

            qmax = cls.ACTIVATION_QMAX

        else:

            qmax = 2 ** (
                bits - 1
            )

        # --------------------------------------------------
        # ABSMAX
        # --------------------------------------------------

        scale = (
            x.abs()
            .max()
            .clamp(
                min=cls.MIN_SCALE
            )
        )

        # --------------------------------------------------
        # MISE À L'ÉCHELLE
        # --------------------------------------------------

        x_scaled = (
            x
            * qmax
            / scale
        )

        # --------------------------------------------------
        # CLIPPING
        # --------------------------------------------------

        x_clipped = torch.clamp(
            x_scaled,
            -qmax + 1,
            qmax - 1,
        )

        # --------------------------------------------------
        # ARRONDI
        # --------------------------------------------------

        x_quantized = torch.round(
            x_clipped
        )

        # --------------------------------------------------
        # DÉQUANTIFICATION
        # --------------------------------------------------

        x_dequantized = (
            x_quantized
            / qmax
        ) * scale

        return x_dequantized

    # ======================================================
    # STE
    # ======================================================

    @staticmethod
    def ste(
        original: torch.Tensor,
        quantized: torch.Tensor,
    ):
        """
        Straight-Through Estimator.

        Forward :
            utilise quantized

        Backward :
            gradient de original
        """

        return (
            original
            + (
                quantized
                - original
            ).detach()
        )

    # ======================================================
    # PRÉPARATION INFÉRENCE
    # ======================================================

    @torch.no_grad()
    def prepare_for_inference(self):
        """
        Prépare les poids ternaires pour l'inférence.

        Le poids est stocké sous forme :

            INT8 {-1, 0, +1}

        avec un scale séparé.
        """

        weight_quantized, scale = (
            self.quantize_weights(
                self.weight
            )
        )

        self.weight_ternary = (
            torch.round(
                weight_quantized
                / scale
            )
            .clamp(
                -1,
                1,
            )
            .to(torch.int8)
        )

        self.weight_scale = (
            scale.detach()
        )

    # ======================================================
    # FORWARD
    # ======================================================

    def forward(
        self,
        x: torch.Tensor,
    ):

        # ==================================================
        # ENTRAÎNEMENT
        # ==================================================

        if self.training:

            weight_quantized, _ = (
                self.quantize_weights(
                    self.weight
                )
            )

            # --------------------------------------------------
            # STE POIDS
            # --------------------------------------------------

            weight_used = self.ste(
                self.weight,
                weight_quantized,
            )

        # ==================================================
        # INFÉRENCE
        # ==================================================

        else:

            if (
                self.weight_ternary.numel()
                == 0
            ):

                self.prepare_for_inference()

            weight_used = (
                self.weight_ternary.to(
                    dtype=x.dtype
                )
                * self.weight_scale.to(
                    dtype=x.dtype
                )
            )

        # ==================================================
        # QUANTIFICATION ACTIVATIONS
        # ==================================================

        x_quantized = (
            self.quantize_activations(
                x
            )
        )

        # ==================================================
        # STE ACTIVATIONS
        # ==================================================

        x_used = self.ste(
            x,
            x_quantized,
        )

        # ==================================================
        # LINEAR
        # ==================================================

        return F.linear(
            x_used,
            weight_used,
            self.bias,
        )