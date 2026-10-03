
import torch

from llm.bitnet_model.bitlinear import BitLinear


def test_bitlinear():
    print("=" * 60)
    print("TEST BITLINEAR")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Configuration
    # --------------------------------------------------

    batch_size = 2
    sequence_length = 8
    in_features = 256
    out_features = 128

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device : {device}")

    # --------------------------------------------------
    # 2. Création du modèle
    # --------------------------------------------------

    model = BitLinear(
        in_features=in_features,
        out_features=out_features,
        bias=False,
    ).to(device)

    print("\n[1] Modèle créé")
    print(f"Input features  : {model.in_features}")
    print(f"Output features : {model.out_features}")

    # --------------------------------------------------
    # 3. Vérification du poids maître
    # --------------------------------------------------

    print("\n[2] Vérification du poids maître")

    print(f"Shape  : {model.weight.shape}")
    print(f"Dtype  : {model.weight.dtype}")
    print(f"Requires grad : {model.weight.requires_grad}")

    assert model.weight.shape == (
        out_features,
        in_features,
    )

    assert model.weight.dtype == torch.float32

    assert model.weight.requires_grad

    print("✓ Poids maître FP32")
    print("✓ Gradient activé")

    # --------------------------------------------------
    # 4. Création des activations
    # --------------------------------------------------

    x = torch.randn(
        batch_size,
        sequence_length,
        in_features,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )

    print("\n[3] Activation créée")
    print(f"Shape : {x.shape}")
    print(f"Dtype : {x.dtype}")

    # --------------------------------------------------
    # 5. Test quantification des poids
    # --------------------------------------------------

    print("\n[4] Test quantification des poids")

    weight_quantized, scale = (
        model.quantize_weights(model.weight)
    )

    # On récupère le pattern ternaire
    weight_ternary = (
        weight_quantized / scale
    )

    unique_values = torch.unique(
        weight_ternary
    )

    print(
        "Valeurs uniques des poids ternaires :",
        unique_values.detach().cpu().tolist(),
    )

    # Vérification
    valid_values = torch.tensor(
        [-1.0, 0.0, 1.0],
        device=device,
    )

    for value in unique_values:
        assert torch.any(
            torch.isclose(value, valid_values)
        )

    print("✓ Poids ternaires valides : {-1, 0, +1}")

    print(f"Scale : {scale.item():.8f}")

    assert scale.item() > 0

    # --------------------------------------------------
    # 6. Test quantification des activations
    # --------------------------------------------------

    print("\n[5] Test quantification des activations")

    x_quantized = model.quantize_activations(x)

    print(
        f"Shape avant : {x.shape}"
    )

    print(
        f"Shape après : {x_quantized.shape}"
    )

    print(
        f"Dtype après : {x_quantized.dtype}"
    )

    assert x_quantized.shape == x.shape

    # Vérification que la quantification
    # a effectivement modifié certaines valeurs.
    difference = (
        x_quantized.detach() - x.detach()
    ).abs()

    print(
        "Erreur maximale de quantification :",
        difference.max().item(),
    )

    print(
        "Erreur moyenne de quantification :",
        difference.mean().item(),
    )

    assert torch.isfinite(
        x_quantized
    ).all()

    print("✓ Activation quantifiée correctement")

    # --------------------------------------------------
    # 7. Test STE
    # --------------------------------------------------

    print("\n[6] Test STE")

    x_ste = model.ste(
        x,
        x_quantized,
    )

    assert x_ste.shape == x.shape

    print("✓ STE activation fonctionnel")

    # --------------------------------------------------
    # 8. Test forward
    # --------------------------------------------------

    print("\n[7] Test forward")

    model.train()

    output = model(x)

    print(f"Input  : {x.shape}")
    print(f"Output : {output.shape}")

    expected_shape = (
        batch_size,
        sequence_length,
        out_features,
    )

    assert output.shape == expected_shape

    assert torch.isfinite(
        output
    ).all()

    print("✓ Forward entraînement fonctionnel")

    # --------------------------------------------------
    # 9. Test backward
    # --------------------------------------------------

    print("\n[8] Test backward")

    loss = output.mean()

    print(f"Loss : {loss.item():.6f}")

    loss.backward()

    print(
        "Gradient weight disponible :",
        model.weight.grad is not None,
    )

    assert model.weight.grad is not None

    assert torch.isfinite(
        model.weight.grad
    ).all()

    print(
        "Gradient moyen :",
        model.weight.grad.abs().mean().item(),
    )

    print("✓ Backward fonctionnel")
    print("✓ Gradient du poids maître disponible")

    # --------------------------------------------------
    # 10. Test préparation inférence
    # --------------------------------------------------

    print("\n[9] Préparation inférence")

    model.eval()

    model.prepare_for_inference()

    print(
        "weight_ternary shape :",
        model.weight_ternary.shape,
    )

    print(
        "weight_ternary dtype :",
        model.weight_ternary.dtype,
    )

    print(
        "weight_scale :",
        model.weight_scale.item(),
    )

    assert model.weight_ternary.shape == (
        out_features,
        in_features,
    )

    assert model.weight_ternary.dtype == torch.int8

    print("✓ Poids ternaires stockés en INT8")

    # --------------------------------------------------
    # 11. Vérification des valeurs ternaires
    # --------------------------------------------------

    inference_values = torch.unique(
        model.weight_ternary
    )

    print(
        "Valeurs dans weight_ternary :",
        inference_values.detach().cpu().tolist(),
    )

    for value in inference_values:
        assert value.item() in (
            -1,
            0,
            1,
        )

    print("✓ Buffer d'inférence contient uniquement {-1,0,+1}")

    # --------------------------------------------------
    # 12. Test forward inférence
    # --------------------------------------------------

    print("\n[10] Test forward inférence")

    with torch.no_grad():
        output_inference = model(x)

    print(
        f"Output inference : {output_inference.shape}"
    )

    assert output_inference.shape == expected_shape

    assert torch.isfinite(
        output_inference
    ).all()

    print("✓ Forward inférence fonctionnel")

    # --------------------------------------------------
    # 13. Comparaison entraînement / inférence
    # --------------------------------------------------

    print("\n[11] Comparaison")

    difference_output = (
        output.detach()
        - output_inference.detach()
    ).abs()

    print(
        "Différence maximale :",
        difference_output.max().item(),
    )

    print(
        "Différence moyenne :",
        difference_output.mean().item(),
    )

    # --------------------------------------------------
    # FIN
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("✓ TOUS LES TESTS BITLINEAR SONT PASSÉS")
    print("=" * 60)


if __name__ == "__main__":
    test_bitlinear()
