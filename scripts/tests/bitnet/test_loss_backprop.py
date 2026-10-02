import torch
import torch.nn as nn

from llm.bitnet_model.bit_transformer import BitTransformer
from llm.bitnet_model.lm_head import LMHead


def main():
    # ==========================================
    # Configuration du modèle
    # ==========================================

    vocab_size = 994
    d_model = 256
    num_heads = 4
    hidden_dim = 680
    num_blocks = 2
    max_sequence_length = 128

    # ==========================================
    # Configuration du test
    # ==========================================

    batch_size = 2
    sequence_length = 8

    # ==========================================
    # Création du Transformer
    # ==========================================

    transformer = BitTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        max_sequence_length=max_sequence_length,
        bias=False,
    )

    # ==========================================
    # Création du LM Head
    # ==========================================

    lm_head = LMHead(
        d_model=d_model,
        vocab_size=vocab_size,
        bias=False,
    )

    # ==========================================
    # Création des données
    # ==========================================

    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, sequence_length),
    )

    # Les labels représentent les tokens que
    # le modèle doit apprendre à prédire.
    labels = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, sequence_length),
    )

    print("Input IDs :")
    print(input_ids)

    print("\nLabels :")
    print(labels)

    # ==========================================
    # Forward
    # ==========================================

    hidden_states = transformer(input_ids)

    logits = lm_head(hidden_states)

    print("\nShape hidden_states :", hidden_states.shape)
    print("Shape logits :", logits.shape)

    # ==========================================
    # Calcul de la Loss
    # ==========================================

    loss_function = nn.CrossEntropyLoss()

    # CrossEntropyLoss attend :
    #
    # logits : [batch, vocab_size, sequence]
    # labels : [batch, sequence]
    #
    # On réorganise donc les logits.

    logits_for_loss = logits.transpose(1, 2)

    loss = loss_function(
        logits_for_loss,
        labels,
    )

    print("\nLoss :", loss.item())

    # ==========================================
    # Vérification de la Loss
    # ==========================================

    assert loss.ndim == 0
    assert torch.isfinite(loss)

    print("✓ Loss calculée correctement.")

    # ==========================================
    # Backpropagation
    # ==========================================

    loss.backward()

    print("\nBackpropagation effectuée.")

    # ==========================================
    # Vérification des gradients
    # ==========================================

    gradient_count = 0

    for name, parameter in transformer.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None

            assert torch.isfinite(parameter.grad).all()

            gradient_count += 1

    for name, parameter in lm_head.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None

            assert torch.isfinite(parameter.grad).all()

            gradient_count += 1

    print("Nombre de paramètres avec gradient :", gradient_count)

    assert gradient_count > 0

    # ==========================================
    # Affichage de quelques gradients
    # ==========================================

    print("\nQuelques gradients :")

    displayed = 0

    for name, parameter in transformer.named_parameters():
        if parameter.grad is not None:
            print(
                f"{name} -> "
                f"shape={parameter.grad.shape}, "
                f"mean={parameter.grad.mean().item():.6f}"
            )

            displayed += 1

            if displayed >= 5:
                break

    # ==========================================
    # Test terminé
    # ==========================================

    print("\n✓ Loss + Backpropagation fonctionnent correctement.")


if __name__ == "__main__":
    main()