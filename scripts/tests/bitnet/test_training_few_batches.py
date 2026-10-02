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
    # Configuration de l'entraînement
    # ==========================================

    batch_size = 2
    sequence_length = 8
    num_batches = 5

    learning_rate = 1e-3

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
    # Fonction de perte
    # ==========================================

    loss_function = nn.CrossEntropyLoss()

    # ==========================================
    # Optimizer
    # ==========================================

    parameters = list(transformer.parameters()) + list(lm_head.parameters())

    optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
    )

    print("Début du test d'entraînement...\n")

    losses = []

    # ==========================================
    # Boucle d'entraînement
    # ==========================================

    for batch_index in range(num_batches):

        # --------------------------------------
        # Génération d'un batch
        # --------------------------------------

        input_ids = torch.randint(
            low=0,
            high=vocab_size,
            size=(batch_size, sequence_length),
        )

        labels = torch.randint(
            low=0,
            high=vocab_size,
            size=(batch_size, sequence_length),
        )

        # --------------------------------------
        # Remise à zéro des gradients
        # --------------------------------------

        optimizer.zero_grad()

        # --------------------------------------
        # Forward
        # --------------------------------------

        hidden_states = transformer(input_ids)

        logits = lm_head(hidden_states)

        # --------------------------------------
        # Préparation pour CrossEntropyLoss
        # --------------------------------------

        logits_for_loss = logits.transpose(1, 2)

        # --------------------------------------
        # Calcul de la Loss
        # --------------------------------------

        loss = loss_function(
            logits_for_loss,
            labels,
        )

        # --------------------------------------
        # Backpropagation
        # --------------------------------------

        loss.backward()

        # --------------------------------------
        # Mise à jour des paramètres
        # --------------------------------------

        optimizer.step()

        # --------------------------------------
        # Sauvegarde de la Loss
        # --------------------------------------

        loss_value = loss.item()

        losses.append(loss_value)

        print(
            f"Batch {batch_index + 1}/{num_batches} "
            f"- Loss : {loss_value:.6f}"
        )

    # ==========================================
    # Vérification
    # ==========================================

    assert len(losses) == num_batches

    for loss_value in losses:
        assert torch.isfinite(
            torch.tensor(loss_value)
        )

    # ==========================================
    # Vérification des paramètres
    # ==========================================

    parameters_with_grad = 0

    for parameter in parameters:

        if parameter.requires_grad:

            assert parameter.grad is not None

            assert torch.isfinite(parameter.grad).all()

            parameters_with_grad += 1

    print(
        "\nNombre de paramètres entraînables :",
        parameters_with_grad,
    )

    assert parameters_with_grad > 0

    # ==========================================
    # Résultat
    # ==========================================

    print("\nLoss avant entraînement :", losses[0])
    print("Loss après entraînement :", losses[-1])

    print(
        "\n✓ Entraînement sur quelques batchs "
        "fonctionne correctement."
    )


if __name__ == "__main__":
    main()