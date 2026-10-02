import os

import torch
import torch.nn as nn

from llm.bitnet_model.bit_transformer import BitTransformer
from llm.bitnet_model.lm_head import LMHead


def create_model():
    """
    Crée le Transformer BitNet et le LM Head
    avec exactement la même architecture.
    """

    vocab_size = 994
    d_model = 256
    num_heads = 4
    hidden_dim = 680
    num_blocks = 2
    max_sequence_length = 128

    transformer = BitTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        max_sequence_length=max_sequence_length,
        bias=False,
    )

    lm_head = LMHead(
        d_model=d_model,
        vocab_size=vocab_size,
        bias=False,
    )

    return transformer, lm_head


def main():

    # ==========================================
    # Configuration
    # ==========================================

    vocab_size = 994
    batch_size = 2
    sequence_length = 8

    learning_rate = 1e-3

    checkpoint_path = "checkpoint/bitnet/checkpoint_bitnet_test.pt"

    # Création automatique du dossier
    os.makedirs(
        os.path.dirname(checkpoint_path),
        exist_ok=True,
    )

    # ==========================================
    # Création du premier modèle
    # ==========================================

    transformer, lm_head = create_model()

    loss_function = nn.CrossEntropyLoss()

    parameters = list(transformer.parameters()) + list(lm_head.parameters())

    optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
    )

    # ==========================================
    # Petit entraînement
    # ==========================================

    print("Entraînement avant sauvegarde...\n")

    for step in range(3):

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

        optimizer.zero_grad()

        hidden_states = transformer(input_ids)

        logits = lm_head(hidden_states)

        logits_for_loss = logits.transpose(1, 2)

        loss = loss_function(
            logits_for_loss,
            labels,
        )

        loss.backward()

        optimizer.step()

        print(
            f"Step {step + 1}/3 - "
            f"Loss : {loss.item():.6f}"
        )

    # ==========================================
    # Sauvegarde des paramètres
    # ==========================================

    transformer_before = {
        key: value.detach().clone()
        for key, value in transformer.state_dict().items()
    }

    lm_head_before = {
        key: value.detach().clone()
        for key, value in lm_head.state_dict().items()
    }

    # ==========================================
    # Création du checkpoint
    # ==========================================

    checkpoint = {
        "model_state_dict": transformer.state_dict(),
        "lm_head_state_dict": lm_head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),

        "step": 3,

        "config": {
            "vocab_size": vocab_size,
            "d_model": 256,
            "num_heads": 4,
            "hidden_dim": 680,
            "num_blocks": 2,
            "max_sequence_length": 128,
        },
    }

    torch.save(
        checkpoint,
        checkpoint_path,
    )

    print(
        f"\nCheckpoint sauvegardé : {checkpoint_path}"
    )

    assert os.path.exists(checkpoint_path)

    # ==========================================
    # Création de nouveaux modèles
    # ==========================================

    new_transformer, new_lm_head = create_model()

    new_parameters = (
        list(new_transformer.parameters())
        + list(new_lm_head.parameters())
    )

    new_optimizer = torch.optim.AdamW(
        new_parameters,
        lr=learning_rate,
    )

    # ==========================================
    # Chargement du checkpoint
    # ==========================================

    loaded_checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    new_transformer.load_state_dict(
        loaded_checkpoint["model_state_dict"]
    )

    new_lm_head.load_state_dict(
        loaded_checkpoint["lm_head_state_dict"]
    )

    new_optimizer.load_state_dict(
        loaded_checkpoint["optimizer_state_dict"]
    )

    print("Checkpoint chargé.")

    # ==========================================
    # Vérification du modèle
    # ==========================================

    for key, value in transformer_before.items():

        restored_value = new_transformer.state_dict()[key]

        assert torch.equal(
            value,
            restored_value,
        ), f"Paramètre différent : {key}"

    # ==========================================
    # Vérification du LM Head
    # ==========================================

    for key, value in lm_head_before.items():

        restored_value = new_lm_head.state_dict()[key]

        assert torch.equal(
            value,
            restored_value,
        ), f"LM Head différent : {key}"

    # ==========================================
    # Vérification du step
    # ==========================================

    assert loaded_checkpoint["step"] == 3

    # ==========================================
    # Vérification de la configuration
    # ==========================================

    config = loaded_checkpoint["config"]

    assert config["vocab_size"] == 994
    assert config["d_model"] == 256
    assert config["num_heads"] == 4
    assert config["hidden_dim"] == 680
    assert config["num_blocks"] == 2
    assert config["max_sequence_length"] == 128

    # ==========================================
    # Test avec le modèle restauré
    # ==========================================

    test_input = torch.randint(
        low=0,
        high=vocab_size,
        size=(1, sequence_length),
    )

    restored_hidden = new_transformer(test_input)

    restored_logits = new_lm_head(restored_hidden)

    print(
        "\nShape sortie modèle restauré :",
        restored_hidden.shape,
    )

    print(
        "Shape logits restaurés :",
        restored_logits.shape,
    )

    assert restored_hidden.shape == (
        1,
        sequence_length,
        256,
    )

    assert restored_logits.shape == (
        1,
        sequence_length,
        vocab_size,
    )

    # ==========================================
    # Nettoyage
    # ==========================================

    os.remove(checkpoint_path)

    assert not os.path.exists(checkpoint_path)

    print("\nCheckpoint supprimé après le test.")

    # ==========================================
    # Résultat
    # ==========================================

    print(
        "\n✓ Sauvegarde et restauration du checkpoint "
        "fonctionnent correctement."
    )


if __name__ == "__main__":
    main()