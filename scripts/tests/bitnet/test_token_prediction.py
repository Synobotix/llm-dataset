import torch

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
    # Création du Transformer BitNet
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
    # Création des tokens d'entrée
    # ==========================================

    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, sequence_length),
    )

    print("Input IDs :")
    print(input_ids)

    print("\nShape input_ids :", input_ids.shape)

    # ==========================================
    # Passage dans le Transformer
    # ==========================================

    hidden_states = transformer(input_ids)

    print("\nHidden states :")
    print(hidden_states)

    print("\nShape hidden_states :", hidden_states.shape)

    # ==========================================
    # Passage dans le LM Head
    # ==========================================

    logits = lm_head(hidden_states)

    print("\nShape logits :", logits.shape)

    # ==========================================
    # Vérification des dimensions
    # ==========================================

    expected_hidden_shape = (
        batch_size,
        sequence_length,
        d_model,
    )

    expected_logits_shape = (
        batch_size,
        sequence_length,
        vocab_size,
    )

    assert hidden_states.shape == expected_hidden_shape

    assert logits.shape == expected_logits_shape

    # ==========================================
    # Prédiction du token
    # ==========================================

    predicted_tokens = torch.argmax(
        logits,
        dim=-1,
    )

    print("\nTokens prédits :")
    print(predicted_tokens)

    print("\nShape predicted_tokens :", predicted_tokens.shape)

    # ==========================================
    # Vérification des prédictions
    # ==========================================

    expected_prediction_shape = (
        batch_size,
        sequence_length,
    )

    assert predicted_tokens.shape == expected_prediction_shape

    assert torch.all(predicted_tokens >= 0)

    assert torch.all(predicted_tokens < vocab_size)

    # ==========================================
    # Prédiction du prochain token
    # ==========================================

    last_logits = logits[:, -1, :]

    next_tokens = torch.argmax(
        last_logits,
        dim=-1,
    )

    print("\nDernier token prédit pour chaque séquence :")
    print(next_tokens)

    print("\nShape next_tokens :", next_tokens.shape)

    assert next_tokens.shape == (batch_size,)

    assert torch.all(next_tokens >= 0)

    assert torch.all(next_tokens < vocab_size)

    # ==========================================
    # Test terminé
    # ==========================================

    print("\n✓ Transformer + LM Head + prédiction des tokens fonctionnent correctement.")


if __name__ == "__main__":
    main()