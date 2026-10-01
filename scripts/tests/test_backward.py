import torch

from llm.data.dataloader import create_train_dataloader
from llm.tokenizer.tokenizer import get_vocab_size
from llm.model.transformer import CausalTransformer
from llm.training.loss import CausalLanguageModelingLoss


EMBEDDING_DIM = 120
NUM_HEADS = 3
FFN_HIDDEN_DIM = 480
NUM_LAYERS = 4
MAX_SEQUENCE_LENGTH = 128


def main():
    print("=" * 60)
    print("TEST DE LA BACKPROPAGATION")
    print("=" * 60)

    vocab_size = get_vocab_size()

    train_loader = create_train_dataloader()

    batch = next(iter(train_loader))

    input_ids = batch["input_ids"]
    labels = batch["labels"]

    model = CausalTransformer(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        ffn_hidden_dim=FFN_HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
    )

    loss_function = CausalLanguageModelingLoss()

    # Forward
    logits = model(input_ids)

    loss = loss_function(
        logits,
        labels,
    )

    print("\nLoss avant backward :")
    print(loss.item())

    # Backpropagation
    model.zero_grad()

    loss.backward()

    print("\nBackpropagation effectuée.")

    print("\n--- VÉRIFICATION DES GRADIENTS ---")

    parameters_with_gradients = 0
    parameters_without_gradients = 0

    total_gradient_values = 0

    for name, parameter in model.named_parameters():

        if parameter.requires_grad:

            if parameter.grad is not None:
                parameters_with_gradients += 1

                total_gradient_values += parameter.grad.numel()

            else:
                parameters_without_gradients += 1

    print(
        f"Paramètres avec gradients    : "
        f"{parameters_with_gradients}"
    )

    print(
        f"Paramètres sans gradients    : "
        f"{parameters_without_gradients}"
    )

    print(
        f"Valeurs de gradients calculées : "
        f"{total_gradient_values:,}"
    )

    print("\n--- EXEMPLE DE GRADIENT ---")

    parameter_name = "token_embedding.embedding.weight"

    parameter = dict(
        model.named_parameters()
    )[parameter_name]

    print(f"Paramètre : {parameter_name}")
    print(f"Shape     : {parameter.shape}")
    print(f"Gradient  : {parameter.grad.shape}")

    print("\nQuelques valeurs du gradient :")

    print(
        parameter.grad.flatten()[:10]
    )

    print("\n--- VÉRIFICATION ---")

    assert loss.requires_grad

    assert loss.grad_fn is not None

    assert parameters_with_gradients > 0

    assert parameters_without_gradients == 0

    assert total_gradient_values > 0

    assert parameter.grad is not None

    assert torch.isfinite(
        parameter.grad
    ).all()

    print("✓ Loss possède un graphe de calcul")
    print("✓ Backpropagation exécutée")
    print("✓ Les paramètres reçoivent des gradients")
    print("✓ Aucun paramètre entraînable sans gradient")
    print("✓ Les gradients sont finis")
    print("✓ Backpropagation fonctionnelle")


if __name__ == "__main__":
    main()