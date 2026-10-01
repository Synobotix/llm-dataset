import torch
from tokenizers import Tokenizer

from llm.model.transformer import Transformer


# ============================================================
# CONFIGURATION
# ============================================================

VOCAB_SIZE = 994
D_MODEL = 120
NUM_HEADS = 3
NUM_LAYERS = 2
FFN_HIDDEN_DIM = 480
MAX_SEQ_LEN = 128

DEVICE = "cpu"

CHECKPOINT_PATH = "checkpoints/student_v1/student_v1_final.pt"
TOKENIZER_PATH = "tokenizer/tokenizer.json"


# ============================================================
# CHARGEMENT DU TOKENIZER
# ============================================================

print("=" * 60)
print("CHARGEMENT DU TOKENIZER")
print("=" * 60)

tokenizer = Tokenizer.from_file(
    TOKENIZER_PATH
)

print("Tokenizer chargé.")
print(
    f"Vocabulaire tokenizer : "
    f"{tokenizer.get_vocab_size()}"
)


# ============================================================
# CRÉATION DU MODÈLE
# ============================================================

print()
print("=" * 60)
print("CRÉATION DU MODÈLE")
print("=" * 60)

model = Transformer(
    vocab_size=VOCAB_SIZE,
    embedding_dim=D_MODEL,
    num_heads=NUM_HEADS,
    ffn_hidden_dim=FFN_HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    max_sequence_length=MAX_SEQ_LEN,
)

model.to(DEVICE)

print("Modèle créé.")


# ============================================================
# CHARGEMENT DU CHECKPOINT
# ============================================================

print()
print("=" * 60)
print("CHARGEMENT DU CHECKPOINT")
print("=" * 60)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
)

print("Clés du checkpoint :")
print(checkpoint.keys())

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print()
print("Checkpoint chargé avec succès.")

if "global_step" in checkpoint:
    print(
        f"Global step : "
        f"{checkpoint['global_step']}"
    )


# ============================================================
# GÉNÉRATION
# ============================================================

def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=30,
):
    """
    Génération autoregressive avec argmax.
    """

    # --------------------------------------------------------
    # Encodage
    # --------------------------------------------------------

    encoding = tokenizer.encode(prompt)

    input_ids = encoding.ids

    print()
    print("Prompt :")
    print(prompt)

    print()
    print("Tokens :")
    print(encoding.tokens)

    print()
    print("IDs :")
    print(input_ids)

    # --------------------------------------------------------
    # Tensor
    # --------------------------------------------------------

    input_tensor = torch.tensor(
        input_ids,
        dtype=torch.long,
        device=DEVICE,
    ).unsqueeze(0)

    # --------------------------------------------------------
    # Génération
    # --------------------------------------------------------

    with torch.no_grad():

        for step in range(max_new_tokens):

            # Le Transformer accepte au maximum
            # MAX_SEQ_LEN tokens.

            context = input_tensor[
                :, -MAX_SEQ_LEN:
            ]

            # Forward
            logits = model(context)

            # Logits du dernier token
            next_token_logits = logits[
                :, -1, :
            ]

            # Choix du token ayant
            # le plus grand logit
            next_token_id = torch.argmax(
                next_token_logits,
                dim=-1,
                keepdim=True,
            )

            # Ajout du token
            input_tensor = torch.cat(
                [
                    input_tensor,
                    next_token_id,
                ],
                dim=1,
            )

    # --------------------------------------------------------
    # Décodage
    # --------------------------------------------------------

    generated_ids = input_tensor[
        0
    ].tolist()

    generated_text = tokenizer.decode(
        generated_ids
    )

    return generated_ids, generated_text


# ============================================================
# TEST
# ============================================================

generated_ids, generated_text = generate(
    model=model,
    tokenizer=tokenizer,
    prompt="Boursorama Epargne",
    max_new_tokens=30,
)


# ============================================================
# RÉSULTAT
# ============================================================

print()
print("=" * 60)
print("RÉSULTAT")
print("=" * 60)

print()
print("IDs générés :")
print(generated_ids)

print()
print("Texte généré :")
print(generated_text)