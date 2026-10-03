from pathlib import Path

import torch

from llm.model.transformer import Transformer
from llm.tokenizer.tokenizer import load_tokenizer


# ============================================================
# CONFIGURATION
# ============================================================

VOCAB_SIZE = 994

EMBEDDING_DIM = 120
NUM_HEADS = 3
FFN_HIDDEN_DIM = 480
NUM_LAYERS = 2
MAX_SEQUENCE_LENGTH = 128

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_PATH = Path(
    "checkpoints/student_v1/student_v1_25docs.pt"
)

MAX_NEW_TOKENS = 50


# ============================================================
# PROMPTS
# ============================================================

TRAIN_PROMPTS = [
    "Planche à neige",
    "La Fed a relevé les taux",
    "Choux de Bruxelles",
    "Bridjet vêtements grande taille",
    "nouvelle augmentation de capital de 40M€",
    "Randonnée depuis la Baraque Michel",
    "1961",
]

RANDOM_PROMPTS = [
    "Le soleil se couche",
]


# ============================================================
# CHARGEMENT DU MODÈLE
# ============================================================

def load_model():

    print("=" * 60)
    print("CHARGEMENT DU STUDENT V1")
    print("=" * 60)

    print(f"Checkpoint : {CHECKPOINT_PATH}")
    print(f"Device     : {DEVICE}")

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"Checkpoint introuvable : {CHECKPOINT_PATH}"
        )

    # --------------------------------------------------------
    # Création du modèle
    # --------------------------------------------------------

    model = Transformer(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        ffn_hidden_dim=FFN_HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
    )

    # --------------------------------------------------------
    # Chargement du checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)

    model.eval()

    print("Modèle chargé.")

    if "global_step" in checkpoint:

        print(
            f"Global step : "
            f"{checkpoint['global_step']}"
        )

    print()

    return model


# ============================================================
# GÉNÉRATION
# ============================================================

@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=50,
):

    # --------------------------------------------------------
    # Tokenisation du prompt
    # --------------------------------------------------------

    encoding = tokenizer.encode(prompt)

    input_ids = encoding.ids

    if len(input_ids) == 0:

        raise ValueError(
            "Le prompt ne contient aucun token."
        )

    input_ids = torch.tensor(
        [input_ids],
        dtype=torch.long,
        device=DEVICE,
    )

    # --------------------------------------------------------
    # Génération autoregressive
    # --------------------------------------------------------

    for _ in range(max_new_tokens):

        # Garder uniquement les 128 derniers tokens
        context = input_ids[
            :, -MAX_SEQUENCE_LENGTH:
        ]

        # ----------------------------------------------------
        # Forward du Transformer
        # ----------------------------------------------------

        logits = model(context)

        # ----------------------------------------------------
        # Logits du dernier token
        # ----------------------------------------------------

        next_token_logits = logits[:, -1, :]

        # ----------------------------------------------------
        # Greedy decoding
        # ----------------------------------------------------

        next_token = torch.argmax(
            next_token_logits,
            dim=-1,
            keepdim=True,
        )

        # ----------------------------------------------------
        # Ajouter le nouveau token
        # ----------------------------------------------------

        input_ids = torch.cat(
            [
                input_ids,
                next_token,
            ],
            dim=1,
        )

    # --------------------------------------------------------
    # Décodage final
    # --------------------------------------------------------

    generated_ids = input_ids[0].tolist()

    generated_text = tokenizer.decode(
        generated_ids
    )

    return generated_text


# ============================================================
# TEST DES PROMPTS
# ============================================================

def test_prompts(
    model,
    tokenizer,
    prompts,
    category,
):

    print()
    print("=" * 60)
    print(category)
    print("=" * 60)

    for index, prompt in enumerate(
        prompts,
        start=1,
    ):

        print()
        print("-" * 60)
        print(f"TEST {index}")
        print("-" * 60)

        print()
        print("Prompt :")
        print(prompt)

        print()
        print("Génération :")

        generated_text = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=MAX_NEW_TOKENS,
        )

        print(generated_text)

        print()


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("TEST DE GÉNÉRATION - STUDENT V1")
    print("=" * 60)
    print()

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    print("Chargement du tokenizer...")

    tokenizer = load_tokenizer()

    print(
        f"Vocabulaire : "
        f"{tokenizer.get_vocab_size()}"
    )

    print()

    # --------------------------------------------------------
    # Modèle
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Prompts provenant des données
    # --------------------------------------------------------

    test_prompts(
        model=model,
        tokenizer=tokenizer,
        prompts=TRAIN_PROMPTS,
        category="PROMPTS ISSUS DES DOCUMENTS D'ENTRAÎNEMENT",
    )

    # --------------------------------------------------------
    # Prompts jamais vus
    # --------------------------------------------------------

    test_prompts(
        model=model,
        tokenizer=tokenizer,
        prompts=RANDOM_PROMPTS,
        category="PROMPTS NOUVEAUX / NON PRÉSENTS DANS LES DONNÉES",
    )

    # --------------------------------------------------------
    # Fin
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("TEST TERMINÉ")
    print("=" * 60)


# ============================================================
# POINT D'ENTRÉE
# ============================================================

if __name__ == "__main__":
    main()