from pathlib import Path

import torch
import torch.nn.functional as F

from tokenizers import Tokenizer

from llm.bitnet_model.bit_transformer import BitTransformer
from llm.bitnet_model.lm_head import LMHead


# ============================================================
# CONFIGURATION
# ============================================================

TOKENIZER_FILE = Path(
    "tokenizer/tokenizer.json"
)

CHECKPOINT_FILE = Path(
    "checkpoint/bitnet/bitnet_5docs.pt"
)


# ============================================================
# CHARGER LE TOKENIZER
# ============================================================

def load_tokenizer():

    if not TOKENIZER_FILE.exists():

        raise FileNotFoundError(
            f"Tokenizer introuvable : "
            f"{TOKENIZER_FILE}"
        )

    tokenizer = Tokenizer.from_file(
        str(TOKENIZER_FILE)
    )

    return tokenizer


# ============================================================
# CHARGER LE CHECKPOINT
# ============================================================

def load_checkpoint(
    checkpoint_path=CHECKPOINT_FILE,
):

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint introuvable : "
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    return checkpoint


# ============================================================
# CRÉER LE MODÈLE BITNET
# ============================================================

def create_bitnet_model(
    checkpoint,
    device,
):

    config = checkpoint["config"]

    # --------------------------------------------------------
    # Récupération de la configuration utilisée pendant
    # l'entraînement.
    # --------------------------------------------------------

    vocab_size = config[
        "vocab_size"
    ]

    d_model = config[
        "d_model"
    ]

    num_heads = config[
        "num_heads"
    ]

    hidden_dim = config[
        "hidden_dim"
    ]

    num_blocks = config[
        "num_blocks"
    ]

    max_sequence_length = config[
        "max_sequence_length"
    ]

    # ========================================================
    # TRANSFORMER
    # ========================================================

    transformer = BitTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        max_sequence_length=max_sequence_length,
        bias=False,
    )

    # ========================================================
    # LM HEAD
    # ========================================================

    lm_head = LMHead(
        d_model=d_model,
        vocab_size=vocab_size,
        bias=False,
    )

    # ========================================================
    # CHARGEMENT DES POIDS
    # ========================================================

    transformer.load_state_dict(
        checkpoint[
            "transformer_state_dict"
        ]
    )

    lm_head.load_state_dict(
        checkpoint[
            "lm_head_state_dict"
        ]
    )

    # ========================================================
    # DEVICE
    # ========================================================

    transformer = transformer.to(
        device
    )

    lm_head = lm_head.to(
        device
    )

    transformer.eval()

    lm_head.eval()

    return (
        transformer,
        lm_head,
    )


# ============================================================
# ENCODER LE PROMPT
# ============================================================

def encode_prompt(
    tokenizer,
    prompt,
    device,
):

    encoding = tokenizer.encode(
        prompt
    )

    token_ids = encoding.ids

    input_ids = torch.tensor(
        token_ids,
        dtype=torch.long,
        device=device,
    )

    # --------------------------------------------------------
    # Ajouter la dimension batch
    #
    # [sequence]
    # devient
    # [1, sequence]
    # --------------------------------------------------------

    input_ids = input_ids.unsqueeze(0)

    return input_ids


# ============================================================
# DÉCODER LES TOKENS
# ============================================================

def decode_tokens(
    tokenizer,
    token_ids,
):

    return tokenizer.decode(
        token_ids
    )


# ============================================================
# GÉNÉRER
# ============================================================

@torch.no_grad()
def generate(
    transformer,
    lm_head,
    tokenizer,
    prompt,
    device,
    max_new_tokens=50,
    temperature=1.0,
):

    # ========================================================
    # TOKENIZATION DU PROMPT
    # ========================================================

    input_ids = encode_prompt(
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
    )

    # ========================================================
    # CONFIGURATION DU CONTEXTE
    # ========================================================

    max_sequence_length = (
        transformer.max_sequence_length
    )

    # ========================================================
    # GÉNÉRATION TOKEN PAR TOKEN
    # ========================================================

    for _ in range(
        max_new_tokens
    ):

        # ----------------------------------------------------
        # Si le contexte dépasse la longueur maximale,
        # on garde les derniers tokens.
        # ----------------------------------------------------

        model_input = input_ids

        if (
            model_input.shape[1]
            > max_sequence_length
        ):

            model_input = model_input[
                :,
                -max_sequence_length:
            ]

        # ----------------------------------------------------
        # Transformer
        # ----------------------------------------------------

        hidden_states = transformer(
            model_input
        )

        # ----------------------------------------------------
        # LM Head
        # ----------------------------------------------------

        logits = lm_head(
            hidden_states
        )

        # ----------------------------------------------------
        # Logits du dernier token
        # ----------------------------------------------------

        next_token_logits = logits[
            :, -1, :
        ]

        # ====================================================
        # TEMPERATURE
        # ====================================================

        if temperature <= 0:

            raise ValueError(
                "temperature doit être supérieur à 0."
            )

        next_token_logits = (
            next_token_logits
            / temperature
        )

        # ====================================================
        # PROBABILITÉS
        # ====================================================

        probabilities = F.softmax(
            next_token_logits,
            dim=-1,
        )

        # ====================================================
        # CHOIX DU PROCHAIN TOKEN
        # ====================================================

        next_token = torch.multinomial(
            probabilities,
            num_samples=1,
        )

        # ----------------------------------------------------
        # Ajouter le token au contexte
        # ----------------------------------------------------

        input_ids = torch.cat(
            [
                input_ids,
                next_token,
            ],
            dim=1,
        )

    # ========================================================
    # DÉCODAGE FINAL
    # ========================================================

    generated_token_ids = (
        input_ids[0]
        .tolist()
    )

    generated_text = decode_tokens(
        tokenizer=tokenizer,
        token_ids=generated_token_ids,
    )

    return generated_text