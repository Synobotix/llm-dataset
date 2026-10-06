import sys
from pathlib import Path


# Ajouter la racine du projet au PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import math

import torch
import torch.nn as nn
from torch.optim import AdamW
from tokenizers import Tokenizer

from llm.config.parameters import (
    MAX_DOCUMENT,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    GRADIENT_CLIP,
    MAX_SEQUENCE_LENGTH,
    TOKENIZER_FILE,
    D_MODEL,
    NUM_HEADS,
    HIDDEN_DIM,
    NUM_BLOCKS,
)

from llm.data.dataloader import (
    create_validation_dataloader,
)

from llm.bitnet_model.bit_transformer import (
    BitTransformer,
)

from llm.bitnet_model.lm_head import (
    LMHead,
)

from scripts.duration import TrainingTimer

from scripts.bitnet.checkpoint_path_bitnet import (
    get_checkpoint_paths,
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# DATASET
# ============================================================

TRAIN_DATASET_FILE = Path(
    "data/tokenized/train.jsonl"
)


# ============================================================
# TOKENIZER
# ============================================================

def load_tokenizer():
    """
    Charge le tokenizer utilisé pour le dataset.
    """

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
# DATASET TRAIN
# ============================================================

class TokenizedDataset(
    torch.utils.data.Dataset
):
    """
    Dataset PyTorch pour le fichier JSONL tokenisé.
    """

    def __init__(
        self,
        file_path: Path,
    ):
        import json

        self.samples = []

        if not file_path.exists():

            raise FileNotFoundError(
                f"Dataset train introuvable : "
                f"{file_path}"
            )

        with file_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            for line_number, line in enumerate(
                file,
                start=1,
            ):

                line = line.strip()

                if not line:
                    continue

                try:

                    sample = json.loads(
                        line
                    )

                except json.JSONDecodeError as error:

                    raise ValueError(
                        f"Ligne {line_number} invalide "
                        f"dans {file_path}: {error}"
                    )

                input_ids = sample.get(
                    "input_ids"
                )

                labels = sample.get(
                    "labels"
                )

                if not isinstance(
                    input_ids,
                    list,
                ):

                    raise ValueError(
                        f"input_ids invalide "
                        f"à la ligne {line_number}"
                    )

                if not isinstance(
                    labels,
                    list,
                ):

                    raise ValueError(
                        f"labels invalides "
                        f"à la ligne {line_number}"
                    )

                if len(input_ids) != MAX_SEQUENCE_LENGTH:

                    raise ValueError(
                        f"input_ids doit contenir "
                        f"{MAX_SEQUENCE_LENGTH} tokens "
                        f"à la ligne {line_number}"
                    )

                if len(labels) != MAX_SEQUENCE_LENGTH:

                    raise ValueError(
                        f"labels doit contenir "
                        f"{MAX_SEQUENCE_LENGTH} tokens "
                        f"à la ligne {line_number}"
                    )

                self.samples.append(
                    {
                        "input_ids": input_ids,
                        "labels": labels,
                    }
                )

    def __len__(self):

        return len(
            self.samples
        )

    def __getitem__(
        self,
        index,
    ):

        sample = self.samples[
            index
        ]

        return {
            "input_ids": torch.tensor(
                sample["input_ids"],
                dtype=torch.long,
            ),

            "labels": torch.tensor(
                sample["labels"],
                dtype=torch.long,
            ),
        }


# ============================================================
# TRAIN DATALOADER
# ============================================================

def create_train_dataloader():
    """
    Crée le DataLoader du dataset d'entraînement.
    """

    dataset = TokenizedDataset(
        TRAIN_DATASET_FILE
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    return dataloader


# ============================================================
# PERPLEXITY
# ============================================================

def calculate_perplexity(loss):
    """
    Calcule la perplexité à partir de la loss.
    """

    try:

        return math.exp(loss)

    except OverflowError:

        return float("inf")


# ============================================================
# GRADIENT NORM
# ============================================================

def calculate_gradient_norm(model):
    """
    Calcule la norme globale des gradients.
    """

    total_norm = 0.0

    for parameter in model.parameters():

        if parameter.grad is None:
            continue

        parameter_norm = (
            parameter.grad.detach()
            .data.norm(2)
        )

        total_norm += (
            parameter_norm.item() ** 2
        )

    return total_norm ** 0.5


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint(
    checkpoint_path,
    transformer,
    lm_head,
    optimizer,
    vocab_size,
):
    """
    Charge le checkpoint précédent et restaure :

    - les poids du Transformer
    - les poids du LM Head
    - l'état de l'optimizer
    - l'époque précédente
    - les métriques précédentes
    """

    checkpoint_path = Path(
        checkpoint_path
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint précédent introuvable : "
            f"{checkpoint_path}"
        )

    print(
        "\nChargement du checkpoint précédent..."
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
    )

    if "config" not in checkpoint:

        raise ValueError(
            "Le checkpoint ne contient pas "
            "de configuration."
        )

    checkpoint_config = (
        checkpoint["config"]
    )

    checkpoint_vocab_size = (
        checkpoint_config.get(
            "vocab_size"
        )
    )

    checkpoint_d_model = (
        checkpoint_config.get(
            "d_model"
        )
    )

    checkpoint_num_heads = (
        checkpoint_config.get(
            "num_heads"
        )
    )

    checkpoint_hidden_dim = (
        checkpoint_config.get(
            "hidden_dim"
        )
    )

    checkpoint_num_blocks = (
        checkpoint_config.get(
            "num_blocks"
        )
    )

    checkpoint_max_sequence_length = (
        checkpoint_config.get(
            "max_sequence_length"
        )
    )

    if checkpoint_vocab_size != vocab_size:

        raise ValueError(
            "\nLe vocabulaire du checkpoint "
            "ne correspond pas au tokenizer actuel.\n"
            f"Checkpoint : {checkpoint_vocab_size}\n"
            f"Tokenizer  : {vocab_size}"
        )

    if checkpoint_d_model != D_MODEL:

        raise ValueError(
            "\nD_MODEL différent du checkpoint.\n"
            f"Checkpoint : {checkpoint_d_model}\n"
            f"Actuel     : {D_MODEL}"
        )

    if checkpoint_num_heads != NUM_HEADS:

        raise ValueError(
            "\nNUM_HEADS différent du checkpoint.\n"
            f"Checkpoint : {checkpoint_num_heads}\n"
            f"Actuel     : {NUM_HEADS}"
        )

    if checkpoint_hidden_dim != HIDDEN_DIM:

        raise ValueError(
            "\nHIDDEN_DIM différent du checkpoint.\n"
            f"Checkpoint : {checkpoint_hidden_dim}\n"
            f"Actuel     : {HIDDEN_DIM}"
        )

    if checkpoint_num_blocks != NUM_BLOCKS:

        raise ValueError(
            "\nNUM_BLOCKS différent du checkpoint.\n"
            f"Checkpoint : {checkpoint_num_blocks}\n"
            f"Actuel     : {NUM_BLOCKS}"
        )

    if (
        checkpoint_max_sequence_length
        != MAX_SEQUENCE_LENGTH
    ):

        raise ValueError(
            "\nMAX_SEQUENCE_LENGTH différent "
            "du checkpoint.\n"
            f"Checkpoint : "
            f"{checkpoint_max_sequence_length}\n"
            f"Actuel     : "
            f"{MAX_SEQUENCE_LENGTH}"
        )

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

    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )

    previous_epoch = checkpoint.get(
        "epoch",
        0,
    )

    previous_train_loss = checkpoint.get(
        "train_loss",
        None,
    )

    previous_validation_loss = checkpoint.get(
        "validation_loss",
        None,
    )

    print(
        f"Checkpoint chargé : "
        f"{checkpoint_path}"
    )

    print(
        f"Époque précédente : "
        f"{previous_epoch}"
    )

    if previous_train_loss is not None:

        print(
            f"Ancienne train loss : "
            f"{previous_train_loss:.6f}"
        )

    if previous_validation_loss is not None:

        print(
            f"Ancienne validation loss : "
            f"{previous_validation_loss:.6f}"
        )

    return previous_epoch


# ============================================================
# TRAIN
# ============================================================

def train_one_epoch(
    transformer,
    lm_head,
    dataloader,
    optimizer,
    criterion,
):
    """
    Effectue une époque complète d'entraînement.
    """

    transformer.train()

    lm_head.train()

    total_loss = 0.0

    total_tokens = 0

    last_gradient_norm = 0.0

    for batch_index, batch in enumerate(
        dataloader,
        start=1,
    ):

        input_ids = batch[
            "input_ids"
        ].to(
            DEVICE
        )

        labels = batch[
            "labels"
        ].to(
            DEVICE
        )

        optimizer.zero_grad()

        hidden_states = transformer(
            input_ids
        )

        logits = lm_head(
            hidden_states
        )

        batch_size = logits.size(0)

        sequence_length = (
            logits.size(1)
        )

        vocab_size = logits.size(2)

        logits = logits.reshape(
            batch_size
            * sequence_length,
            vocab_size,
        )

        labels = labels.reshape(
            batch_size
            * sequence_length
        )

        loss = criterion(
            logits,
            labels,
        )

        loss.backward()

        transformer_gradient_norm = (
            calculate_gradient_norm(
                transformer
            )
        )

        lm_head_gradient_norm = (
            calculate_gradient_norm(
                lm_head
            )
        )

        last_gradient_norm = (
            transformer_gradient_norm ** 2
            + lm_head_gradient_norm ** 2
        ) ** 0.5

        torch.nn.utils.clip_grad_norm_(
            list(
                transformer.parameters()
            )
            + list(
                lm_head.parameters()
            ),
            GRADIENT_CLIP,
        )

        optimizer.step()

        number_of_tokens = (
            batch_size
            * sequence_length
        )

        total_loss += (
            loss.item()
            * number_of_tokens
        )

        total_tokens += (
            number_of_tokens
        )

        print(
            f"\rBatch "
            f"{batch_index:>4}/"
            f"{len(dataloader):<4}"
            f" | Loss : "
            f"{loss.item():.6f}",
            end="",
        )

    print()

    average_loss = (
        total_loss / total_tokens
        if total_tokens > 0
        else float("inf")
    )

    return (
        average_loss,
        last_gradient_norm,
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    transformer,
    lm_head,
    dataloader,
    criterion,
):
    """
    Évalue le modèle sur le dataset de validation.
    """

    transformer.eval()

    lm_head.eval()

    total_loss = 0.0

    total_tokens = 0

    for batch in dataloader:

        input_ids = batch[
            "input_ids"
        ].to(
            DEVICE
        )

        labels = batch[
            "labels"
        ].to(
            DEVICE
        )

        hidden_states = transformer(
            input_ids
        )

        logits = lm_head(
            hidden_states
        )

        batch_size = logits.size(0)

        sequence_length = (
            logits.size(1)
        )

        vocab_size = logits.size(2)

        logits = logits.reshape(
            batch_size
            * sequence_length,
            vocab_size,
        )

        labels = labels.reshape(
            batch_size
            * sequence_length
        )

        loss = criterion(
            logits,
            labels,
        )

        number_of_tokens = (
            batch_size
            * sequence_length
        )

        total_loss += (
            loss.item()
            * number_of_tokens
        )

        total_tokens += (
            number_of_tokens
        )

    average_loss = (
        total_loss / total_tokens
        if total_tokens > 0
        else float("inf")
    )

    return average_loss


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    checkpoint_path,
    transformer,
    lm_head,
    optimizer,
    epoch,
    train_loss,
    validation_loss,
    gradient_norm,
    vocab_size,
):
    """
    Sauvegarde le nouveau checkpoint.
    """

    checkpoint_path = Path(
        checkpoint_path
    )

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_ppl = calculate_perplexity(
        train_loss
    )

    validation_ppl = calculate_perplexity(
        validation_loss
    )

    checkpoint = {

        "epoch":
            epoch,

        "transformer_state_dict":
            transformer.state_dict(),

        "lm_head_state_dict":
            lm_head.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "train_loss":
            train_loss,

        "validation_loss":
            validation_loss,

        "train_ppl":
            train_ppl,

        "validation_ppl":
            validation_ppl,

        "train_pll_per_token":
            train_loss,

        "validation_pll_per_token":
            validation_loss,

        "gradient_norm":
            gradient_norm,

        "config": {

            "vocab_size":
                vocab_size,

            "d_model":
                transformer.d_model,

            "num_heads":
                transformer.num_heads,

            "hidden_dim":
                transformer.hidden_dim,

            "num_blocks":
                transformer.num_blocks,

            "max_sequence_length":
                transformer.max_sequence_length,

            "epochs":
                EPOCHS,

            "learning_rate":
                LEARNING_RATE,

            "weight_decay":
                WEIGHT_DECAY,

            "gradient_clip":
                GRADIENT_CLIP,
        },
    }

    torch.save(
        checkpoint,
        checkpoint_path,
    )

    print(
        f"\nNouveau checkpoint sauvegardé : "
        f"{checkpoint_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "ENTRAÎNEMENT BITNET"
    )

    print("=" * 70)

    # ========================================================
    # DEVICE
    # ========================================================

    print(
        f"\nDevice : {DEVICE}"
    )

    if DEVICE.type == "cuda":

        print(
            f"GPU : "
            f"{torch.cuda.get_device_name(0)}"
        )

    # ========================================================
    # CHECKPOINT PATHS
    # ========================================================

    (
        previous_checkpoint,
        new_checkpoint,
    ) = get_checkpoint_paths()

    print(
        "\nCheckpoint à charger : "
        f"{previous_checkpoint}"
    )

    print(
        "Checkpoint à sauvegarder : "
        f"{new_checkpoint}"
    )

    # ========================================================
    # TOKENIZER
    # ========================================================

    print(
        "\nChargement du tokenizer..."
    )

    tokenizer = load_tokenizer()

    vocab_size = (
        tokenizer.get_vocab_size()
    )

    print(
        f"Vocabulaire réel : "
        f"{vocab_size}"
    )

    # ========================================================
    # DATALOADERS
    # ========================================================

    print(
        "\nCréation des DataLoaders..."
    )

    train_loader = (
        create_train_dataloader()
    )

    validation_loader = (
        create_validation_dataloader()
    )

    print(
        f"Train batches      : "
        f"{len(train_loader)}"
    )

    print(
        f"Validation batches : "
        f"{len(validation_loader)}"
    )

    print(
        f"Batch size         : "
        f"{BATCH_SIZE}"
    )

    print(
        f"Sequence length    : "
        f"{MAX_SEQUENCE_LENGTH}"
    )

    # ========================================================
    # TRANSFORMER
    # ========================================================

    print(
        "\nCréation du BitTransformer..."
    )

    transformer = BitTransformer(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
    ).to(
        DEVICE
    )

    # ========================================================
    # LM HEAD
    # ========================================================

    print(
        "Création du LM Head..."
    )

    lm_head = LMHead(
        d_model=D_MODEL,
        vocab_size=vocab_size,
    ).to(
        DEVICE
    )

    # ========================================================
    # PARAMETERS
    # ========================================================

    transformer_parameters = sum(
        parameter.numel()
        for parameter in transformer.parameters()
    )

    lm_head_parameters = sum(
        parameter.numel()
        for parameter in lm_head.parameters()
    )

    total_parameters = (
        transformer_parameters
        + lm_head_parameters
    )

    print(
        f"\nParamètres Transformer : "
        f"{transformer_parameters:,}"
    )

    print(
        f"Paramètres LM Head    : "
        f"{lm_head_parameters:,}"
    )

    print(
        f"Paramètres totaux     : "
        f"{total_parameters:,}"
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = AdamW(
        list(
            transformer.parameters()
        )
        + list(
            lm_head.parameters()
        ),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # ========================================================
    # LOSS
    # ========================================================

    criterion = nn.CrossEntropyLoss()

    # ========================================================
    # LOAD PREVIOUS CHECKPOINT
    # ========================================================

    if previous_checkpoint is not None:

        previous_epoch = (
            load_checkpoint(
                checkpoint_path=previous_checkpoint,
                transformer=transformer,
                lm_head=lm_head,
                optimizer=optimizer,
                vocab_size=vocab_size,
            )
        )

    else:

        previous_epoch = 0

        print(
            "\nAucun checkpoint précédent."
        )

        print(
            "Entraînement à partir de zéro."
        )

    # ========================================================
    # CONFIGURATION
    # ========================================================

    print(
        "\nConfiguration :"
    )

    print(
        f"  Documents           = "
        f"{MAX_DOCUMENT}"
    )

    print(
        f"  Vocabulaire         = "
        f"{vocab_size}"
    )

    print(
        f"  D model             = "
        f"{D_MODEL}"
    )

    print(
        f"  Num heads           = "
        f"{NUM_HEADS}"
    )

    print(
        f"  Hidden dim          = "
        f"{HIDDEN_DIM}"
    )

    print(
        f"  Num blocks          = "
        f"{NUM_BLOCKS}"
    )

    print(
        f"  Max sequence        = "
        f"{MAX_SEQUENCE_LENGTH}"
    )

    print(
        f"  Batch size          = "
        f"{BATCH_SIZE}"
    )

    print(
        f"  Nouvelles époques   = "
        f"{EPOCHS}"
    )

    print(
        f"  Learning rate       = "
        f"{LEARNING_RATE}"
    )

    print(
        f"  Weight decay        = "
        f"{WEIGHT_DECAY}"
    )

    print(
        f"  Gradient clip       = "
        f"{GRADIENT_CLIP}"
    )

    print(
        f"  Ancien checkpoint   = "
        f"{previous_checkpoint}"
    )

    print(
        f"  Nouveau checkpoint  = "
        f"{new_checkpoint}"
    )

    # ========================================================
    # TRAINING
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "REPRISE DE L'ENTRAÎNEMENT"
    )

    print(
        "=" * 70
    )

    print(
        f"\nReprise après l'époque "
        f"{previous_epoch}"
    )

    print(
        f"Nouvelles époques : "
        f"{EPOCHS}"
    )

    # ========================================================
    # CHRONOMÉTRAGE
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "CHRONOMÉTRAGE DE L'ENTRAÎNEMENT"
    )

    print(
        "=" * 70
    )

    training_timer = (
        TrainingTimer()
    )

    training_timer.start()

    # ========================================================
    # EPOCHS
    # ========================================================

    for local_epoch in range(
        1,
        EPOCHS + 1,
    ):

        global_epoch = (
            previous_epoch
            + local_epoch
        )

        print(
            f"\nÉPOQUE "
            f"{global_epoch} "
            f"(nouvelle époque "
            f"{local_epoch}/{EPOCHS})"
        )

        print(
            "-" * 70
        )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        (
            train_loss,
            gradient_norm,
        ) = train_one_epoch(
            transformer=transformer,
            lm_head=lm_head,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        validation_loss = validate(
            transformer=transformer,
            lm_head=lm_head,
            dataloader=validation_loader,
            criterion=criterion,
        )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        train_ppl = calculate_perplexity(
            train_loss
        )

        validation_ppl = (
            calculate_perplexity(
                validation_loss
            )
        )

        print(
            f"\nTrain loss       : "
            f"{train_loss:.6f}"
        )

        print(
            f"Validation loss  : "
            f"{validation_loss:.6f}"
        )

        print(
            f"Train PPL        : "
            f"{train_ppl:.4f}"
        )

        print(
            f"Validation PPL   : "
            f"{validation_ppl:.4f}"
        )

        print(
            f"Gradient norm    : "
            f"{gradient_norm:.6f}"
        )

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        save_checkpoint(
            checkpoint_path=new_checkpoint,
            transformer=transformer,
            lm_head=lm_head,
            optimizer=optimizer,
            epoch=global_epoch,
            train_loss=train_loss,
            validation_loss=validation_loss,
            gradient_norm=gradient_norm,
            vocab_size=vocab_size,
        )

    # ========================================================
    # DURÉE
    # ========================================================

    training_duration = (
        training_timer.stop()
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "DURÉE TOTALE DE L'ENTRAÎNEMENT"
    )

    print(
        "=" * 70
    )

    print(
        f"\nDurée totale de l'entraînement : "
        f"{training_timer.format_duration(training_duration)}"
    )

    print(
        f"Durée totale en secondes : "
        f"{training_duration:.2f}s"
    )

    # ========================================================
    # FIN
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "POURSUITE DE L'ENTRAÎNEMENT TERMINÉE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nDernière époque : "
        f"{previous_epoch + EPOCHS}"
    )

    print(
        f"Ancien checkpoint : "
        f"{previous_checkpoint}"
    )

    print(
        f"Nouveau checkpoint : "
        f"{new_checkpoint}"
    )


if __name__ == "__main__":

    main()