from pathlib import Path


# ============================================================
# DATASET
# ============================================================

# Nombre maximum de documents utilisés pour cette expérience.
#
# Pour passer à 40 documents :
#
#     MAX_DOCUMENT = 40
#
# Pour passer à 100 :
#
#     MAX_DOCUMENT = 100
#
MAX_DOCUMENT = 50000

# Proportion du dataset utilisée pour l'entraînement.
TRAIN_RATIO = 0.80

# Seed permettant de conserver un split reproductible.
RANDOM_SEED = 42


# ============================================================
# TOKENIZER
# ============================================================

# Taille maximale demandée au tokenizer.
#
# ATTENTION :
# Ce n'est PAS forcément la taille réelle du vocabulaire.
#
# Exemple :
#
#     MAX_VOCAB_SIZE = 16000
#
# peut produire :
#
#     vocabulaire réel = 994
#     ou 2500
#     ou 7000
#     etc.
#
MAX_VOCAB_SIZE = 50000

MIN_FREQUENCY = 2

SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
]


# ============================================================
# TOKENISATION
# ============================================================

BLOCK_SIZE = 256


# ============================================================
# BITNET / TRANSFORMER
# ============================================================

D_MODEL = 256

NUM_HEADS = 8

HIDDEN_DIM = 680

NUM_BLOCKS = 480

MAX_SEQUENCE_LENGTH = BLOCK_SIZE


# ============================================================
# ENTRAÎNEMENT
# ============================================================

BATCH_SIZE = 2

EPOCHS = 2

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 0.01

GRADIENT_CLIP = 1.0

# ============================================================
# SAUVEGARDE PÉRIODIQUE
# ============================================================

# Sauvegarde un checkpoint tous les N steps
SAVE_EVERY_N_STEPS = 1000   # ← à changer ici

# Activer le nettoyage automatique des anciens checkpoints step
AUTO_CLEAN_STEP_CHECKPOINTS = True


# ============================================================
# CHEMINS
# ============================================================

DATA_DIR = Path("data")

PROCESSED_DIR = DATA_DIR / "processed"

TOKENIZED_DIR = DATA_DIR / "tokenized"

TOKENIZER_DIR = Path("tokenizer")


# Dataset brut nettoyé.
CLEAN_DATASET_FILE = (
    PROCESSED_DIR / "c4_clean.jsonl"
)


# Dataset après split.
TRAIN_FILE = (
    PROCESSED_DIR / "train.jsonl"
)

VALIDATION_FILE = (
    PROCESSED_DIR / "validation.jsonl"
)


# Dataset tokenisé.
TRAIN_OUTPUT_FILE = (
    TOKENIZED_DIR / "train.jsonl"
)

VALIDATION_OUTPUT_FILE = (
    TOKENIZED_DIR / "validation.jsonl"
)


# Tokenizer.
TOKENIZER_FILE = (
    TOKENIZER_DIR / "tokenizer.json"
)

TOKENIZER_VOCAB_FILE = (
    TOKENIZER_DIR / "vocab.json"
)

TOKENIZER_CONFIG_FILE = (
    TOKENIZER_DIR / "config.json"
)


# ============================================================
# AFFICHAGE
# ============================================================

def print_parameters():
    """
    Affiche les paramètres principaux du projet.
    """

    print("=" * 70)
    print("PARAMÈTRES DU PROJET BITNET")
    print("=" * 70)

    print()
    print("DATASET")
    print(f"  Documents maximum : {MAX_DOCUMENT}")
    print(f"  Train ratio       : {TRAIN_RATIO}")
    print(f"  Random seed       : {RANDOM_SEED}")

    print()
    print("TOKENIZER")
    print(f"  Max vocab size    : {MAX_VOCAB_SIZE}")
    print(f"  Min frequency     : {MIN_FREQUENCY}")

    print()
    print("TOKENISATION")
    print(f"  Block size        : {BLOCK_SIZE}")

    print()
    print("BITNET")
    print(f"  D model           : {D_MODEL}")
    print(f"  Num heads         : {NUM_HEADS}")
    print(f"  Hidden dim        : {HIDDEN_DIM}")
    print(f"  Num blocks        : {NUM_BLOCKS}")
    print(f"  Max sequence      : {MAX_SEQUENCE_LENGTH}")

    print()
    print("ENTRAÎNEMENT")
    print(f"  Batch size        : {BATCH_SIZE}")
    print(f"  Epochs            : {EPOCHS}")
    print(f"  Learning rate     : {LEARNING_RATE}")
    print(f"  Weight decay      : {WEIGHT_DECAY}")
    print(f"  Gradient clip     : {GRADIENT_CLIP}")

    print("=" * 70)