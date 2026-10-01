from pathlib import Path

from torch.utils.data import DataLoader

from llm.data.dataset import CausalLanguageModelingDataset


TRAIN_FILE = Path(
    "data/tokenized/train.jsonl"
)

VALIDATION_FILE = Path(
    "data/tokenized/validation.jsonl"
)


BATCH_SIZE = 2

SHUFFLE_TRAIN = True

NUM_WORKERS = 0


def create_train_dataloader():

    dataset = CausalLanguageModelingDataset(
        TRAIN_FILE
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE_TRAIN,
        num_workers=NUM_WORKERS,
    )

    return dataloader


def create_validation_dataloader():

    dataset = CausalLanguageModelingDataset(
        VALIDATION_FILE
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    return dataloader