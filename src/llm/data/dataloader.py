from pathlib import Path

from torch.utils.data import DataLoader

from llm.data.dataset import LLMDataset


def create_dataloader(
    dataset_path: str | Path,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    """
    Crée un DataLoader pour le dataset donné.
    """

    dataset = LLMDataset(dataset_path)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader


def create_train_dataloader(
    dataset_path: str | Path,
    batch_size: int,
    num_workers: int = 0,
) -> DataLoader:
    """
    DataLoader d'entraînement.
    """

    return create_dataloader(
        dataset_path=dataset_path,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )


def create_validation_dataloader(
    dataset_path: str | Path,
    batch_size: int,
    num_workers: int = 0,
) -> DataLoader:
    """
    DataLoader de validation.
    """

    return create_dataloader(
        dataset_path=dataset_path,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )