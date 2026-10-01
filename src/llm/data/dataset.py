import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class CausalLanguageModelingDataset(Dataset):
    """
    Dataset PyTorch pour l'entraînement d'un
    Transformer causal de type GPT.
    """

    def __init__(self, file_path: str | Path):

        self.file_path = Path(file_path)

        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Dataset introuvable : {self.file_path}"
            )

        self.samples = []

        self._load_data()

    def _load_data(self):
        """
        Charge les séquences depuis le JSONL.
        """

        with self.file_path.open(
            "r",
            encoding="utf-8"
        ) as file:

            for line_number, line in enumerate(
                file,
                start=1
            ):

                line = line.strip()

                if not line:
                    continue

                try:
                    sample = json.loads(line)

                except json.JSONDecodeError as error:

                    print(
                        f"[WARNING] "
                        f"Ligne {line_number} ignorée : "
                        f"{error}"
                    )

                    continue

                input_ids = sample.get(
                    "input_ids"
                )

                labels = sample.get(
                    "labels"
                )

                if not isinstance(
                    input_ids,
                    list
                ):
                    continue

                if not isinstance(
                    labels,
                    list
                ):
                    continue

                if len(input_ids) != len(labels):
                    raise ValueError(
                        f"Longueurs différentes à la "
                        f"ligne {line_number}"
                    )

                self.samples.append(
                    {
                        "input_ids": input_ids,
                        "labels": labels,
                    }
                )

    def __len__(self):
        """
        Nombre total de séquences.
        """

        return len(self.samples)

    def __getitem__(self, index):
        """
        Retourne une séquence sous forme de tensors PyTorch.
        """

        sample = self.samples[index]

        input_ids = torch.tensor(
            sample["input_ids"],
            dtype=torch.long
        )

        labels = torch.tensor(
            sample["labels"],
            dtype=torch.long
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
        }