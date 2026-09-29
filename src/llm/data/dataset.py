
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class LLMDataset(Dataset):
    """
    Dataset pour l'entraînement d'un modèle de langage causal.

    Format attendu du JSONL :

    {
        "input_ids": [12, 45, 78, ...],
        "labels": [45, 78, 91, ...]
    }
    """

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)

        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Dataset introuvable : {self.file_path}"
            )

        self.samples = []

        self._load()

    def _load(self) -> None:
        with self.file_path.open("r", encoding="utf-8") as file:

            for line_number, line in enumerate(file, start=1):

                line = line.strip()

                if not line:
                    continue

                try:
                    data = json.loads(line)

                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"JSON invalide ligne {line_number} "
                        f"dans {self.file_path}"
                    ) from exc

                if "input_ids" not in data:
                    raise KeyError(
                        f"'input_ids' absent ligne {line_number} "
                        f"dans {self.file_path}"
                    )

                if "labels" not in data:
                    raise KeyError(
                        f"'labels' absent ligne {line_number} "
                        f"dans {self.file_path}"
                    )

                input_ids = data["input_ids"]
                labels = data["labels"]

                if not isinstance(input_ids, list):
                    raise TypeError(
                        f"'input_ids' doit être une liste "
                        f"ligne {line_number}"
                    )

                if not isinstance(labels, list):
                    raise TypeError(
                        f"'labels' doit être une liste "
                        f"ligne {line_number}"
                    )

                if len(input_ids) != len(labels):
                    raise ValueError(
                        f"'input_ids' et 'labels' doivent avoir "
                        f"la même longueur ligne {line_number}"
                    )

                if len(input_ids) == 0:
                    raise ValueError(
                        f"Séquence vide ligne {line_number}"
                    )

                self.samples.append(
                    {
                        "input_ids": input_ids,
                        "labels": labels,
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
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
