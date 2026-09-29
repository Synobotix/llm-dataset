
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader


class Trainer:
    """
    Gestionnaire de l'entraînement du Student v1.

    Pipeline d'un batch :

        input_ids
             ↓
        Transformer causal
             ↓
        logits
             ↓
        CrossEntropyLoss
             ↓
        backward()
             ↓
        gradient clipping
             ↓
        optimizer.step()
             ↓
        scheduler.step()
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        validation_dataloader: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        device: str = "cpu",
        epochs: int = 1,
        gradient_clip: float = 1.0,
        eval_every: int = 20,
        save_every: int = 50,
        checkpoint_dir: str = "checkpoints/student_v1",
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.device = torch.device(device)
        self.epochs = epochs

        self.gradient_clip = gradient_clip
        self.eval_every = eval_every
        self.save_every = save_every

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.global_step = 0

        self.model.to(self.device)

    # ========================================================
    # ENTRAÎNEMENT COMPLET
    # ========================================================

    def train(self) -> None:

        print("=" * 60)
        print("DÉBUT DE L'ENTRAÎNEMENT")
        print("=" * 60)

        print(f"Device      : {self.device}")
        print(f"Epochs      : {self.epochs}")
        print(
            f"Train size  : "
            f"{len(self.train_dataloader.dataset)}"
        )

        if self.validation_dataloader is not None:
            print(
                f"Validation  : "
                f"{len(self.validation_dataloader.dataset)}"
            )

        print("=" * 60)

        for epoch in range(1, self.epochs + 1):

            train_loss = self._train_epoch(epoch)

            print()
            print(
                f"Epoch {epoch}/{self.epochs} "
                f"- Loss : {train_loss:.4f}"
            )

            if self.validation_dataloader is not None:

                validation_loss = self.evaluate()

                print(
                    f"Validation Loss : "
                    f"{validation_loss:.4f}"
                )

        self.save_checkpoint(
            filename="student_v1_final.pt"
        )

        print("=" * 60)
        print("ENTRAÎNEMENT TERMINÉ")
        print("=" * 60)

    # ========================================================
    # UNE EPOCH
    # ========================================================

    def _train_epoch(self, epoch: int) -> float:

        self.model.train()

        total_loss = 0.0
        num_batches = 0

        for batch_index, batch in enumerate(
            self.train_dataloader,
            start=1
        ):

            input_ids = batch["input_ids"].to(
                self.device,
                non_blocking=True
            )

            labels = batch["labels"].to(
                self.device,
                non_blocking=True
            )

            # ------------------------------------------------
            # Remise à zéro des gradients
            # ------------------------------------------------

            self.optimizer.zero_grad(
                set_to_none=True
            )

            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

            logits = self.model(input_ids)

            # ------------------------------------------------
            # Loss
            # ------------------------------------------------

            loss = self._compute_loss(
                logits,
                labels
            )

            # ------------------------------------------------
            # Backward
            # ------------------------------------------------

            loss.backward()

            # ------------------------------------------------
            # Gradient clipping
            # ------------------------------------------------

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip
            )

            # ------------------------------------------------
            # Mise à jour des paramètres
            # ------------------------------------------------

            self.optimizer.step()

            # ------------------------------------------------
            # Scheduler
            # ------------------------------------------------

            if self.scheduler is not None:
                self.scheduler.step()

            self.global_step += 1

            total_loss += loss.item()
            num_batches += 1

            # ------------------------------------------------
            # Logging
            # ------------------------------------------------

            if (
                self.eval_every > 0
                and self.global_step % self.eval_every == 0
            ):
                learning_rate = self.optimizer.param_groups[0]["lr"]

                print(
                    f"Step {self.global_step:04d} "
                    f"| Epoch {epoch} "
                    f"| Loss {loss.item():.4f} "
                    f"| LR {learning_rate:.6f}"
                )

            # ------------------------------------------------
            # Checkpoint
            # ------------------------------------------------

            if (
                self.save_every > 0
                and self.global_step % self.save_every == 0
            ):
                self.save_checkpoint(
                    filename=(
                        f"checkpoint_step_"
                        f"{self.global_step}.pt"
                    )
                )

        return total_loss / max(num_batches, 1)

    # ========================================================
    # ÉVALUATION
    # ========================================================

    def evaluate(self) -> float:

        if self.validation_dataloader is None:
            raise ValueError(
                "Validation DataLoader absent."
            )

        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():

            for batch in self.validation_dataloader:

                input_ids = batch["input_ids"].to(
                    self.device,
                    non_blocking=True
                )

                labels = batch["labels"].to(
                    self.device,
                    non_blocking=True
                )

                logits = self.model(input_ids)

                loss = self._compute_loss(
                    logits,
                    labels
                )

                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(num_batches, 1)

    # ========================================================
    # CALCUL DE LA LOSS
    # ========================================================

    @staticmethod
    def _compute_loss(
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:

        batch_size = logits.size(0)
        sequence_length = logits.size(1)
        vocab_size = logits.size(2)

        logits = logits.reshape(
            batch_size * sequence_length,
            vocab_size
        )

        labels = labels.reshape(
            batch_size * sequence_length
        )

        return nn.functional.cross_entropy(
            logits,
            labels
        )

    # ========================================================
    # CHECKPOINT
    # ========================================================

    def save_checkpoint(
        self,
        filename: str
    ) -> None:

        checkpoint_path = (
            self.checkpoint_dir / filename
        )

        checkpoint = {
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": (
                self.optimizer.state_dict()
            ),
        }

        if self.scheduler is not None:

            checkpoint[
                "scheduler_state_dict"
            ] = self.scheduler.state_dict()

        torch.save(
            checkpoint,
            checkpoint_path
        )

        print(
            f"Checkpoint sauvegardé : "
            f"{checkpoint_path}"
        )
