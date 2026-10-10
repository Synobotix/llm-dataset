# scripts/bitnet/heartbeat/heartbeat.py

"""
Module de heartbeat pour l'entraînement BitNet.

Objectif :
    Émettre un message périodique pendant l'entraînement afin de :
    - prouver au client Jupyter (nbclient) que la cellule est active ;
    - éviter le timeout de cellule silencieuse ;
    - surveiller l'usage mémoire GPU en temps réel ;
    - fournir une trace horodatée pour diagnostiquer les blocages.

Usage :
    from scripts.bitnet.heartbeat.heartbeat import Heartbeat

    heartbeat = Heartbeat(interval=30)

    for step, batch in enumerate(dataloader):
        ...
        heartbeat.beat(step=step, loss=loss.item())
"""

import sys
import time
from datetime import datetime, timedelta

import torch


class Heartbeat:
    """
    Émet un message à intervalle régulier pour maintenir la session active
    et surveiller la mémoire GPU.
    """

    def __init__(
        self,
        interval: float = 30.0,
        enabled: bool = True,
        prefix: str = "💓",
    ):
        """
        Args:
            interval: intervalle en secondes entre deux battements.
            enabled: active ou désactive le heartbeat.
            prefix: préfixe visuel des messages.
        """

        self.interval = float(interval)
        self.enabled = bool(enabled)
        self.prefix = prefix

        self._last_beat_time = time.time()
        self._start_time = time.time()
        self._beat_count = 0

    # ------------------------------------------------------------------
    # CONTRÔLE
    # ------------------------------------------------------------------

    def reset(self):
        """Réinitialise le compteur et le timer."""
        self._last_beat_time = time.time()
        self._start_time = time.time()
        self._beat_count = 0

    def elapsed(self) -> float:
        """Temps écoulé depuis le début."""
        return time.time() - self._start_time

    def elapsed_human(self) -> str:
        """Temps écoulé formaté."""
        return str(timedelta(seconds=int(self.elapsed())))

    # ------------------------------------------------------------------
    # MÉMOIRE
    # ------------------------------------------------------------------

    @staticmethod
    def _gpu_memory_string() -> str:
        """Retourne l'usage mémoire GPU sous forme lisible."""

        if not torch.cuda.is_available():
            return ""

        try:
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            return f" | GPU: {allocated:.2f}/{reserved:.2f} Go"
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # HEARTBEAT
    # ------------------------------------------------------------------

    def beat(
        self,
        step: int | None = None,
        loss: float | None = None,
        extra: str = "",
        force: bool = False,
    ):
        """
        Émet un heartbeat si l'intervalle est écoulé.

        Args:
            step: numéro de step actuel.
            loss: loss actuelle.
            extra: information supplémentaire libre.
            force: force l'émission même si l'intervalle n'est pas écoulé.
        """

        if not self.enabled:
            return

        now = time.time()

        if not force and (now - self._last_beat_time) < self.interval:
            return

        self._last_beat_time = now
        self._beat_count += 1

        # -------- Construction du message --------

        timestamp = datetime.now().strftime("%H:%M:%S")

        parts = [f"{self.prefix} [{timestamp}]"]

        if step is not None:
            parts.append(f"Step {step}")

        if loss is not None:
            parts.append(f"Loss: {loss:.4f}")

        parts.append(f"| {self.elapsed_human()}")

        mem = self._gpu_memory_string()
        if mem:
            parts.append(mem)

        if extra:
            parts.append(extra)

        message = " ".join(parts)

        # -------- Émission --------

        print(message, flush=True)
        sys.stdout.flush()

    def __call__(self, **kwargs):
        """Permet d'appeler l'instance directement."""
        self.beat(**kwargs)