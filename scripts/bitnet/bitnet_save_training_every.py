"""
Sauvegarde périodique du checkpoint BitNet.

Sauvegarde tous les N steps (par défaut 10 000) :
- En local  : checkpoints/bitnet/checkpoint_step<N>.pt
- Sur HF    : checkpoints/bitnet/checkpoint_step<N>.pt

Objectif : ne pas perdre l'entraînement si le serveur coupe.
"""

import sys
from pathlib import Path


# ============================================================
# RACINE DU PROJET
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# CONFIGURATION
# ============================================================

# Intervalle de sauvegarde (en steps)
SAVE_EVERY_N_STEPS = 10_000

# Dossier local des checkpoints
LOCAL_CHECKPOINT_DIR = Path("checkpoints/bitnet")

# Dossier distant sur HF
HF_CHECKPOINT_FOLDER = "checkpoints/bitnet"


# ============================================================
# CLASSE DE SUIVI
# ============================================================

class StepCheckpointSaver:
    """
    Gère la sauvegarde périodique du checkpoint à intervalles
    réguliers de steps.
    """

    def __init__(
        self,
        save_every_n_steps=SAVE_EVERY_N_STEPS,
        local_dir=LOCAL_CHECKPOINT_DIR,
        hf_folder=HF_CHECKPOINT_FOLDER,
        enabled=True,
    ):
        self.save_every_n_steps = save_every_n_steps
        self.local_dir = Path(local_dir)
        self.hf_folder = hf_folder
        self.enabled = enabled

        # Compteur interne
        self.last_saved_step = 0

        # Créer le dossier local
        self.local_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Faut-il sauvegarder maintenant ?
    # --------------------------------------------------------

    def should_save(self, current_step):
        """
        Retourne True si on doit sauvegarder à ce step.
        """

        if not self.enabled:
            return False

        if current_step < self.save_every_n_steps:
            return False

        # Sauvegarder si on a dépassé le dernier multiple
        next_milestone = (
            (self.last_saved_step // self.save_every_n_steps) + 1
        ) * self.save_every_n_steps

        return current_step >= next_milestone

    # --------------------------------------------------------
    # Nom du fichier
    # --------------------------------------------------------

    def get_checkpoint_path(self, step):
        """
        Retourne le chemin local du checkpoint pour ce step.
        """

        filename = f"checkpoint_step{step}.pt"

        return self.local_dir / filename

    # --------------------------------------------------------
    # Sauvegarde
    # --------------------------------------------------------

    def save(
        self,
        step,
        save_checkpoint_fn,
        **kwargs,
    ):
        """
        Sauvegarde le checkpoint via la fonction fournie.

        Paramètres :
            step                : step actuel
            save_checkpoint_fn  : fonction de sauvegarde (celle de train_bitnet.py)
            **kwargs            : arguments passés à save_checkpoint_fn
                                  (transformer, lm_head, optimizer, etc.)

        Retourne le chemin du fichier sauvegardé (ou None).
        """

        if not self.should_save(step):
            return None

        checkpoint_path = self.get_checkpoint_path(step)

        print()
        print("=" * 70)
        print(f"💾 SAUVEGARDE PÉRIODIQUE — STEP {step}")
        print("=" * 70)

        # ----------------------------------------------------
        # Sauvegarde locale
        # ----------------------------------------------------

        try:
            save_checkpoint_fn(
                checkpoint_path=checkpoint_path,
                **kwargs,
            )

            size_mb = checkpoint_path.stat().st_size / (1024 * 1024)

            print(f"\n✅ Checkpoint local : {checkpoint_path}")
            print(f"   Taille          : {size_mb:.2f} Mo")

        except Exception as error:

            print(f"\n❌ Échec sauvegarde locale : {error}")
            return None

        # ----------------------------------------------------
        # Upload HF
        # ----------------------------------------------------

        try:

            from scripts.hub.hub_sync import upload_file_to_hub

            remote_path = f"{self.hf_folder}/{checkpoint_path.name}"

            upload_file_to_hub(
                checkpoint_path,
                remote_path=remote_path,
            )

            print(f"\n✅ Upload HF : {remote_path}")

        except Exception as error:

            print(f"\n⚠️ Upload HF échoué : {error}")
            print("   (le checkpoint local est intact)")

        # ----------------------------------------------------
        # Mettre à jour le compteur
        # ----------------------------------------------------

        self.last_saved_step = step

        print()
        print("=" * 70)
        print(f"PROCHAIN CHECKPOINT : step {step + self.save_every_n_steps}")
        print("=" * 70)

        return checkpoint_path

    # --------------------------------------------------------
    # État actuel
    # --------------------------------------------------------

    def print_status(self):

        print(
            f"💾 Sauvegarde périodique : "
            f"tous les {self.save_every_n_steps:,} steps"
        )

        print(
            f"   Prochain : step "
            f"{self.last_saved_step + self.save_every_n_steps:,}"
        )