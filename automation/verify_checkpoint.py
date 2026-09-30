
from pathlib import Path

import torch


PROJECT_PATH = Path("/content/llm-dataset")
CHECKPOINT_DIR = PROJECT_PATH / "checkpoints/student_v1"
FINAL_CHECKPOINT = CHECKPOINT_DIR / "student_v1_final.pt"


def verify_checkpoint() -> None:
    """
    Vérifie les checkpoints générés par l'entraînement Student v1.
    """

    print("=" * 60)
    print("VERIFICATION CHECKPOINT STUDENT V1")
    print("=" * 60)

    if not CHECKPOINT_DIR.exists():
        raise FileNotFoundError(
            f"Dossier checkpoint introuvable : {CHECKPOINT_DIR}"
        )

    checkpoints = sorted(CHECKPOINT_DIR.glob("*.pt"))

    if not checkpoints:
        raise FileNotFoundError(
            "Aucun checkpoint .pt trouvé."
        )

    for checkpoint_path in checkpoints:

        size_mb = checkpoint_path.stat().st_size / (1024 * 1024)

        print(
            f"{checkpoint_path.name} : "
            f"{size_mb:.2f} MB"
        )

    if not FINAL_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint final introuvable : {FINAL_CHECKPOINT}"
        )

    print()
    print(f"Chargement : {FINAL_CHECKPOINT.name}")

    checkpoint = torch.load(
        FINAL_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    print("Type :", type(checkpoint))
    print("Clés :", list(checkpoint.keys()))

    required_keys = {
        "global_step",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
    }

    missing_keys = required_keys - set(checkpoint.keys())

    if missing_keys:
        raise RuntimeError(
            f"Clés manquantes : {sorted(missing_keys)}"
        )

    global_step = checkpoint["global_step"]

    print("Global step :", global_step)

    if global_step != 189:
        raise RuntimeError(
            f"Global step inattendu : {global_step}. "
            f"Valeur attendue : 189."
        )

    print()
    print("CHECKPOINT STUDENT V1 : OK")
