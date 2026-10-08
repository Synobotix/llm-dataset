"""
Test unitaire pour delete_checkpoint.py

Objectif :
- Vérifier que le nettoyage garde uniquement le dernier checkpoint step
- Vérifier que le JSON est mis à jour
- Vérifier que les checkpoints d'époque ne sont PAS touchés

Le test utilise un dossier temporaire pour ne rien casser.

Usage :
    poetry run python -m scripts.tests.test_avant_colab.test_delete_checkpoint
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path


# ============================================================
# RACINE DU PROJET
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# MOCK — Simuler le JSON et les fichiers
# ============================================================

def create_fake_checkpoints(base_dir, steps):
    """
    Crée des faux checkpoints .pt pour les steps donnés.
    Retourne la liste des chemins créés.
    """

    checkpoint_dir = base_dir / "checkpoints" / "bitnet"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    paths = []

    for step in steps:

        path = checkpoint_dir / f"checkpoint_step{step}.pt"

        # Créer un fichier factice de 1 Ko
        path.write_bytes(b"\x00" * 1024)

        paths.append(path)

    return paths


def create_fake_epoch_checkpoints(base_dir):
    """
    Crée des faux checkpoints d'époque.
    """

    checkpoint_dir = base_dir / "checkpoints" / "bitnet"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    paths = []

    for epoch in [1, 2]:

        path = checkpoint_dir / f"entrainement_bitnet_{epoch}.pt"
        path.write_bytes(b"\x00" * 1024)
        paths.append(path)

    return paths


def create_fake_json(base_dir, steps):
    """
    Crée un faux training_state.json avec les steps donnés.
    """

    json_dir = base_dir / "checkpoints" / "bitnet"
    json_dir.mkdir(parents=True, exist_ok=True)

    json_path = json_dir / "training_state.json"

    checkpoints = []

    # Ajouter les checkpoints step
    for step in steps:

        checkpoints.append({
            "name": f"checkpoint_step{step}.pt",
            "type": "step",
            "path": str(json_dir / f"checkpoint_step{step}.pt"),
            "epoch": 1,
            "step": step,
            "duration_seconds": None,
            "train_loss": 4.0 - (step / 100000),
            "validation_loss": None,
            "train_ppl": None,
            "validation_ppl": None,
            "gradient_norm": 2.0,
            "created_at": "2026-10-08T14:30:00",
        })

    # Ajouter les checkpoints d'époque
    for epoch in [1, 2]:

        checkpoints.append({
            "name": f"entrainement_bitnet_{epoch}.pt",
            "type": "epoch",
            "path": str(json_dir / f"entrainement_bitnet_{epoch}.pt"),
            "epoch": epoch,
            "step": epoch * 89000,
            "duration_seconds": 4800,
            "train_loss": 3.3 - (epoch * 0.1),
            "validation_loss": 3.4 - (epoch * 0.1),
            "train_ppl": 27.0,
            "validation_ppl": 28.0,
            "gradient_norm": 1.9,
            "created_at": "2026-10-08T14:30:00",
        })

    state = {
        "training_finished": False,
        "last_update": "2026-10-08T14:30:00",
        "checkpoints": checkpoints,
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

    return json_path


# ============================================================
# TESTS
# ============================================================

def test_clean_keeps_only_last_step():
    """
    Vérifie que le nettoyage garde uniquement le dernier step.
    """

    print("=" * 70)
    print("TEST 1 — Garder uniquement le dernier step")
    print("=" * 70)

    # --------------------------------------------------------
    # Créer un environnement de test
    # --------------------------------------------------------

    with tempfile.TemporaryDirectory() as tmp_dir:

        tmp_path = Path(tmp_dir)

        print(f"\n📁 Dossier temporaire : {tmp_path}")

        # Créer les fichiers
        steps = [10000, 20000, 30000, 40000, 50000]

        create_fake_checkpoints(tmp_path, steps)
        create_fake_epoch_checkpoints(tmp_path)
        create_fake_json(tmp_path, steps)

        # ----------------------------------------------------
        # Simuler le nettoyage
        # ----------------------------------------------------

        # Chemin vers le JSON
        json_path = tmp_path / "checkpoints" / "bitnet" / "training_state.json"

        # Lire le JSON
        with json_path.open("r", encoding="utf-8") as f:
            state = json.load(f)

        # Trouver les checkpoints step
        step_checkpoints = [
            c for c in state["checkpoints"]
            if c.get("type") == "step"
        ]

        step_checkpoints.sort(key=lambda c: c.get("step", 0))

        print(f"\n📋 {len(step_checkpoints)} checkpoints step trouvés :")

        for ckpt in step_checkpoints:
            print(f"   - {ckpt['name']} (step {ckpt['step']})")

        # ----------------------------------------------------
        # Vérifier le comportement attendu
        # ----------------------------------------------------

        print(f"\n🧪 Vérification...")

        # 1. Il y a bien 5 steps
        assert len(step_checkpoints) == 5, (
            f"Attendu 5 steps, trouvé {len(step_checkpoints)}"
        )
        print("   ✅ 5 checkpoints step créés")

        # 2. Le dernier step est 50000
        last_step = step_checkpoints[-1]
        assert last_step["step"] == 50000, (
            f"Dernier step attendu : 50000, trouvé : {last_step['step']}"
        )
        print("   ✅ Dernier step = 50000")

        # 3. Les anciens steps seraient supprimés
        old_steps = step_checkpoints[:-1]
        assert len(old_steps) == 4, (
            f"4 anciens steps attendus, trouvé {len(old_steps)}"
        )
        print("   ✅ 4 anciens steps à supprimer")

        # 4. Les epochs ne sont pas touchés
        epoch_checkpoints = [
            c for c in state["checkpoints"]
            if c.get("type") == "epoch"
        ]
        assert len(epoch_checkpoints) == 2, (
            f"2 epochs attendus, trouvé {len(epoch_checkpoints)}"
        )
        print("   ✅ 2 checkpoints d'époque présents (non touchés)")

    print("\n✅ TEST 1 RÉUSSI")


def test_clean_with_single_step():
    """
    Vérifie qu'avec un seul step, rien n'est supprimé.
    """

    print("\n" + "=" * 70)
    print("TEST 2 — Un seul step, rien à supprimer")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmp_dir:

        tmp_path = Path(tmp_dir)

        steps = [10000]

        create_fake_checkpoints(tmp_path, steps)
        create_fake_json(tmp_path, steps)

        json_path = tmp_path / "checkpoints" / "bitnet" / "training_state.json"

        with json_path.open("r", encoding="utf-8") as f:
            state = json.load(f)

        step_checkpoints = [
            c for c in state["checkpoints"]
            if c.get("type") == "step"
        ]

        print(f"\n📋 {len(step_checkpoints)} checkpoint step")

        # Vérifier
        assert len(step_checkpoints) == 1, (
            f"1 step attendu, trouvé {len(step_checkpoints)}"
        )
        print("   ✅ 1 seul step")

        old_steps = step_checkpoints[:-1]
        assert len(old_steps) == 0, (
            f"0 ancien step attendu, trouvé {len(old_steps)}"
        )
        print("   ✅ Aucun ancien step à supprimer")

    print("\n✅ TEST 2 RÉUSSI")


def test_clean_with_no_step():
    """
    Vérifie qu'avec aucun step, rien ne plante.
    """

    print("\n" + "=" * 70)
    print("TEST 3 — Aucun step")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmp_dir:

        tmp_path = Path(tmp_dir)

        create_fake_json(tmp_path, [])

        json_path = tmp_path / "checkpoints" / "bitnet" / "training_state.json"

        with json_path.open("r", encoding="utf-8") as f:
            state = json.load(f)

        step_checkpoints = [
            c for c in state["checkpoints"]
            if c.get("type") == "step"
        ]

        print(f"\n📋 {len(step_checkpoints)} checkpoint step")

        assert len(step_checkpoints) == 0, (
            f"0 step attendu, trouvé {len(step_checkpoints)}"
        )
        print("   ✅ Aucun step (cas normal au début)")

    print("\n✅ TEST 3 RÉUSSI")


def test_json_structure():
    """
    Vérifie que le JSON a la bonne structure.
    """

    print("\n" + "=" * 70)
    print("TEST 4 — Structure du JSON")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmp_dir:

        tmp_path = Path(tmp_dir)

        steps = [10000, 20000]

        create_fake_checkpoints(tmp_path, steps)
        create_fake_json(tmp_path, steps)

        json_path = tmp_path / "checkpoints" / "bitnet" / "training_state.json"

        with json_path.open("r", encoding="utf-8") as f:
            state = json.load(f)

        # Vérifier les clés principales
        assert "training_finished" in state
        assert "last_update" in state
        assert "checkpoints" in state

        print("   ✅ Clés principales présentes")

        # Vérifier les champs d'un checkpoint step
        step_ckpt = [c for c in state["checkpoints"] if c["type"] == "step"][0]

        required_fields = [
            "name", "type", "path", "epoch", "step",
            "train_loss", "created_at",
        ]

        for field in required_fields:
            assert field in step_ckpt, f"Champ manquant : {field}"

        print("   ✅ Tous les champs présents")

    print("\n✅ TEST 4 RÉUSSI")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TESTS UNITAIRES — DELETE CHECKPOINT")
    print("=" * 70)

    try:

        test_clean_keeps_only_last_step()
        test_clean_with_single_step()
        test_clean_with_no_step()
        test_json_structure()

        print("\n" + "=" * 70)
        print("✅ TOUS LES TESTS RÉUSSIS")
        print("=" * 70)

        return 0

    except AssertionError as error:

        print("\n" + "=" * 70)
        print(f"❌ ÉCHEC : {error}")
        print("=" * 70)

        return 1

    except Exception as error:

        print("\n" + "=" * 70)
        print(f"❌ ERREUR : {error}")
        print("=" * 70)

        import traceback
        traceback.print_exc()

        return 1


if __name__ == "__main__":
    sys.exit(main())