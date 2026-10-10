# scripts/bitnet/heartbeat/gpu_monitor.py

"""
Moniteur GPU totalement indépendant.

Ce fichier est conçu pour être lancé comme un PROCESSUS SÉPARÉ,
complètement détaché de train_bitnet.py.

Il n'importe RIEN depuis train_bitnet.py, et train_bitnet.py
n'importe RIEN depuis ce fichier.

Usage :
    # Depuis un terminal :
    python -m scripts.bitnet.heartbeat.gpu_monitor

    # En arrière-plan :
    nohup python -m scripts.bitnet.heartbeat.gpu_monitor &

    # Depuis une cellule Kaggle :
    subprocess.Popen(["poetry", "run", "python", "-u", "-m",
                      "scripts.bitnet.heartbeat.gpu_monitor"],
                     start_new_session=True)

Fonctionnement :
    - Collecte les métriques GPU toutes les 5 secondes
    - Lit training_state.json pour associer le step courant
    - Écrit dans checkpoints/bitnet/gpu_metrics.json
    - Upload le fichier sur Hugging Face après chaque collecte
    - Tourne indéfiniment jusqu'à Ctrl+C ou kill
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch


# ============================================================
# RACINE DU PROJET
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# CONFIGURATION
# ============================================================

# Fichier JSON de sortie
GPU_METRICS_FILE = Path("checkpoints/bitnet/gpu_metrics.json")

# Fichier training_state.json (pour lire le step courant)
TRAINING_STATE_FILE = Path("checkpoints/bitnet/training_state.json")

# Chemin distant sur Hugging Face
HF_REMOTE_PATH = "checkpoints/bitnet/gpu_metrics.json"

# Intervalle entre deux collectes (secondes)
INTERVAL_SECONDS = 5.0

# Nombre maximum d'entrées conservées dans l'historique
MAX_HISTORY_ENTRIES = 5000

# Activer/désactiver l'upload HF
UPLOAD_TO_HF = True

# Activer/désactiver l'affichage console
VERBOSE = True


# ============================================================
# COLLECTE DES MÉTRIQUES
# ============================================================

def collect_gpu_metrics():
    """
    Collecte les métriques GPU à un instant donné.

    Retourne un dict ou None si aucun GPU n'est disponible.
    """

    if not torch.cuda.is_available():
        return None

    metrics = {
        "timestamp": datetime.now().isoformat(),
        "device_count": torch.cuda.device_count(),
        "devices": [],
    }

    # ---------- Via nvidia-smi ----------

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )

        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 6:
                continue

            index, name, mem_used, mem_total, util, temp = parts

            mem_used = int(mem_used)
            mem_total = int(mem_total)

            metrics["devices"].append({
                "index": int(index),
                "name": name,
                "memory_used_mb": mem_used,
                "memory_total_mb": mem_total,
                "memory_percent": round(
                    mem_used / mem_total * 100, 2
                ) if mem_total > 0 else 0.0,
                "utilization_gpu_percent": int(util),
                "temperature_c": int(temp),
            })

    except Exception as error:
        metrics["nvidia_smi_error"] = str(error)

    # ---------- Via PyTorch ----------

    try:
        total_mem = (
            torch.cuda.get_device_properties(0).total_memory / 1e9
        )
        metrics["torch"] = {
            "allocated_go": round(
                torch.cuda.memory_allocated() / 1e9, 3
            ),
            "reserved_go": round(
                torch.cuda.memory_reserved() / 1e9, 3
            ),
            "total_go": round(total_mem, 3),
        }
    except Exception as error:
        metrics["torch"] = {"error": str(error)}

    return metrics


# ============================================================
# LECTURE DU STEP COURANT
# ============================================================

def read_current_step():
    """
    Lit le step courant depuis training_state.json.

    Retourne un dict {"epoch": int, "step": int} ou None.
    """

    if not TRAINING_STATE_FILE.exists():
        return None

    try:
        with TRAINING_STATE_FILE.open("r", encoding="utf-8") as f:
            state = json.load(f)

        checkpoints = state.get("checkpoints", [])

        if not checkpoints:
            return None

        last = checkpoints[-1]

        return {
            "epoch": last.get("epoch"),
            "step": last.get("step"),
        }

    except Exception:
        return None


# ============================================================
# ÉCRITURE DU FICHIER
# ============================================================

def append_metrics_to_file(metrics):
    """
    Ajoute une entrée au fichier gpu_metrics.json.

    Le fichier contient :
        {
            "last_update": str,
            "entry_count": int,
            "history": [ ... ]
        }
    """

    history = []

    if GPU_METRICS_FILE.exists():
        try:
            with GPU_METRICS_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
                history = data.get("history", [])
        except Exception:
            history = []

    history.append(metrics)

    # Tronquer si trop long
    if len(history) > MAX_HISTORY_ENTRIES:
        history = history[-MAX_HISTORY_ENTRIES:]

    data = {
        "last_update": metrics["timestamp"],
        "entry_count": len(history),
        "history": history,
    }

    GPU_METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)

    with GPU_METRICS_FILE.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ============================================================
# UPLOAD HF
# ============================================================

def upload_metrics_to_hf():
    """
    Upload gpu_metrics.json vers Hugging Face.

    Échoue silencieusement si l'upload échoue.
    """

    if not UPLOAD_TO_HF:
        return

    try:
        from scripts.hub.hub_sync import upload_file_to_hub

        upload_file_to_hub(
            GPU_METRICS_FILE,
            remote_path=HF_REMOTE_PATH,
        )

    except Exception as error:
        print(f"   ⚠️ Upload GPU metrics HF échoué : {error}")


# ============================================================
# BOUCLE PRINCIPALE
# ============================================================

def main():
    """Boucle de surveillance infinie."""

    print("=" * 70)
    print("🧠 GPU MONITOR — STANDALONE")
    print("=" * 70)

    print(f"\nIntervalle     : {INTERVAL_SECONDS}s")
    print(f"Fichier sortie : {GPU_METRICS_FILE}")
    print(f"Fichier step   : {TRAINING_STATE_FILE}")
    print(f"Upload HF      : {'✅' if UPLOAD_TO_HF else '❌'}")
    print(f"Historique max : {MAX_HISTORY_ENTRIES} entrées")

    if not torch.cuda.is_available():
        print("\n❌ Aucun GPU détecté. Arrêt.")
        return

    print(f"\nGPU détecté(s) : {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"   [{i}] {torch.cuda.get_device_name(i)}")

    print("\n🚀 Démarrage de la boucle de surveillance...")
    print("   (Ctrl+C pour arrêter)\n")

    beat_count = 0

    try:
        while True:

            metrics = collect_gpu_metrics()

            if metrics is not None:

                # Associer le step courant
                current = read_current_step()

                if current is not None:
                    metrics["current_epoch"] = current["epoch"]
                    metrics["current_step"] = current["step"]

                # Écrire dans le fichier
                append_metrics_to_file(metrics)

                # Upload HF
                upload_metrics_to_hf()

                # Affichage console
                beat_count += 1

                if VERBOSE and metrics["devices"]:

                    d = metrics["devices"][0]

                    step_str = (
                        f"step={current['step']}"
                        if current else "step=?"
                    )

                    print(
                        f"🧠 [{datetime.now().strftime('%H:%M:%S')}] "
                        f"#{beat_count:05d} | "
                        f"{step_str} | "
                        f"GPU {d['memory_used_mb']}/{d['memory_total_mb']} Mo "
                        f"({d['memory_percent']}%) | "
                        f"Util {d['utilization_gpu_percent']}% | "
                        f"{d['temperature_c']}°C",
                        flush=True,
                    )

            time.sleep(INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print(f"\n\n🛑 Arrêt du GPU Monitor.")
        print(f"   {beat_count} collectes effectuées.")
        print(f"   Fichier : {GPU_METRICS_FILE}")


if __name__ == "__main__":
    main()