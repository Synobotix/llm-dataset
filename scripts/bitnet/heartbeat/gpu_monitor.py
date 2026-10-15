# scripts/bitnet/heartbeat/gpu_monitor.py

"""
Collecte des métriques GPU + CPU + RAM via nvidia-smi, psutil et PyTorch.

Écrit aussi un fichier monitoring.json à chaque step.
"""

import json
import subprocess
import time
from pathlib import Path

import torch

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


# ============================================================
# CHEMINS
# ============================================================

MONITORING_JSON_PATH = Path("monitoring.json")


# ============================================================
# MÉTRIQUES GPU
# ============================================================

def get_gpu_metrics():
    """
    Retourne un dictionnaire des métriques GPU.
    """

    metrics = {
        "available": False,
        "device_count": 0,
        "devices": [],
        "torch": {},
    }

    if not torch.cuda.is_available():
        return metrics

    metrics["available"] = True
    metrics["device_count"] = torch.cuda.device_count()

    # ---------- Via nvidia-smi ----------

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,"
                "utilization.gpu,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )

        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 7:
                continue

            (
                index,
                name,
                mem_used,
                mem_total,
                util,
                temp,
                power,
            ) = parts

            mem_used = int(float(mem_used))
            mem_total = int(float(mem_total))

            try:
                power_val = float(power)
            except ValueError:
                power_val = None

            metrics["devices"].append({
                "index": int(index),
                "name": name,
                "memory_used_mb": mem_used,
                "memory_total_mb": mem_total,
                "memory_percent": round(
                    mem_used / mem_total * 100, 2
                ) if mem_total > 0 else 0.0,
                "utilization_gpu_percent": int(float(util)),
                "temperature_c": int(float(temp)),
                "power_w": power_val,
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
        metrics["torch"]["error"] = str(error)

    return metrics


# ============================================================
# MÉTRIQUES CPU + RAM
# ============================================================

def get_cpu_metrics():
    """
    Retourne un dictionnaire des métriques CPU + RAM + disque.
    """

    metrics = {
        "available": PSUTIL_AVAILABLE,
        "cpu": {},
        "ram": {},
        "disk": {},
        "load_avg": {},
    }

    if not PSUTIL_AVAILABLE:
        metrics["error"] = "psutil non installé (pip install psutil)"
        return metrics

    try:
        # ---------- CPU ----------
        metrics["cpu"] = {
            "percent": psutil.cpu_percent(interval=None),
            "count_logical": psutil.cpu_count(logical=True),
            "count_physical": psutil.cpu_count(logical=False),
            "freq_mhz": (
                psutil.cpu_freq().current
                if psutil.cpu_freq() else None
            ),
        }

        # ---------- RAM ----------
        ram = psutil.virtual_memory()
        metrics["ram"] = {
            "total_go": round(ram.total / 1e9, 3),
            "used_go": round(ram.used / 1e9, 3),
            "available_go": round(ram.available / 1e9, 3),
            "percent": ram.percent,
        }

        # ---------- Disque ----------
        disk = psutil.disk_usage("/")
        metrics["disk"] = {
            "total_go": round(disk.total / 1e9, 3),
            "used_go": round(disk.used / 1e9, 3),
            "free_go": round(disk.free / 1e9, 3),
            "percent": disk.percent,
        }

        # ---------- Load average ----------
        try:
            load1, load5, load15 = psutil.getloadavg()
            metrics["load_avg"] = {
                "1min": round(load1, 2),
                "5min": round(load5, 2),
                "15min": round(load15, 2),
            }
        except (AttributeError, OSError):
            pass

    except Exception as error:
        metrics["error"] = str(error)

    return metrics


# ============================================================
# MÉTRIQUES COMBINÉES
# ============================================================

def get_system_metrics():
    """
    Retourne toutes les métriques : GPU + CPU + RAM.
    """

    return {
        "timestamp": time.time(),
        "gpu": get_gpu_metrics(),
        "cpu": get_cpu_metrics(),
    }


# ============================================================
# ÉCRITURE DANS monitoring.json
# ============================================================

def write_monitoring_json(
    step: int,
    loss: float = None,
    gradient_norm: float = None,
    epoch: int = None,
    extra: dict = None,
    path: Path = MONITORING_JSON_PATH,
):
    """
    Écrit un fichier monitoring.json avec l'état actuel
    du système + les métriques d'entraînement.
    """

    payload = {
        "step": step,
        "epoch": epoch,
        "loss": loss,
        "gradient_norm": gradient_norm,
        "updated_at": time.strftime(
            "%Y-%m-%dT%H:%M:%S"
        ),
        "system": get_system_metrics(),
    }

    if extra:
        payload.update(extra)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Écriture atomique (évite la corruption si lecture concurrente)
    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    tmp_path.replace(path)

    return payload