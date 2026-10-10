# scripts/bitnet/heartbeat/gpu_monitor.py

"""
Collecte des métriques GPU via nvidia-smi et PyTorch.

Utilisable pour enrichir les checkpoints et le training_state.json.
"""

import subprocess
import torch


def get_gpu_metrics():
    """
    Retourne un dictionnaire des métriques GPU.

    Structure :
        {
            "available": bool,
            "device_count": int,
            "devices": [
                {
                    "index": int,
                    "name": str,
                    "memory_used_mb": int,
                    "memory_total_mb": int,
                    "memory_percent": float,
                    "utilization_gpu_percent": int,
                    "temperature_c": int,
                },
                ...
            ],
            "torch": {
                "allocated_go": float,
                "reserved_go": float,
                "total_go": float,
            }
        }
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
        metrics["torch"]["error"] = str(error)

    return metrics