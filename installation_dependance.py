#!/usr/bin/env python3
"""
Installation de l'environnement de développement :
- Poetry
- Python 3.11
- Environnement virtuel Poetry
- Dépendances du projet
- torch CUDA 12.1
"""

import os
import subprocess
import sys


# ============================================================
# Helpers
# ============================================================

def run(cmd, check=True, capture=False, shell=False):
    """Exécute une commande et affiche un feedback."""
    print(f"▶️  {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    result = subprocess.run(
        cmd,
        check=check,
        shell=shell,
        capture_output=capture,
        text=True,
    )
    return result


def ensure_path():
    """Ajoute ~/.local/bin au PATH si nécessaire."""
    local_bin = os.path.expanduser("~/.local/bin")
    if local_bin not in os.environ["PATH"]:
        os.environ["PATH"] = f"{local_bin}:{os.environ['PATH']}"


# ============================================================
# 1. Installation de Poetry
# ============================================================

def install_poetry():
    print("\n" + "=" * 70)
    print("1. Installation de Poetry")
    print("=" * 70)

    # Vérifie si poetry est déjà installé
    check = subprocess.run(
        ["which", "poetry"],
        capture_output=True, text=True,
    )
    if check.returncode == 0:
        print(f"✅ Poetry déjà installé : {check.stdout.strip()}")
        ensure_path()
        return

    run(
        "curl -sSL https://install.python-poetry.org | python3 -",
        shell=True,
    )
    ensure_path()

    result = run(["poetry", "--version"], capture=True)
    print(f"✅ {result.stdout.strip()}")


# ============================================================
# 2. Installation de Python 3.11
# ============================================================

def install_python311():
    print("\n" + "=" * 70)
    print("2. Installation de Python 3.11")
    print("=" * 70)

    # Vérifie si python3.11 est déjà présent
    check = subprocess.run(
        ["which", "python3.11"],
        capture_output=True, text=True,
    )
    if check.returncode == 0:
        result = run(["python3.11", "--version"], capture=True)
        print(f"✅ {result.stdout.strip()} (déjà installé)")
        return

    run(["apt-get", "update"])
    run([
        "apt-get", "install", "-y",
        "python3.11", "python3.11-venv", "python3.11-dev",
    ])

    result = run(["python3.11", "--version"], capture=True)
    print(f"✅ {result.stdout.strip()}")


# ============================================================
# 3. Configuration de l'environnement Poetry
# ============================================================

def setup_poetry_env():
    print("\n" + "=" * 70)
    print("3. Configuration de l'environnement Poetry (Python 3.11)")
    print("=" * 70)

    run(["poetry", "env", "use", "python3.11"])
    run(["poetry", "install", "--no-root"])

    result = run(
        ["poetry", "run", "python", "--version"],
        capture=True,
    )
    print(f"✅ Python dans Poetry : {result.stdout.strip()}")


# ============================================================
# 4. Installation de torch CUDA 12.1
# ============================================================

def install_torch_cuda():
    print("\n" + "=" * 70)
    print("4. Installation de torch CUDA 12.1")
    print("=" * 70)

    # Désinstalle torch CPU si présent
    run(
        ["poetry", "run", "pip", "uninstall", "-y",
         "torch", "torchvision", "torchaudio"],
        check=False,
    )

    result = subprocess.run(
        ["poetry", "run", "pip", "install",
         "torch==2.4.1", "torchvision==0.19.1", "torchaudio==2.4.1",
         "--index-url", "https://download.pytorch.org/whl/cu121"],
        capture_output=True, text=True,
    )

    if result.returncode != 0:
        print("❌ Échec de l'installation torch CUDA")
        print("STDOUT:", result.stdout[-1000:])
        print("STDERR:", result.stderr[-2000:])
        raise RuntimeError("Installation torch CUDA échouée")

    print("✅ torch CUDA installé")


# ============================================================
# 5. Vérification finale
# ============================================================

def verify():
    print("\n" + "=" * 70)
    print("5. Vérification finale")
    print("=" * 70)

    result = subprocess.run(
        ["poetry", "run", "python", "-c",
         "import torch; "
         "print('torch', torch.__version__); "
         "print('cuda', torch.cuda.is_available()); "
         "print('gpu', torch.cuda.get_device_name(0) "
         "if torch.cuda.is_available() else 'N/A')"],
        capture_output=True, text=True,
    )
    print(result.stdout)


# ============================================================
# Main
# ============================================================

def main():
    if os.geteuid() != 0:
        print("⚠️  Ce script doit être lancé en root (apt-get).")
        print("   Sur Colab, vous êtes root par défaut.")
        sys.exit(1)

    install_poetry()
    install_python311()
    setup_poetry_env()
    install_torch_cuda()
    verify()

    print("\n🎉 Installation terminée.")


if __name__ == "__main__":
    main()
