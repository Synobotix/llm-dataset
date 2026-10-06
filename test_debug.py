import os
from pathlib import Path

print("=" * 70)
print("VÉRIFICATION DES FICHIERS")
print("=" * 70)

# CWD
print(f"\nCWD : {os.getcwd()}")

# PYTHONPATH
print(f"PYTHONPATH : {os.environ.get('PYTHONPATH', 'NON DÉFINI')}")

# HF_TOKEN
token = os.environ.get("HF_TOKEN")
print(f"HF_TOKEN : {'✅ ' + token[:6] + '...' + token[-4:] if token else '❌ manquant'}")

# Arborescence
print("\n" + "=" * 70)
print("ARBORESCENCE")
print("=" * 70)

for folder in [
    "data/processed",
    "data/tokenized",
    "tokenizer",
    "checkpoints/bitnet",
    "documentation/result/bitnet",
    "src/llm/tokenizer",
    "scripts/nemotron",
]:

    path = Path(folder)

    print(f"\n📁 {folder}/")

    if not path.exists():
        print(f"   ❌ Absent")
        continue

    files = sorted(path.glob("*"))

    if not files:
        print(f"   ⚠️ Vide")
        continue

    for f in files[:10]:
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            print(f"   ✅ {f.name} ({size_kb:.2f} Ko)")
        else:
            print(f"   📁 {f.name}/")

# Vérifier les imports
print("\n" + "=" * 70)
print("IMPORTS")
print("=" * 70)

import subprocess

for module in [
    "llm",
    "llm.config.parameters",
    "llm.tokenizer",
    "llm.tokenizer.train_tokenizer",
    "scripts",
    "scripts.nemotron",
    "scripts.nemotron.download_and_prepare",
]:

    r = subprocess.run(
        ["poetry", "run", "python", "-c", f"import {module}"],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )

    status = "✅" if r.returncode == 0 else "❌"
    print(f"{status} {module}")

    if r.returncode != 0:
        err = r.stderr.strip().split("\n")[-1] if r.stderr else "unknown"
        print(f"   → {err[:120]}")