"""
Script de distillation Knowledge Distillation pour BitNet.
Utilise un teacher HF pour générer des logits, puis entraîne le BitNet student
avec une loss KL divergence (teacher logits vs student logits).
"""

import json
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from llm.config.parameters import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    GRADIENT_CLIP,
    MAX_SEQUENCE_LENGTH,
    TOKENIZER_FILE,
    D_MODEL,
    NUM_HEADS,
    HIDDEN_DIM,
    NUM_BLOCKS,
)

from llm.bitnet_model.bit_transformer import BitTransformer
from llm.bitnet_model.lm_head import LMHead


# ============================================================
# CONFIGURATION
# ============================================================

# Teacher model (HF Hub - doit être accessible)
TEACHER_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # ou "HuggingFaceTB/SmolLM2-360M-Instruct" pour test rapide

# Dataset source pour distillation
DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
DATASET_LANG = None  # None = pas de filtre langue (tout garder, y compris anglais)

# Paramètres distillation
NUM_DISTILL_SAMPLES = 5000        # Nombre d'échantillons à générer
MAX_SOURCE_LENGTH = 1024          # Longueur max texte source
MAX_TARGET_LENGTH = 512           # Longueur max génération teacher
DISTILL_BATCH_SIZE = 4            # Batch pour génération teacher

# Entraînement student
STUDENT_BATCH_SIZE = 2
STUDENT_EPOCHS = 8

# Paths
DISTILLED_DATA_FILE = Path("data/distilled/kd_distilled.jsonl")
TOKENIZED_TRAIN_FILE = Path("data/tokenized/kd_train.jsonl")
TOKENIZED_VAL_FILE = Path("data/tokenized/kd_val.jsonl")
CHECKPOINT_DIR = Path("checkpoint/bitnet_kd")
FINAL_CHECKPOINT = CHECKPOINT_DIR / "bitnet_kd_final.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16


# ============================================================
# DATASET & TOKENIZATION
# ============================================================

class TokenizedKDDataset(Dataset):
    """Dataset pour distillation (data distillation, pas de logits teacher)."""
    
    def __init__(self, file_path: Path, max_length: int):
        self.samples = []
        self.max_length = max_length
        
        print(f"Chargement dataset KD: {file_path}")
        with file_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[WARN] Ligne {line_num}: {e}")
                    continue
                
                input_ids = sample.get("input_ids")
                labels = sample.get("labels")
                
                if not isinstance(input_ids, list) or len(input_ids) != max_length:
                    continue
                if not isinstance(labels, list) or len(labels) != max_length:
                    continue
                
                self.samples.append({
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                })
        
        print(f"  Échantillons valides: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def load_student_tokenizer():
    if not TOKENIZER_FILE.exists():
        raise FileNotFoundError(f"Tokenizer introuvable: {TOKENIZER_FILE}")
    return Tokenizer.from_file(str(TOKENIZER_FILE))


# ============================================================
# ÉTAPE 1: GÉNÉRATION DONNÉES DISTILLATION (Teacher -> Text)
# ============================================================
# Note: Comme le teacher et le student ont des tokenizers différents,
# on fait de la "data distillation" (génération de texte) plutôt que
# de la "logit distillation" (KL divergence). Le student s'entraîne
# ensuite avec CE loss standard sur les données générées.

def build_distillation_prompt(source_text: str) -> str:
    # Prompt en anglais pour modèles anglais (Qwen, Mistral, Llama)
    return f"""You are a teacher model. From the text below, produce a clear, coherent, and self-contained response summarizing the key information. Do not invent anything. Return only the response.

Source text:
{source_text}

Response:"""


def load_teacher():
    print(f"Chargement teacher: {TEACHER_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Config robuste pour Colab GPU
    if DEVICE.type == "cuda":
        try:
            model = AutoModelForCausalLM.from_pretrained(
                TEACHER_MODEL,
                torch_dtype=DTYPE,
                device_map="auto",
                attn_implementation="flash_attention_2",
                low_cpu_mem_usage=True,
            )
            print("✅ Loaded with flash_attention_2 + device_map=auto")
        except Exception as e:
            print(f"⚠️ flash_attention_2 failed: {e}, fallback to eager")
            model = AutoModelForCausalLM.from_pretrained(
                TEACHER_MODEL,
                torch_dtype=DTYPE,
                device_map="auto",
                attn_implementation="eager",
                low_cpu_mem_usage=True,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            TEACHER_MODEL,
            torch_dtype=DTYPE,
            low_cpu_mem_usage=True,
        ).to(DEVICE)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_teacher_responses(teacher_model, teacher_tokenizer, texts: list[str]) -> list[str]:
    """Génère réponses du teacher pour une liste de textes source."""
    prompts = [build_distillation_prompt(t[:MAX_SOURCE_LENGTH * 3]) for t in texts]
    
    inputs = teacher_tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SOURCE_LENGTH,
        padding=True,
    ).to(teacher_model.device)
    
    gen_outputs = teacher_model.generate(
        **inputs,
        max_new_tokens=MAX_TARGET_LENGTH,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=teacher_tokenizer.pad_token_id,
        eos_token_id=teacher_tokenizer.eos_token_id,
        use_cache=True,
    )
    
    responses = []
    for i, out in enumerate(gen_outputs):
        resp = teacher_tokenizer.decode(
            out[inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        responses.append(resp.strip())
    
    return responses


def prepare_distillation_data():
    """Génère le dataset de distillation (texte seulement, pas de logits)."""
    print("=" * 60)
    print("ÉTAPE 1: GÉNÉRATION DONNÉES DISTILLATION (Teacher -> Text)")
    print("=" * 60)
    
    # Charger teacher
    teacher_model, teacher_tokenizer = load_teacher()
    print(f"Teacher vocab size: {teacher_tokenizer.vocab_size}")
    
    # Charger dataset source
    from datasets import load_dataset
    print(f"\nChargement dataset: {DATASET_NAME} ({DATASET_CONFIG})")
    ds = load_dataset(DATASET_NAME, name=DATASET_CONFIG, split="train", streaming=True)
    if DATASET_LANG:
        print(f"Filtrage langue: {DATASET_LANG}...")
        ds = ds.filter(lambda x: x.get("language", "") == DATASET_LANG or x.get("lang", "") == DATASET_LANG)
        print("Filtrage appliqué (streaming)")
    else:
        print("Pas de filtre langue (toutes les langues conservées)")
    
    # Charger student tokenizer
    student_tokenizer = load_student_tokenizer()
    student_vocab_size = student_tokenizer.get_vocab_size()
    print(f"Student vocab size: {student_vocab_size}")
    
    DISTILLED_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    count = 0
    skipped = 0
    batch_texts = []
    batch_sources = []
    scanned = 0
    
    for item in tqdm(ds, total=NUM_DISTILL_SAMPLES * 5, desc="Scan + Distill"):
        scanned += 1
        if count >= NUM_DISTILL_SAMPLES:
            break
        
        text = item.get("text", "").strip()
        if not text or len(text) < 100:
            skipped += 1
            continue
        
        if scanned % 1000 == 0:
            print(f"\n  Scanned: {scanned}, Found: {count}, Skipped: {skipped}")
        
        batch_texts.append(text)
        batch_sources.append(text)
        
        if len(batch_texts) >= DISTILL_BATCH_SIZE:
            try:
                # Générer réponses seulement
                responses = generate_teacher_responses(
                    teacher_model, teacher_tokenizer, batch_texts
                )
                
                # Tokeniser avec student tokenizer
                for src, resp in zip(batch_sources, responses):
                    if len(resp.strip()) < 20:
                        skipped += 1
                        continue
                    
                    # Créer séquence: prompt + response
                    full_text = build_distillation_prompt(src[:MAX_SOURCE_LENGTH * 3]) + resp
                    
                    # Tokeniser
                    encoding = student_tokenizer.encode(full_text)
                    token_ids = encoding.ids
                    
                    pad_id = student_tokenizer.token_to_id("<pad>")
                    if pad_id is None:
                        pad_id = 0
                    
                    if len(token_ids) < MAX_SEQUENCE_LENGTH + 1:
                        token_ids = token_ids + [pad_id] * (MAX_SEQUENCE_LENGTH + 1 - len(token_ids))
                    else:
                        token_ids = token_ids[:MAX_SEQUENCE_LENGTH + 1]
                    
                    input_ids = token_ids[:-1]
                    labels = token_ids[1:]
                    
                    # Pas de logits teacher (vocab mismatch)
                    example = {
                        "source": src,
                        "response": resp,
                        "input_ids": input_ids,
                        "labels": labels,
                        "teacher": TEACHER_MODEL,
                    }
                    
                    with DISTILLED_DATA_FILE.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(example, ensure_ascii=False) + "\n")
                    
                    count += 1
                    if count >= NUM_DISTILL_SAMPLES:
                        break
                        
            except Exception as e:
                print(f"\nErreur batch: {e}")
                skipped += len(batch_texts)
            
            batch_texts = []
            batch_sources = []
            
            if count % 100 == 0:
                print(f"  Progression: {count}/{NUM_DISTILL_SAMPLES}")
    
    print(f"\n✓ Génération terminée: {count} exemples, {skipped} skipped")
    print(f"  Fichier: {DISTILLED_DATA_FILE}")
    return count


# ============================================================
# ÉTAPE 2: TOKENISATION / PRÉPARATION TRAIN/VAL
# ============================================================

def split_and_tokenize_kd_data():
    """Split train/val et tokenise (déjà fait dans prepare_distillation_data)."""
    print("=" * 60)
    print("ÉTAPE 2: SPLIT TRAIN/VAL")
    print("=" * 60)
    
    import random
    random.seed(42)
    
    all_samples = []
    with DISTILLED_DATA_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_samples.append(json.loads(line))
    
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * 0.9)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")
    
    TOKENIZED_TRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOKENIZED_VAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    for samples, out_file in [(train_samples, TOKENIZED_TRAIN_FILE), (val_samples, TOKENIZED_VAL_FILE)]:
        with out_file.open("w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps({
                    "input_ids": s["input_ids"],
                    "labels": s["labels"],
                }, ensure_ascii=False) + "\n")
    
    print(f"  → {TOKENIZED_TRAIN_FILE}")
    print(f"  → {TOKENIZED_VAL_FILE}")


# ============================================================
# LOSS: Standard CE (pas de KD car vocab mismatch teacher/student)
# ============================================================

class DistillLoss(nn.Module):
    """Standard CE loss for data distillation."""
    
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
    
    def forward(self, student_logits, labels, teacher_logits=None):
        # teacher_logits ignored (vocab mismatch)
        ce = self.ce_loss(student_logits, labels)
        return ce, ce, torch.tensor(0.0, device=student_logits.device)


def create_student_model(vocab_size: int):
    """Crée le modèle BitNet student."""
    transformer = BitTransformer(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
    ).to(DEVICE)
    
    lm_head = LMHead(
        d_model=D_MODEL,
        vocab_size=vocab_size,
    ).to(DEVICE)
    
    return transformer, lm_head


def train_kd():
    print("=" * 60)
    print("ÉTAPE 3: ENTRAÎNEMENT BITNET AVEC KD")
    print("=" * 60)
    
    # Tokenizer
    tokenizer = load_student_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocabulaire student: {vocab_size}")
    
    # DataLoaders
    train_dataset = TokenizedKDDataset(TOKENIZED_TRAIN_FILE, MAX_SEQUENCE_LENGTH)
    val_dataset = TokenizedKDDataset(TOKENIZED_VAL_FILE, MAX_SEQUENCE_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=STUDENT_BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=STUDENT_BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Modèle
    transformer, lm_head = create_student_model(vocab_size)
    
    total_params = sum(p.numel() for p in transformer.parameters()) + sum(p.numel() for p in lm_head.parameters())
    print(f"Paramètres totaux: {total_params:,}")
    
    # Optimizer
    optimizer = AdamW(
        list(transformer.parameters()) + list(lm_head.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    
    # Loss
    criterion = DistillLoss()
    
    # Training loop
    best_val_loss = float("inf")
    
    for epoch in range(1, STUDENT_EPOCHS + 1):
        print(f"\n{'='*50}")
        print(f"EPOCH {epoch}/{STUDENT_EPOCHS}")
        print(f"{'='*50}")
        
        # Train
        transformer.train()
        lm_head.train()
        
        train_loss_sum = 0
        train_tokens = 0
        
        for batch in tqdm(train_loader, desc=f"Train E{epoch}"):
            input_ids = batch["input_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            
            optimizer.zero_grad()
            
            hidden = transformer(input_ids)
            logits = lm_head(hidden)
            
            # Reshape
            b, s, v = logits.shape
            logits = logits.reshape(b * s, v)
            labels = labels.reshape(b * s)
            
            loss, _, _ = criterion(logits, labels, None)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(transformer.parameters()) + list(lm_head.parameters()),
                GRADIENT_CLIP,
            )
            optimizer.step()
            
            tokens = b * s
            train_loss_sum += loss.item() * tokens
            train_tokens += tokens
        
        avg_train_loss = train_loss_sum / train_tokens
        
        # Validation
        transformer.eval()
        lm_head.eval()
        
        val_loss_sum = 0
        val_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Val E{epoch}"):
                input_ids = batch["input_ids"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)
                
                hidden = transformer(input_ids)
                logits = lm_head(hidden)
                
                b, s, v = logits.shape
                logits = logits.reshape(b * s, v)
                labels = labels.reshape(b * s)
                
                loss, _, _ = criterion(logits, labels, None)
                
                tokens = b * s
                val_loss_sum += loss.item() * tokens
                val_tokens += tokens
        
        avg_val_loss = val_loss_sum / val_tokens
        val_ppl = math.exp(avg_val_loss) if avg_val_loss < 20 else float("inf")
        train_ppl = math.exp(avg_train_loss) if avg_train_loss < 20 else float("inf")
        
        print(f"\nTrain Loss: {avg_train_loss:.4f} | PPL: {train_ppl:.2f}")
        print(f"Val   Loss: {avg_val_loss:.4f} | PPL: {val_ppl:.2f}")
        
        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "transformer_state_dict": transformer.state_dict(),
                "lm_head_state_dict": lm_head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "config": {
                    "vocab_size": vocab_size,
                    "d_model": D_MODEL,
                    "num_heads": NUM_HEADS,
                    "hidden_dim": HIDDEN_DIM,
                    "num_blocks": NUM_BLOCKS,
                    "max_seq_len": MAX_SEQUENCE_LENGTH,
                    "teacher_model": TEACHER_MODEL,
                },
            }, FINAL_CHECKPOINT)
            print(f"  ✓ Best model saved: {FINAL_CHECKPOINT}")
    
    print(f"\n✓ Entraînement terminé. Best val loss: {best_val_loss:.4f}")
    print(f"  Checkpoint: {FINAL_CHECKPOINT}")


# ============================================================
# MAIN
# ============================================================

def main():
    global TEACHER_MODEL, NUM_DISTILL_SAMPLES, STUDENT_EPOCHS
    
    import argparse
    parser = argparse.ArgumentParser(description="BitNet Data Distillation")
    parser.add_argument("--step", choices=["generate", "split", "train", "all"], default="all",
                        help="Étape à exécuter")
    parser.add_argument("--teacher", type=str, default=TEACHER_MODEL,
                        help="Modèle teacher HF")
    parser.add_argument("--samples", type=int, default=NUM_DISTILL_SAMPLES,
                        help="Nombre d'échantillons distillation")
    parser.add_argument("--epochs", type=int, default=STUDENT_EPOCHS,
                        help="Epochs entraînement student")
    args = parser.parse_args()
    
    TEACHER_MODEL = args.teacher
    NUM_DISTILL_SAMPLES = args.samples
    STUDENT_EPOCHS = args.epochs
    
    print("=" * 60)
    print("BITNET DATA DISTILLATION PIPELINE")
    print("=" * 60)
    print(f"Teacher: {TEACHER_MODEL}")
    print(f"Samples: {NUM_DISTILL_SAMPLES}")
    print(f"Epochs: {STUDENT_EPOCHS}")
    print(f"Device: {DEVICE}")
    print(f"Mode: Data Distillation (CE loss only, vocab mismatch)")
    
    if args.step in ["generate", "all"]:
        prepare_distillation_data()
    
    if args.step in ["split", "all"]:
        split_and_tokenize_kd_data()
    
    if args.step in ["train", "all"]:
        train_kd()
    
    print("\n✓ Pipeline terminé")


if __name__ == "__main__":
    main()