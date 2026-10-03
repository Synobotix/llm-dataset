import json
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from llm.tokenizer.tokenizer import load_tokenizer
from llm.config.parameters import BLOCK_SIZE, MAX_VOCAB_SIZE

# Configuration
# TEACHER_MODEL = "meta-llama/Llama-3.2-1B-Instruct"  # gated
# TEACHER_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # open, bon pour FR, ~1.5B params
# TEACHER_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # plus petit, plus rapide CPU
TEACHER_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"  # tout petit, très rapide CPU
# TEACHER_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"  # open, meilleur mais plus lourd
DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"  # ou "sample-100BT" pour plus
DATASET_LANG = "fr"
NUM_SAMPLES = 10  # test ultra-rapide
MAX_SOURCE_LENGTH = 1024
MAX_TARGET_LENGTH = 512
BATCH_SIZE = 4
OUTPUT_FILE = Path("data/distilled/hf_distilled.jsonl")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16


def load_teacher_model():
    print(f"Chargement du teacher: {TEACHER_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)
    print("Tokenizer chargé")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if DEVICE == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            TEACHER_MODEL,
            torch_dtype=DTYPE,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
    else:
        print("Chargement modèle sur CPU (peut prendre du temps)...")
        model = AutoModelForCausalLM.from_pretrained(
            TEACHER_MODEL,
            torch_dtype=DTYPE,
            low_cpu_mem_usage=True,
        ).to(DEVICE)
        print("Modèle chargé sur CPU")
    model.eval()
    return model, tokenizer


def build_distillation_prompt(source_text: str) -> str:
    return f"""Tu es un modèle enseignant. À partir du texte suivant, produis une réponse claire, cohérente et autonome qui résume les informations importantes. N'invente rien. Retourne uniquement la réponse.

Texte source:
{source_text}

Réponse:"""


def generate_teacher_response(model, tokenizer, prompt: str) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SOURCE_LENGTH,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_TARGET_LENGTH,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    return response.strip()


def generate_batch(model, tokenizer, prompts: list[str]) -> list[str]:
    """Génère en batch pour accélérer sur CPU."""
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SOURCE_LENGTH,
        padding=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_TARGET_LENGTH,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    responses = []
    for i, out in enumerate(outputs):
        response = tokenizer.decode(
            out[inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        responses.append(response.strip())
    return responses


def save_distilled_example(output_path: Path, source: str, prompt: str, response: str, teacher: str):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    example = {
        "source": source,
        "prompt": prompt,
        "response": response,
        "teacher": teacher,
    }
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(example, ensure_ascii=False) + "\n")


def main():
    print("=" * 60)
    print("DISTILLATION AVEC HUGGING FACE TEACHER")
    print("=" * 60)
    print(f"Teacher: {TEACHER_MODEL}")
    print(f"Dataset: {DATASET_NAME} ({DATASET_CONFIG}, lang={DATASET_LANG})")
    print(f"Samples: {NUM_SAMPLES}")
    print(f"Device: {DEVICE}")
    print(f"Output: {OUTPUT_FILE}")

    # 1. Charger le teacher
    model, tokenizer = load_teacher_model()

    # 2. Charger le dataset (streaming)
    print("\nChargement du dataset (streaming)...")
    ds = load_dataset(
        DATASET_NAME,
        name=DATASET_CONFIG,
        split="train",
        streaming=True,
    )

    # Filtrer par langue si disponible
    if DATASET_LANG:
        ds = ds.filter(lambda x: x.get("language", "") == DATASET_LANG or x.get("lang", "") == DATASET_LANG)

    # 3. Distillation
    print("\nDébut de la distillation...")
    count = 0
    skipped = 0
    batch_prompts = []
    batch_sources = []

    for idx, item in enumerate(tqdm(ds, total=NUM_SAMPLES * 10, desc="Scan dataset")):
        if count >= NUM_SAMPLES:
            break

        text = item.get("text", "").strip()
        if not text or len(text) < 100:
            skipped += 1
            continue
        
        if idx % 1000 == 0:
            print(f"\n  Scanned {idx} items, found {count} valid, skipped {skipped}")

        source_text = text[:MAX_SOURCE_LENGTH * 3]
        prompt = build_distillation_prompt(source_text)

        batch_prompts.append(prompt)
        batch_sources.append(source_text)

        if len(batch_prompts) >= BATCH_SIZE:
            try:
                responses = generate_batch(model, tokenizer, batch_prompts)

                for src, pr, resp in zip(batch_sources, batch_prompts, responses):
                    if len(resp.strip()) < 20:
                        skipped += 1
                        continue
                    save_distilled_example(
                        output_path=OUTPUT_FILE,
                        source=src,
                        prompt=pr,
                        response=resp,
                        teacher=TEACHER_MODEL,
                    )
                    count += 1
                    if count >= NUM_SAMPLES:
                        break

            except Exception as e:
                print(f"\nErreur batch: {e}")
                skipped += len(batch_prompts)

            batch_prompts = []
            batch_sources = []

            if count % 20 == 0:
                print(f"\nProgression: {count}/{NUM_SAMPLES} (skipped: {skipped})")

    # Traiter le dernier batch
    if batch_prompts and count < NUM_SAMPLES:
        try:
            responses = generate_batch(model, tokenizer, batch_prompts)
            for src, pr, resp in zip(batch_sources, batch_prompts, responses):
                if len(resp.strip()) < 20:
                    continue
                save_distilled_example(
                    output_path=OUTPUT_FILE,
                    source=src,
                    prompt=pr,
                    response=resp,
                    teacher=TEACHER_MODEL,
                )
                count += 1
        except Exception as e:
            print(f"\nErreur final batch: {e}")

    print(f"\n{'=' * 60}")
    print("DISTILLATION TERMINÉE")
    print(f"{'=' * 60}")
    print(f"Exemples générés: {count}")
    print(f"Skipped: {skipped}")
    print(f"Fichier: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()