"""
5_generate.py

Teste la génération du Student entraîné sur quelques prompts.

Usage : poetry run python scripts/test_local/5_generate.py
"""
import sys
sys.path.insert(0, "src")

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from llm.model.transformer import CausalTransformer

VOCAB_SIZE = 300
EMBEDDING_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 2
FF_HIDDEN_DIM = 128
MAX_SEQ_LEN = 32


def generate(model, tokenizer, prompt, max_new_tokens=20, temperature=0.8):
    device = next(model.parameters()).device
    ids = tokenizer.encode(prompt).ids
    ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            ids_cond = ids[:, -MAX_SEQ_LEN:]
            logits = model(ids_cond)
            next_logits = logits[0, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_id.unsqueeze(0)], dim=1)

    return tokenizer.decode(ids[0].tolist())


def main():
    tokenizer = Tokenizer.from_file("tokenizer/tokenizer_test.json")

    model = CausalTransformer(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ff_hidden_dim=FF_HIDDEN_DIM,
        max_sequence_length=MAX_SEQ_LEN,
    )
    model.load_state_dict(torch.load("checkpoints/student_test.pt", map_location="cpu"))

    prompts = [
        "Le Togo est",
        "Les ordinateurs modernes",
        "La photosynthèse est le processus",
        "L'université de Lomé",
    ]

    for prompt in prompts:
        result = generate(model, tokenizer, prompt, max_new_tokens=20)
        print(f"\nPrompt : {prompt}")
        print(f"Génération : {result}")


if __name__ == "__main__":
    main()
