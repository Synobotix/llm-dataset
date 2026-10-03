# Référence : Tokenizer

## Emplacement

`tokenizer/tokenizer.json` (généré par `tokenizer/train_tokenizer.py`)

## Chargement

Classique : `llm.tokenizer.tokenizer.load_tokenizer()` ou via `tokenizers.Tokenizer.from_file()`
BitNet : `llm.inference.generate_bitnet.load_tokenizer()` / `tokenizers.Tokenizer.from_file(TOKENIZER_FILE)`

## Utilisation

- `tokenizer.encode(text)` → `Encoding(ids=[...], tokens=[...])`
- `tokenizer.decode(ids)` → texte
- `vocab_size` via `tokenizer.get_vocab_size()` ou config

## Entraînement tokenizer

`tokenizer/train_tokenizer.py` : entraîne un tokenizer (BPE/WordPiece selon config `tokenizers`) sur corpus.
