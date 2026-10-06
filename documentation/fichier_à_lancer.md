modification des paramètres: 
src/llm/config/parameter.py

Augmentation de documents:

# 1. Split des 20 documents
poetry run python scripts/split_dataset.py

# 2. Entraînement du tokenizer
On ne change plus, c'est le score pour prédire les futurs tokens sans pour les entraînements from scratch

poetry run python tokenizer/train_tokenizer.py

# 3. Vérification du vocabulaire
poetry run python -c "from llm.tokenizer.tokenizer import load_tokenizer; t=load_tokenizer(); print('Vocabulaire :', t.get_vocab_size())"

# 4. Tokenisation du dataset
poetry run python scripts/tokenize_dataset.py

# 5. Entraînement BitNet
Mais avant il faut changer le checkpoint de poursuite d'entrainement

poetry run python -m scripts.bitnet.train_bitnet
poetry run python -m scripts.gpt_classic.train

# 6. Test prompt
Mais avant il faut charger le dernier checkpoint et personnalisé le prompt scripts/bitnet/prompt.py

poetry run python -m scripts.bitnet.inference

-----------------------------------------------------------------
changement de nombre de block:

# Entraînement BitNet
poetry run python scripts/train_bitnet.py