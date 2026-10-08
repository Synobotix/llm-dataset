


Accéder d'abord aller au .env avec documentation/chargement_de_la_variable_env.md

modification des paramètres: 
src/llm/config/parameter.py

From scratch:

# supprime tous les datas sur huggin face hub
poetry run python -m scripts.bitnet.reset_data.delete_all_data_json_jsonl_pt_hf

# supprime tous les datas sur le serveur d'entrainement
poetry run python -m scripts.bitnet.reset_data.delete_all_data_json_jsonl_pt_local

# Split des  documents sur huggin face
poetry run python -m scripts.nemotron.download_and_prepare

poetry run python -m scripts.split_dataset


# 2. Entraînement du tokenizer
# On ne change plus, c'est le score pour prédire les futurs tokens sans pour les entraînements from scratch

poetry run python -m tokenizer.train_tokenizer

# 3. Vérification du vocabulaire
poetry run python -c "from llm.tokenizer.tokenizer import load_tokenizer; t=load_tokenizer(); print('Vocabulaire :', t.get_vocab_size())"

# 4. Tokenisation du dataset
poetry run python -m scripts.tokenize_dataset

# 5. Entraînement BitNet
# Mais avant il faut changer le checkpoint de poursuite d'entrainement

poetry run python -m scripts.bitnet.train_bitnet
poetry run python -m scripts.gpt_classic.train

# 6. Test prompt
Mais avant il faut charger le dernier checkpoint et personnalisé le prompt scripts/bitnet/prompt.py

poetry run python -m scripts.bitnet.inference

-----------------------------------------------------------------
changement de nombre de block:

# Entraînement BitNet
poetry run python scripts/train_bitnet.py

-----------------------------------------------------------------------------
Sur colab:

session
colab new --gpu T4 --session trainer

entrainement from scratch, à noter il faut alors supprimer les checkpoints

export HF_TOKEN=$(grep "^HF_TOKEN=" .env | cut -d= -f2)

colab exec -s trainer \
    -f notebooks/training_colab_bitnet.ipynb \
    --env HF_TOKEN=$HF_TOKEN \
    --timeout 3600