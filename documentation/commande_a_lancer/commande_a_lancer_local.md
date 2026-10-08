cd ~/liste_projet/stage/llm

#Nb: le "cd" est le fichier à lancer pour accéder au projet dans votre PC

# Charger le token dans le shell
export HF_TOKEN=$(grep "^HF_TOKEN=" .env | cut -d= -f2)

# Vérifier
echo "${HF_TOKEN:0:6}...${HF_TOKEN: -4}"
# → hf_abc...xyz1

# Puis lancer le script
poetry run python -m scripts.bitnet.reset_data.delete_all_data_json_jsonl_pt_hf