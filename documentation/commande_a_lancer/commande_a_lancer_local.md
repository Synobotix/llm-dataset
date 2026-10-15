cd ~/liste_projet/stage/llm

ssh: 01ae87e23a127c49f0cadc3b7f094bf4

#Nb: le "cd" est le fichier à lancer pour accéder au projet dans votre PC

01ae87e23a127c49f0cadc3b7f094bf4
cat > .env << 'EOF'
HF_TOKEN=hf_votre_token_ici
EOF
azerty
01ae87e23a127c49f0cadc3b7f094bf4

ED25519 key fingerprint is SHA256:O9cfsdN1p9ZyPFBIj6bXqB2kxzgF91muX6mgMDPhXro.
# Charger le token dans le shell
export HF_TOKEN=$(grep "^HF_TOKEN=" .env | cut -d= -f2)

# Vérifier
echo "${HF_TOKEN:0:6}...${HF_TOKEN: -4}"
# → hf_abc...xyz1

# Puis lancer le script
poetry run python -m scripts.bitnet.reset_data.delete_all_data_json_jsonl_pt_hf