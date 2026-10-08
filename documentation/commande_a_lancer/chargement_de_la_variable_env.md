A-Sur le local:
Étape 1 — Créer le fichier .env à la racine du projet
bash
cd ~/liste_projet/stage/llm

# Créer le fichier .env
cat > .env << 'EOF'
HF_TOKEN=hf_votre_token_ici
EOF
# ⚠️ Remplace hf_votre_token_ici par ton vrai token Hugging Face.

# 🔒 Étape 2 — Sécuriser le fichier
bash
# Permissions : lecture/écriture uniquement pour toi
chmod 600 .env

# Vérifier
ls -la .env
# → -rw------- 1 msspr msspr ... .env
🚫 Étape 3 — Vérifier que .env est ignoré par Git dans gitignore

