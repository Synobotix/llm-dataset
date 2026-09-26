Récupération de donnée via C4 de hugging face


----------------------------------------------------------------
1-nettoyage:

C4 français 
    |
 Streaming
    |
 Nettoyage (Unicode, HTML, URLs, emails, espaces)
    |
 Filtres (qualitétexte ,trop court, trop de répétitions, trop de caractères non alphabétiques,spam évident)
    |
 Déduplication - Corpus propre

a-installation des dépendances:
poetry add datasets transformers torch tqdm

b-récupération d'une partie du dataset avec scripts/test_C4.py pour le teste
c-récupération de 1000 document avec scripts/prepare_C4.py
d-nettoie le HTML, les URLs, les emails, les espaces et on élimine les documents manifestement mauvais sans supprimer trop de contenu utile avec le scripts/clean_C4.py
d-vérifation du corpus après le nettoyage avec scripts/inspect_clean_C4.py
e-le dataset brut est conservé dans data/raw/c4_clean.jsonl
e-le dataset nettoyé est conservé dans data/processed/c4_clean.jsonl

