import pandas as pd
import numpy as np

# 1. Chargement
try:
    df = pd.read_csv("DATASET.csv")
    print("Fichier chargé.")
except FileNotFoundError:
    print("Erreur : Fichier introuvable.")
    exit()

# Nettoyage : conversion des streams en nombres
df['streams'] = pd.to_numeric(df['streams'], errors='coerce')

# --- PARTIE 1 : Variables Textuelles ---

# B. Nombre de mots
df['title_word_count'] = df['track_name'].astype(str).apply(lambda x: len(x.split()))

# C. Indicateur de Collaboration / Remix
keywords = ['feat', 'ft.', 'remix', 'with']
df['is_collab_track'] = df['track_name'].astype(str).str.lower().apply(
    lambda x: 1 if any(word in x for word in keywords) else 0
)

# --- PARTIE 2 : Target Encoding sur l'Artiste ---

col_artiste = 'artist(s)_name' 

# ATTENTION : Si tu n'as pas encore créé 'streams_per_month' dans CE script, 
# on utilise 'streams' par défaut pour éviter un crash.
if 'streams_per_month' in df.columns:
    col_cible = 'streams_per_month'
else:
    print("Attention: 'streams_per_month' n'existe pas, on utilise 'streams' pour l'encoding.")
    col_cible = 'streams'

# Calcul de la moyenne (Target Encoding)
artist_means = df.groupby(col_artiste)[col_cible].transform("mean")
df['artist_target_encoded'] = artist_means

# --- ETAPE 3 : SAUVEGARDE (C'est ce qu'il manquait) ---

nom_fichier_sortie = "DATASET_ENRICHIE.csv"
df.to_csv(nom_fichier_sortie, index=False)

print(f"\nsuccès ! Le nouveau dataset avec les variables ajoutées a été sauvegardé sous : {nom_fichier_sortie}")
print(f"Nombre de colonnes : {df.shape[1]}")