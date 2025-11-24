import pandas as pd
import numpy as np 
from datetime import datetime

print("Chargement du fichier...")
try:
    df = pd.read_csv("final_dataset.csv")
except UnicodeDecodeError:
    df = pd.read_csv("final_dataset.csv", encoding='latin1')

# --- DEBUT AJOUT FEATURE ENGINEERING ---
print("Calcul de la feature 'streams_per_month'...")

# 1. Conversion de streams en numérique (gestion des erreurs)
df['streams'] = pd.to_numeric(df['streams'], errors='coerce')

# 2. Création de la date de sortie
df['release_date'] = pd.to_datetime(dict(year=df['released_year'], 
                                         month=df['released_month'], 
                                         day=df['released_day']))

# 3. Calcul de l'ancienneté (par rapport à la date la plus récente du dataset)
reference_date = df['release_date'].max()
df['days_since_release'] = (reference_date - df['release_date']).dt.days

# 4. Conversion en mois (avec sécurité anti-division par zéro)
df['months_since_release'] = df['days_since_release'] / 30.44
df['months_since_release'] = df['months_since_release'].clip(lower=1) 

# 5. Création de la variable finale
df['streams_per_month'] = df['streams'] / df['months_since_release']


# On compte combien de chansons chaque artiste a placé dans le Top
artist_counts = df['artist(s)_name'].value_counts()

# On attribue ce score à chaque ligne
df['artist_dominance'] = df['artist(s)_name'].map(artist_counts)

# --- FIN AJOUT FEATURE ENGINEERING ---


# Drop the "_y" columns (the ones from the secondary file that are duplicates)
# Note: J'ai corrigé la virgule manquante après 'artist_lastfm'
cols_to_drop = ['key_y', 'mode_y', 'artist_name', 'in_apple_playlists', 'in_apple_charts', 'in_deezer_playlists', 
                'in_deezer_charts', 'in_shazam_charts', 'popularity', 'mbid', 'artist_mb', 'artist_lastfm',
                'country_mb', 'key', 'tags_mb', 'duration_ms', 'music_genre', 'key_x'] 

# Drop the "Scale Duplicates" (Keep your original % versions, drop the float versions)
cols_to_drop += ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness', 'valence', 'artist_mb', 'artist_lastfm', 
                 'country_lastfm', 'tags_lastfm', 'listeners_lastfm', 'scrobbles_lastfm', 'ambiguous_artist', 'instance_id', 'loudness', 'tempo', 'obtained_date'
                 ]

# Drop columns created to calculate streams_per_month
cols_to_drop += ['days_since_release', 'release_date', 'months_since_release', 'streams_per_day'] 

# Drop columns that actually don't exist in your dataframe but might cause errors
df_clean = df.drop(columns=cols_to_drop, errors='ignore')

# Verification supplémentaire 
df_clean = df_clean.drop(columns='country_mb', errors='ignore')

# Verify the cleanup
# On s'assure que streams ET streams_per_month sont valides
df_clean.dropna(subset=['streams', 'streams_per_month'], inplace=True)

# Check variables
print("\nColonnes finales :")
print(list(df_clean.columns))

# Check shape
print("\nTaille du dataset final :")
print(df_clean.shape)

df_clean.to_csv("DATASET.csv", index=False)

print("File saved successfully as 'DATASET.csv'!")