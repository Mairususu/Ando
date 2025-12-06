import pandas as pd
import numpy as np 
from datetime import datetime

print("Chargement du fichier...")
try:
    df = pd.read_csv("spotify-2023.csv")
except UnicodeDecodeError:
    df = pd.read_csv("spotify-2023.csv", encoding='latin1')

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

# B. Nombre de mots
df['title_word_count'] = df['track_name'].astype(str).apply(lambda x: len(x.split()))

# --- FIN AJOUT FEATURE ENGINEERING ---


# Drop the "_y" columns (the ones from the secondary file that are duplicates)
# Note: J'ai corrigé la virgule manquante après 'artist_lastfm'
cols_to_drop = ['instrumentalness_%', 'liveness_%', 'speechiness_%', 'key', 'mode', 'artist(s)_name', 'in_apple_playlists', 
                'in_apple_charts', 'in_deezer_playlists', 'in_deezer_charts', 'in_shazam_charts', 'popularity']

# Drop duplicates : 
df_clean = df.drop_duplicates(subset=['track_name'], keep='first')


# Drop columns created to calculate streams_per_month
cols_to_drop += ['days_since_release', 'release_date', 'months_since_release', 'streams_per_day', 'track_name'] 

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