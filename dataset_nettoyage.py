import pandas as pd
import numpy as np 


print("Chargement du fichier...")
try:
    df = pd.read_csv("final_dataset.csv")
except UnicodeDecodeError:
    df = pd.read_csv("final_dataset.csv", encoding='latin1')


# Drop the "_y" columns (the ones from the secondary file that are duplicates)
cols_to_drop = ['key_y', 'mode_y', 'artist_name', 'in_apple_playlists', 'in_apple_charts', 'in_deezer_playlists', 
                'in_deezer_charts', 'in_shazam_charts', 'popularity', 'mbid', 'artist_mb', 'artist_lastfm'
                'country_mb', 'key', 'tags_mb', 'duration_ms', 'music_genre', 'key_x'] 

# Drop the "Scale Duplicates" (Keep your original % versions, drop the float versions)
# Check your data first to be sure, but usually you want to stick to one source.
cols_to_drop += ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness', 'valence', 'artist_mb', 'artist_lastfm', 
                 'country_lastfm', 'tags_lastfm', 'listeners_lastfm', 'scrobbles_lastfm', 'ambiguous_artist', 'instance_id', 'loudness', 'tempo', 'obtained_date'
                 ]

# Drop columns that actually don't exist in your dataframe but might cause errors
# (We use ignore_errors=True to be safe)
df_clean = df.drop(columns=cols_to_drop, errors='ignore')
df_clean= df_clean.drop(columns='country_mb')

# Verify the cleanup

df_clean.dropna(subset=['streams'], inplace=True)


# print(df_clean.isnull().sum())

# Check variables
print(list(df_clean.columns))

# Check shape
print(df_clean.shape)


df_clean.to_csv("DATASET.csv", index=False)

print("File saved successfully!")