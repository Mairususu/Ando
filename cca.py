import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler

# Chargement et Création de la Variable (Rappel) ---
try:
    df = pd.read_csv("spotify-2023.csv") ##n'est pas l'original mais le traité
except UnicodeDecodeError:
    df = pd.read_csv("spotify-2023.csv", encoding='latin1')

# Nettoyage de base
df['streams'] = pd.to_numeric(df['streams'], errors='coerce')

# Création de streams_per_day
df['release_date'] = pd.to_datetime(df[['released_year', 'released_month', 'released_day']].rename(columns={'released_year': 'year', 'released_month': 'month', 'released_day': 'day'}))
date_ref = df['release_date'].max()
df['days_since_release'] = (date_ref - df['release_date']).dt.days + 1
df['streams_per_day'] = df['streams'] / df['days_since_release']
df['streams_per_month'] = df['streams_per_day']/30 #environ 30 jours dans un mois.

# Sélection des Variables 

# X : Caractéristiques Musicales (On enlève l'année)
X_cols = [
    'bpm', 'danceability_%', 'valence_%', 'energy_%',
    'acousticness_%', 'artist_count','artist_dominance','title_word_count'
]

# Y : Indicateurs de Succès (Avec la nouvelle variable)
Y_cols = [
    'streams_per_month',       #  Nouvelle variable au lieu de stream 
    'in_spotify_playlists', 
    'in_spotify_charts', 
    #'in_apple_charts',
    #'in_apple_playlists', 
    #'in_deezer_playlists', 
    #'in_deezer_charts', 
    #'in_shazam_charts'
]#JE LES AI ENLEVE CAR ELLES NE SONT PAS DANS NOTRE DATASET.

# Nettoyage Final 
# On nettoie tout (virgules, strings) dans les colonnes sélectionnées
subset_cols = X_cols + Y_cols
for col in subset_cols:
    if df[col].dtype == 'object':
        df[col] = df[col].astype(str).str.replace(',', '')
    df[col] = pd.to_numeric(df[col], errors='coerce')

df_cca = df[subset_cols].dropna()

# Fit CCA 
scaler_x = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_x.fit_transform(df_cca[X_cols]) # X est bien scaled
Y_scaled = scaler_y.fit_transform(df_cca[Y_cols])

cca = CCA(n_components=1)
cca.fit(X_scaled, Y_scaled)
X_c, Y_c = cca.transform(X_scaled, Y_scaled)

plt.figure(figsize=(10, 8))

# Plot the songs as points
# We color them by 'streams_per_day' to see if popular songs cluster
sns.scatterplot(
    x=X_c[:, 0], 
    y=Y_c[:, 0], 
    hue=np.log1p(df_cca['streams_per_month']), # Log scale for better color gradient
    palette='viridis',
    alpha=0.7
)

# Add a diagonal line (Perfect correlation would lie on this line)

### essai pour afficher X et Y début
print("Composante canonique X :", X_c.ravel())
print("Composante canonique Y :", Y_c.ravel())

print("coefficients valeurs de x : ", cca.x_weights_)
print("coefficients valeurs de y : ", cca.y_weights_)




X_loadings = np.corrcoef(X_scaled.T, X_c[:,0])[len(X_cols):, :len(X_cols)]
Y_loadings = np.corrcoef(Y_scaled.T, Y_c[:,0])[len(Y_cols):, :len(Y_cols)]

# print(pd.Series(X_loadings[0,0], index=X_cols))
# print(pd.Series(Y_loadings[0,0], index=Y_cols))



### essai pour afficher X et Y fini

min_val = min(X_c.min(), Y_c.min())
max_val = max(X_c.max(), Y_c.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Correlation')

plt.xlabel('Musical Variate (X Canonical Score)')
plt.ylabel('Popularity Variate (Y Canonical Score)')
plt.title(f'CCA Latent Space (Correlation: {np.corrcoef(X_c[:,0], Y_c[:,0])[0,1]:.3f})')
plt.legend(title='Log(Streams/Day)')
plt.grid(True, alpha=0.3)
plt.show()






# # Corrélation
# print(f"Nouvelle Corrélation Canonique : {np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1]:.4f}")

# # Visualisation 
# x_weights = pd.Series(cca.x_weights_[:, 0], index=X_cols).sort_values()
# y_weights = pd.Series(cca.y_weights_[:, 0], index=Y_cols).sort_values()

# fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# x_weights.plot(kind='barh', ax=axes[0], color='skyblue')
# axes[0].set_title("Drivers Musicaux (X)")
# axes[0].axvline(0, color='black', linewidth=0.8)

# y_weights.plot(kind='barh', ax=axes[1], color='salmon')
# axes[1].set_title("Drivers de Popularité (Y)")
# axes[1].axvline(0, color='black', linewidth=0.8)

# plt.tight_layout()
# plt.show()

# Interpretation 

# et bien et bien : 

