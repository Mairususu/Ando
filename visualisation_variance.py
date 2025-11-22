import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Supposons que 'df' est ton DataFrame chargé (équivalent de 'data' en R)
df = pd.read_csv("DATASET.csv") 

# 1. Conversion en numérique (comme tes lignes as.numeric)
# On définit la liste des colonnes à convertir
cols_to_convert = [
    'artist_count', 'released_year', 'released_month', 'released_day',
    'in_spotify_playlists', 'in_spotify_charts', 'bpm', 
    'danceability_%', 'valence_%', 'energy_%', 'acousticness_%', # J'ai adapté les noms (R remplace souvent % par .)
    'instrumentalness_%', 'liveness_%', 'days_since_release'
    # Ajoute 'artist_dominance' si elle existe dans ton df
]

# On boucle pour convertir et gérer les erreurs (coerce transforme les erreurs en NaN)
for col in cols_to_convert:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 2. Sélection et Normalisation (Étape cruciale !)
# On ne garde que les numériques
df_numeric = df.select_dtypes(include=[np.number])

# On supprime les lignes avec des NaN (équivalent de na.omit())
df_numeric = df_numeric.dropna()

# On normalise (Centre et réduit : Moyenne = 0, Variance = 1)
# Équivalent de la fonction scale() de R
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_numeric), columns=df_numeric.columns)

# 3. Préparation pour le tri (Spécificité Python)
# ggplot le fait en direct avec reorder(), en Python on doit calculer l'ordre avant.
# On calcule la variance de chaque colonne pour trier le graphique
sorted_index = df_scaled.var().sort_values(ascending=True).index

# 4. Pivot (Équivalent de pivot_longer)
df_melted = df_scaled.melt(var_name='Variable', value_name='Valeur')

# 5. Création du Graphique
plt.figure(figsize=(12, 8))

# Équivalent de geom_boxplot + coord_flip
sns.boxplot(
    data=df_melted,
    x='Valeur',           # En X pour avoir l'effet coord_flip (horizontal)
    y='Variable',         # Les variables en Y
    order=sorted_index,   # On applique l'ordre calculé par variance
    color='orange',       # Couleur de remplissage
    fliersize=3,          # Taille des outliers
    flierprops={"marker": "o", "markerfacecolor": "red", "markeredgecolor": "red"} # Outliers rouges
)

# Esthétique (Theme minimal)
sns.set_style("whitegrid")
plt.title("Comparaison de la Variance des Variables (Normalisées)", fontsize=16)
plt.xlabel("Distribution Standardisée")
plt.ylabel("Variables")

# Affichage
plt.tight_layout()
plt.show()