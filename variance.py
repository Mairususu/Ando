import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 1. Chargement du fichier (remplacez par le nom de votre fichier)
df = pd.read_csv("DATASET.csv")

# 2. Sélection des variables numériques
# (J'ai repris la liste exacte visible sur votre image pour être précis)
colonnes_a_analyser = [
    'bpm', 
    'danceability_%',
    'valence_%', 
    'energy_%', 
    'acousticness_%', 
    'instrumentalness_%', 
    'liveness_%', 
    'speechiness_%', 
    'artist_count',
    'artist_dominance',
    'artist_target_encoded',
    'title_word_count'
]

# On filtre pour ne garder que les colonnes qui existent vraiment dans le fichier
cols_finales = [col for col in colonnes_a_analyser if col in df.columns]
df_subset = df[cols_finales].dropna() # On supprime les lignes avec des valeurs manquantes

# 3. Standardisation (Mise à l'échelle)
# C'est l'étape OBLIGATOIRE pour comparer la variance sur un même graphique
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_subset), columns=df_subset.columns)

# 4. Création du Graphique
plt.figure(figsize=(12, 10)) # Taille de l'image (Largeur, Hauteur)

# Paramétrage des points rouges (outliers)
points_rouges = {
    "markerfacecolor": "red", 
    "markeredgecolor": "red", 
    "marker": "o", 
    "markersize": 4
}

# Affichage du Boxplot
sns.boxplot(
    data=df_scaled, 
    orient="h",            # Orientation horizontale
    color="#ffcc5c",       # Couleur jaune/orange des boîtes
    flierprops=points_rouges # Application du style points rouges
)

# 5. Esthétique
plt.title("Comparaison de la Variance des Variables (Normalisées)", fontsize=16)
plt.xlabel("Distribution Standardisée", fontsize=12)
plt.ylabel("Variables", fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.5) # Grille légère verticale

plt.tight_layout() # Ajuste les marges automatiquement
plt.show()