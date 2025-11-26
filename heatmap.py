import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Préparation des données ---

# Remplacez par le nom de votre fichier
df = pd.read_csv("DATASET_ENRICHIE.csv")

# Liste des variables que vous avez sélectionnées
variables_explicatives = [
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

variable_cible = 'streams_per_month' 

cols_a_analyser = variables_explicatives + [variable_cible]
df_corr = df[cols_a_analyser].copy().dropna()

# --- 2. Calcul de la Matrice de Corrélation ---

corr_matrix = df_corr.corr()

# --- 3. Visualisation (Heatmap) ---

# J'ai légèrement augmenté la hauteur (10, 10) pour donner de l'air
plt.figure(figsize=(10, 10))

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

ax = sns.heatmap(
    corr_matrix, 
    mask=mask, 
    annot=True, 
    cmap='coolwarm', 
    fmt=".2f", 
    vmin=-1, 
    vmax=1,
    cbar_kws={"shrink": 0.8} # Réduit un peu la barre de légende latérale
)

plt.title("Matrice de Corrélation des Variables Sélectionnées", fontsize=16)

# --- LES CORRECTIONS SONT ICI ---
# 1. On incline les étiquettes de l'axe X pour qu'elles prennent moins de place verticale
plt.xticks(rotation=45, ha='right')

# 2. Cette commande recalcule automatiquement les marges pour que rien ne soit coupé
plt.tight_layout()

plt.show()

# --- 4. Analyse rapide ---
print("\nCorrélation avec la variable cible (streams_per_month) :")
print(corr_matrix[variable_cible].sort_values(ascending=False))