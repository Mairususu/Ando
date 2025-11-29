import pandas as pd
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing 
import matplotlib.pyplot as plt 

# 1. Lire les données
# J'ajoute 'encoding' car ce fichier CSV spécifique a souvent des problèmes d'encodage
try:
    df = pd.read_csv("DATASET.csv")
except UnicodeDecodeError:
    df = pd.read_csv("DATASET.csv", encoding='latin1')

# 2. CORRECTION : Sélectionner UNIQUEMENT les colonnes numériques pour la PCA
# C'est l'étape la plus importante.

df_variables = df[['artist_count', 'bpm', 'danceability_%', 'valence_%', 'energy_%', 'acousticness_%', 
                         'artist_dominance', 'title_word_count']]

df_numerical = df_variables.select_dtypes(include=np.number)

# 3. CORRECTION : Gérer les valeurs manquantes (NaN)
# Le 'scale' et la PCA ne peuvent pas gérer les NaN. Nous supprimons les lignes avec des NaN.
df_numerical = df_numerical.dropna()

# 4. Centrer et réduire les données
# On utilise maintenant notre DataFrame nettoyé 'df_numerical'
scaled_data = preprocessing.scale(df_numerical)

# --- Le reste de votre code est presque parfait ---

# Créer l'objet PCA : 
pca = PCA()
pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)

# Calculer la variance expliquée
per_var = np.round(pca.explained_variance_ratio_ * 100, decimals = 1)
print(per_var)

# Créer les labels pour le "scree plot" (PC1, PC2...)
# Note: le nombre de PC est maintenant basé sur 'df_numerical'
labels = [ 'PC' + str(x) for x in range(1, len(per_var)+1)]

# Créer le barplot (Scree Plot)
plt.figure(1) # Ajout pour gérer plusieurs graphiques
plt.bar(x = range(1, len(per_var) + 1), height = per_var, tick_label = labels)
plt.ylabel('Pourcentage de variance expliquée')
plt.xlabel('Composants principaux')
plt.title('Scree Plot')
# --- Fin du premier graphique ---


# --- Deuxième graphique (Scatter Plot) ---
plt.figure(2) # Ajout pour gérer plusieurs graphiques
pca_df = pd.DataFrame(pca_data, columns = labels)
plt.scatter(pca_df.PC1, pca_df.PC2)
plt.title('Graphique PCA')
plt.xlabel(f'PC1 - {per_var[0]}%') # Utilisation de f-strings (plus moderne)
plt.ylabel(f'PC2 - {per_var[1]}%')

# --- Troisième graphique (PCA components)
# --- À ajouter à la fin de votre script ---

# 1. Récupérer les noms des variables initiales
feature_names = df_numerical.columns

# 2. Créer un DataFrame lisible des "Poids" (Loadings)
# On transpose (.T) pour avoir les variables en lignes et les PC en colonnes
loadings_df = pd.DataFrame(
    pca.components_.T, 
    columns=labels, 
    index=feature_names
)
# 3. Afficher les 5 premières dimensions (puisque ce sont elles qui vous intéressent)
print(loadings_df.iloc[:, :5])
# --- Optionnel : Visualisation sous forme de carte de chaleur (Heatmap) ---
import seaborn as sns
plt.figure(figsize=(12, 8))
# On affiche les corrélations pour les 5 premières PC
sns.heatmap(loadings_df.iloc[:, :5], annot=True, cmap='coolwarm', center=0)
plt.title("Contribution des variables aux 5 premières dimensions")



# Afficher les deux graphiques
plt.show()

# Les 5 premières composantes expliquent 81.2% de la variance
# Les 4 premières expliquent 71.3%