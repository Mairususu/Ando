import pandas as pd
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing 
import matplotlib.pyplot as plt 

# 1. Lire les données
# J'ajoute 'encoding' car ce fichier CSV spécifique a souvent des problèmes d'encodage
try:
    df = pd.read_csv("spotify-2023.csv")
except UnicodeDecodeError:
    df = pd.read_csv("spotify-2023.csv", encoding='latin1')

# 2. CORRECTION : Sélectionner UNIQUEMENT les colonnes numériques pour la PCA
# C'est l'étape la plus importante.
df_numerical = df.select_dtypes(include=np.number)

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
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Components')
plt.title('Scree Plot')
# --- Fin du premier graphique ---


# --- Deuxième graphique (Scatter Plot) ---
plt.figure(2) # Ajout pour gérer plusieurs graphiques
pca_df = pd.DataFrame(pca_data, columns = labels)
plt.scatter(pca_df.PC1, pca_df.PC2)
plt.title('My PCA graph')
plt.xlabel(f'PC1 - {per_var[0]}%') # Utilisation de f-strings (plus moderne)
plt.ylabel(f'PC2 - {per_var[1]}%')


# Afficher les deux graphiques
plt.show()

# Les 10 première composantes expliquent 80% de la variance 