from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

#lire data
df = pd.read_csv("adapt2.csv", sep=',')
# supprimer la première colonne
data = df.iloc[:, 1:]

# Garder uniquement les colonnes numériques
data_numeric = data.select_dtypes(include=[np.number])

# Supprimer toutes les lignes contenant au moins un NaN
data_numeric = data_numeric.dropna()

print(data_numeric.head())      # affiche les 5 premières lignes
print(data_numeric.shape)       # affiche (nb_lignes, nb_colonnes)

#normalisation
scaler = StandardScaler().fit(data_numeric)
x_train_norm = scaler.transform(data_numeric)

#détermination du nombre de classes optimal pour séparer les donées (valeur de K)
k_range = np.arange(2, 10)
distortion_value = []

for k in k_range:
  km = KMeans(n_clusters=k, random_state=42)

  km.fit(x_train_norm)

  distortion_value.append(km.inertia_) #calcul de la distortion / SSE (sum of sqare error)

#trace la distortion en fonction du nombre de classe
plt.plot(k_range, distortion_value)
plt.xlabel('Number of clusters')
plt.ylabel('Distortion score')
plt.show()
# On choisit K grâce à la méthode du coude
# Rq: si plusieurs coudes, prendre celui avec le moins de classes possible pour simplifier le modèle.
# Version 1: K=4

K=4
