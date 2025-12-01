from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

#lire data
df = pd.read_csv("DATASET.csv", sep=',')

# Colonnes à garder
colonnes_a_garder = [
    "valence_%",
    "streams_per_month",
    "energy_%",
    "danceability_%",
    "acousticness_%",
    "bpm",
    "artist_count",
    "title_word_count",
    "artist_dominance"
]

# Garder seulement ces colonnes
data = df[colonnes_a_garder]

# Garder uniquement les colonnes numériques
data_numeric = data.select_dtypes(include=[np.number])

# Supprimer toutes les lignes contenant au moins un NaN
data_numeric = data_numeric.dropna()

print(data_numeric.head())      # affiche les 5 premières lignes
print(data_numeric.shape)       # affiche (nb_lignes, nb_colonnes)

#normalisation
scaler = StandardScaler().fit(data_numeric)
x_norm = scaler.transform(data_numeric)

# Séparation en jeu d'entraînement / test
X_train, X_test = train_test_split(x_norm,
    test_size=0.2,      # 20% pour le test, modifiable
    random_state=42,
    shuffle=True
)

print("Taille train :", X_train.shape)
print("Taille test  :", X_test.shape)

#détermination du nombre de classes optimal pour séparer les donées (valeur de K)
k_range = np.arange(2, 10)
distortion_value = []

for k in k_range:
  km = KMeans(n_clusters=k, random_state=42)

  km.fit(x_norm)

  distortion_value.append(km.inertia_) #calcul de la distortion / SSE (sum of sqare error)

#trace la distortion en fonction du nombre de classe
plt.plot(k_range, distortion_value)
plt.xlabel('Number of clusters')
plt.ylabel('Distortion score')
plt.show()
# On choisit K grâce à la méthode du coude
# Rq: si plusieurs coudes, prendre celui avec le moins de classes possible pour simplifier le modèle.
# Version 1: K=4

K=6

#Entrainement
# Initialization
km = KMeans(K)

# Training
km.fit(X_train)


#Applications à nos données
# Prediction on train set
train_pred = km.predict(X_train)

# Prediction on test set
test_pred = km.predict(X_test)

#Score de silhouette censé rester stable entre X train et Xtest(nouvelles valeurs)
# Silhouette score for the training set
print(silhouette_score(X_train, train_pred))

# Silhouette score for the test set
print(silhouette_score(X_test, test_pred))



#Centroïdes
for i in range (K):
 print("Centroïde du cluster", i )
 print(km.cluster_centers_[i])

 #Ajouter la variable popularité dans le clustering pour l'intérprétation
 
 """Résultats: 

Centroïde du cluster 0
[-0.94939317 -0.12257811 -0.68754308 -0.74639046  0.50427281  0.06219014
 -0.53147957 -0.22950689  2.81764815]
-> Chansons tristes, calmes, acoustiques, artiste très dominant, peu populaires
Popularité : légèrement en dessous de la moyenne

Centroïde du cluster 1
[ 0.53919987  1.00589624  0.27961943  0.314011   -0.20291819  0.22366733
  1.67370286  1.21141682 -0.64941992]
-> Chansons joyeuses et populaires, titres longs, artiste moins dominant
Popularité : très élevée → hits populaires

Centroïde du cluster 2
[-0.5886433  -0.21267508 -1.28553148 -0.89388046  1.60461074 -0.0597724
 -0.36234205  0.31238438 -0.22155694]
 -> Chansons lentes, tristes, acoustiques, peu populaires.
 Popularité : légèrement faible

Centroïde du cluster 3
[-0.41094173 -0.340329    0.57214228 -0.78817102 -0.62697174  0.66101078
 -0.29604804 -0.01426162 -0.23877966]
 Chansons un peu tristes mais énergiques et rapides, peu populaires.
 Popularité : faible

Centroïde du cluster 4
[-0.33157573  0.25151877 -0.42150817  0.72367146 -0.31236857  0.11693633
 -0.5226228  -0.50257805  1.30154522]
 Chansons dansantes, un peu tristes, artiste dominant, populaires
Popularité : légèrement au-dessus de la moyenne

Centroïde du cluster 5
[ 0.57353961 -0.14098694  0.41628956  0.6221687  -0.25564909 -0.45808532
 -0.01531247 -0.35601069 -0.35583299]
 Chansons joyeuses, dansantes, énergiques, peu populaires.
 Popularité : légèrement faible



 Silhouette score : faible (0.145 / 0.113), donc clusters pas très séparés, mais permettent quand même de repérer des tendances
"""