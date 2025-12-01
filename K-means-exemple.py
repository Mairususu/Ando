from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import numpy as np

import matplotlib.pyplot as plt

# Data frame
iris = load_iris()

X = iris.data[:, ]
Y = iris.target

#Séparation du jeu d'entrainement et de test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

#normalisation
scaler = StandardScaler().fit(x_train)

x_train_norm = scaler.transform(x_train)

x_test_norm = scaler.transform(x_test)

#détermination du nombre de classes optimal pour séparer les donées (valeur de K)
k_range = np.arange(2, 10)
distortion_value = []

for k in k_range:
  km = KMeans(k)

  km.fit(x_train_norm)

  distortion_value = np.append(distortion_value, km.inertia_) #calcul de la distortion / SSE (sum of sqare error)

#trace la distortion en fonction du nombre de classe
plt.plot(k_range, distortion_value)
plt.xlabel('Number of clusters')
plt.ylabel('Distortion score')
plt.show()
# On choisit K grâce à la méthode du coude
# Rq: si plusieurs coudes, prendre celui avec le moins de classes possible pour simplifier le modèle.

#Entrainement
# Initialization
km = KMeans(3)

# Training
km.fit(x_train_norm)


#Applications à nos données
# Prediction on train set
train_pred = km.predict(x_train_norm)

# Prediction on test set
test_pred = km.predict(x_test_norm)

#Score de silhouette censé rester stable entre X train et Xtest(nouvelles valeurs)
# Silhouette score for the training set
print(silhouette_score(x_train_norm, train_pred))

# Silhouette score for the test set
print(silhouette_score(x_test_norm, test_pred))



 
#visualisation
iris['feature_names']

plt.scatter(x_train_norm[:, 0], x_train_norm[:, 2], c=train_pred)
plt.xlabel('Longueur des sépales')
plt.ylabel('Longueur des pétales')
plt.show()

#Calcul des centroides

# Initialization
km = KMeans(3)

# Training
km.fit(x_train_norm)

# Prediction
train_pred = km.predict(x_train_norm)

plt.scatter(x_train_norm[:, 0], x_train_norm[:, 2], c=train_pred)
plt.scatter(km.cluster_centers_[0, 0], km.cluster_centers_[0, 2], color='violet', marker="X", s=200)
plt.scatter(km.cluster_centers_[1, 0], km.cluster_centers_[1, 2], color='green', marker="X", s=200)
plt.scatter(km.cluster_centers_[2, 0], km.cluster_centers_[2, 2], color='goldenrod', marker="X", s=200)
plt.show()