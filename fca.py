import numpy as np
import skfda
from skfda.preprocessing.dim_reduction.projection import FPCA
from sklearn.cross_decomposition import CCA
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# EXEMPLE DE DONNÉES
# ------------------------------------------------------------
# Deux groupes de variables fonctionnelles :
# X : 50 courbes simulées
# Y : 50 autres courbes correspondant à X

n_samples = 50
grid = np.linspace(0, 1, 200)

# Exemple de signaux fonctionnels
X_data = np.array([np.sin(2*np.pi*grid) + 0.1*np.random.randn(len(grid)) for _ in range(n_samples)])
Y_data = np.array([np.cos(2*np.pi*grid) + 0.1*np.random.randn(len(grid)) for _ in range(n_samples)])

fd_X = skfda.FDataGrid(data_matrix=X_data, grid_points=grid)
fd_Y = skfda.FDataGrid(data_matrix=Y_data, grid_points=grid)

# ------------------------------------------------------------
# 1. FPCA – réduction fonctionnelle
# ------------------------------------------------------------
fpca = FPCA(n_components=5)

X_scores = fpca.fit_transform(fd_X)
Y_scores = fpca.fit_transform(fd_Y)

# ------------------------------------------------------------
# 2. FCA via CCA sur les scores fonctionnels
# ------------------------------------------------------------
cca = CCA(n_components=2)
cca.fit(X_scores, Y_scores)

X_c, Y_c = cca.transform(X_scores, Y_scores)

print("Corrélations canoniques :", np.corrcoef(X_c[:,0], Y_c[:,0])[0,1])

# ------------------------------------------------------------
# 3. Visualisation
# ------------------------------------------------------------
plt.scatter(X_c[:,0], Y_c[:,0])
plt.xlabel("Composante canonique X1")
plt.ylabel("Composante canonique Y1")
plt.title("Functional Correspondence Analysis (FCA)")
plt.grid(True)
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# ------------------------------------------------------------
# 1. Chargement du dataset
# ------------------------------------------------------------
df = pd.read_csv("datset.csv")

# Sélection uniquement des colonnes numériques
num_df = df.select_dtypes(include=[np.number])

# ------------------------------------------------------------
# 2. Définition des groupes de variables (modifiable)
# ------------------------------------------------------------
group_X_cols = ["streams", "in_spotify_playlists", "in_spotify_charts", "streams_per_month"]
group_Y_cols = ["danceability_%", "valence_%", "energy_%", "acousticness_%",
                "instrumentalness_%", "liveness_%", "speechiness_%"]

X = num_df[group_X_cols].to_numpy()
Y = num_df[group_Y_cols].to_numpy()

# ------------------------------------------------------------
# 3. Standardisation
# ------------------------------------------------------------
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
Y = (Y - np.mean(Y, axis=0)) / np.std(Y, axis=0)

# ------------------------------------------------------------
# 4. CCA (FCA) MANUELLE
# ------------------------------------------------------------
# Matrices de covariance
Sxx = np.cov(X, rowvar=False)
Syy = np.cov(Y, rowvar=False)
Sxy = np.cov(X, Y, rowvar=False)[:X.shape[1], X.shape[1]:]
Syx = Sxy.T

# Matrices inverses
Sxx_inv = np.linalg.inv(Sxx)
Syy_inv = np.linalg.inv(Syy)

# Calcul des valeurs propres
eigvals, eigvecs = np.linalg.eig(Sxx_inv @ Sxy @ Syy_inv @ Syx)

# Tri
idx = eigvals.argsort()[::-1]
eigvals = np.real(eigvals[idx])
eigvecs = np.real(eigvecs[:, idx])

# vecteurs canoniques
A = eigvecs[:, 0]  # composante canonique pour X
B = np.linalg.inv(Syy) @ Syx @ A / np.sqrt(eigvals[0])  # composante canonique Y

# Projections canoniques
X_c = X @ A
Y_c = Y @ B

corr = np.corrcoef(X_c, Y_c)[0, 1]
print("Corrélation canonique principale :", corr)

# ------------------------------------------------------------
# 5. VISUALISATION FCA
# ------------------------------------------------------------
plt.scatter(X_c, Y_c)
plt.xlabel("Composante canonique X1")
plt.ylabel("Composante canonique Y1")
plt.title("FCA – Popularité vs Caractéristiques audio")
plt.grid(True)
plt.show()

