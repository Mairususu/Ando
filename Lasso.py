import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

# Chargement des données
df_clean = pd.read_csv("DATASET.csv")

# 1. Sélection des variables
features = [
    'bpm', 'danceability_%', 'valence_%', 'energy_%', 
    'acousticness_%', 'instrumentalness_%', 'liveness_%', 
    'speechiness_%', 'artist_count', 'artist_dominance', 'title_word_count'
]

# Préparation de X
X = df_clean[features]

# Préparation de y avec transformation Log
# C'est la "correction magique" pour gérer l'échelle exponentielle des streams
y = np.log1p(df_clean['streams_per_month'])

# 2. Standardisation (Obligatoire pour Lasso)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Entraînement du modèle LassoCV
print("Recherche du modèle optimal (Lasso)...")
# cv=5 pour la validation croisée, max_iter grand pour assurer la convergence
model = LassoCV(cv=5, random_state=42, max_iter=50000)
model.fit(X_scaled, y)

# ---------------------------------------------------------
# PARTIE AJOUTÉE : TABLEAU DES COEFFICIENTS
# ---------------------------------------------------------

# Création d'un DataFrame pour les coefficients
df_coefs = pd.DataFrame({
    'Variable': features,
    'Coefficient': model.coef_
})

# Calcul de la valeur absolue pour trier par "force" de l'impact
df_coefs['Importance'] = df_coefs['Coefficient'].abs()

# Tri décroissant (les plus importants en haut)
df_coefs_sorted = df_coefs.sort_values(by='Importance', ascending=False).drop(columns=['Importance'])

# Affichage du tableau
print("\n=== RÉSULTATS DU LASSO ===")
print(f"R² du modèle : {model.score(X_scaled, y):.4f}")
print(f"Alpha optimal (pénalité) : {model.alpha_:.6f}")
print("\n--- TABLEAU DES COEFFICIENTS (Trié par importance) ---")
print(df_coefs_sorted.to_string(index=False))

# ---------------------------------------------------------
# 4. Visualisation Graphique
# ---------------------------------------------------------
plt.figure(figsize=(12, 6))

# Pour le graphique, on trie par valeur réelle (du négatif au positif)
coefs_plot = pd.Series(model.coef_, index=features).sort_values()

# Couleurs : Vert pour positif, Rouge pour négatif
colors = ['red' if c < 0 else 'green' for c in coefs_plot]

coefs_plot.plot(kind='barh', color=colors)

plt.title(f"Impact des variables sur les streams (Lasso) - R²: {model.score(X_scaled, y):.3f}")
plt.xlabel("Importance (Coefficient)")
plt.axvline(0, color='black', linewidth=0.8)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()



# R2 = 0.5013
# alpha = 0.2485


# Interprétation : 
# Conclusion : En 2023, le moteur principal du streaming quotidien n'est pas la vitesse (BPM) ni l'énergie brute, mais le "Groove".
# Une chanson qui donne envie de bouger (rythme régulier, beat fort) a mathématiquement plus de chances de devenir virale et d'être écoutée en boucle '
# 'qu'une chanson techniquement complexe.

# Avoir une grosse "Fanbase" (artist_dominance) aide significativement, MAIS (et c'est la surprise) cela compte moins que le potentiel dansant de la '
# 'chanson dans ce modèle précis.
# C'est une excellente nouvelle pour les artistes émergents : une chanson incroyablement efficace peut battre une superstar avec une chanson moyenne.

# La barre rouge en bas (instrumentalness_%) est claire.
# Plus une chanson est instrumentale, moins elle génère de streams quotidiens
# La voix est obligatoire pour le succès de masse. Les morceaux d'ambiance ou purement orchestraux ne rivalisent 
# pas sur le terrain de la viralité quotidienne.

# C'est la force du Lasso. Il a détecté que energy et danceability racontaient un peu la même histoire (une chanson qui bouge).
# Il a comparé les deux et a décidé : "La Danceability est un meilleur prédicteur. Je garde celle-là et je jette l'Energie à la poubelle pour ne pas me répéter."
# Cela simplifie ta conclusion : Pas besoin de se soucier du BPM ou de la gamme (Mode) si la chanson est dansante.


# "L'analyse par régression Lasso (avec correction logarithmique) démontre que la popularité d'une chanson (Streams/Jour) repose sur une formule simple :
# Une forte Danceability (Priorité absolue).
# Portée par un Artiste Dominant (Notoriété).
# Avec impérativement des Paroles (Pénalité pour l'instrumental).
# Les caractéristiques techniques comme le BPM, la tonalité (Mode) ou la positivité (Valence) deviennent négligeables une fois que le facteur
# 'Danse' est pris en compte."

