import pandas as pd
import matplotlib.pyplot as plt
import numpy as np # On a besoin de numpy pour le log
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler


df_clean = pd.read_csv("DATASET.csv")

# 1. Sélection des variables
features = [
    'bpm', 'danceability_%', 'valence_%', 'energy_%', 
    'acousticness_%', 'instrumentalness_%', 'liveness_%', 
    'speechiness_%', 'artist_count', 'artist_dominance'
]

# Préparation de X (avec mode_x transformé)
X_temp = df_clean[features + ['mode_x']]
X = pd.get_dummies(X_temp, columns=['mode_x'], drop_first=True)

# --- LA CORRECTION MAGIQUE ---
# Au lieu de prédire 1 000 000 vs 10, on prédit log(1 000 000) vs log(10)
# np.log1p fait log(1 + x) pour gérer les zéros proprement
y_original = df_clean['streams_per_day']
y = np.log1p(y_original) 

# 2. Standardisation (Toujours obligatoire)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. LassoCV (On réessaie)
# J'ai réduit le cv à 5 et augmenté les itérations pour aider le modèle
print("Recherche du modèle optimal...")
model = LassoCV(cv=5, random_state=42, max_iter=50000)
model.fit(X_scaled, y)

# 4. Visualisation
coefs = pd.Series(model.coef_, index=X.columns).sort_values()

plt.figure(figsize=(12, 6))
colors = ['red' if c < 0 else 'green' for c in coefs]
coefs.plot(kind='barh', color=colors)

plt.title("Impact sur le LOG des Streams (Correction d'échelle)")
plt.xlabel("Importance (Coefficient)")
plt.axvline(0, color='black', linewidth=0.8)
plt.grid(axis='x', alpha=0.3)
plt.show()

print(f"Nouveau R² : {model.score(X_scaled, y):.4f}")
print(f"Alpha choisi : {model.alpha_:.4f}") # Il devrait être beaucoup plus petit !



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

