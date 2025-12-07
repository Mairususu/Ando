import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

df = pd.read_csv('DATASET.csv')

# 1. Sélection des variables (basé sur ta slide "Variables Retenues")
features = [
    'bpm', 'danceability_%', 'valence_%', 'energy_%', 
    'acousticness_%', 'artist_count', 'artist_dominance', 
    'title_word_count'
]



X = df[features]

# 2. Standardisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. PCA
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# 4. t-SNE CORRIGÉ
# J'ai retiré 'n_iter' qui posait problème.
# J'ai mis learning_rate à 200 (valeur classique) au cas où 'auto' planterait aussi sur votre version.
tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate=200)

X_tsne = tsne.fit_transform(X_pca)

# 5. Visualisation
df_tsne = pd.DataFrame(data=X_tsne, columns=['TSNE1', 'TSNE2'])
df_tsne['Streams'] = df['streams_per_month']

plt.figure(figsize=(12, 8))
sns.scatterplot(
    x='TSNE1', y='TSNE2',
    data=df_tsne,
    hue='Streams',
    palette='viridis',
    alpha=0.7,
    s=60
)

plt.title('Visualisation t-SNE', fontsize=16)
plt.show()