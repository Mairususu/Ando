# Principal component analysis (PCA) is a dimensionality reduction technique that transforms 
# a data set into a set of orthogonal components — called principal components — which capture 
# the maximum variance in the data


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing 
import matplotlib.pyplot as plt 

# --- 1. Load Data ---
try:
    df = pd.read_csv("DATASET.csv")
except UnicodeDecodeError:
    df = pd.read_csv("DATASET.csv", encoding='latin1')

# --- 2. Select Your 12 Numeric Variables ---
# numeric_cols = [
#     'artist_count',
#     'released_year',
#     'released_month',
#     'released_day',
#     'bpm',
#     'danceability_%',
#     'valence_%',
#     'energy_%',
#     'acousticness_%',
#     'instrumentalness_%',
#     'liveness_%',
#     'speechiness_%',
#     'artist_dominance',
#     'streams_per_month'
# ] 

numeric_cols = [
    'valence_%',
    'energy_%',
    'danceability_%',
    'acousticness_%', 
    'bpm', 
    'artist_count',
    'streams_per_month'
]

# Create the DataFrame using only these columns
df_explanatory = df[numeric_cols].copy()

# --- 3. Clean Data ---
# CRITICAL: Drop any rows with missing values (NaN) in these columns
df_explanatory = df_explanatory.dropna()

print(f"Shape of data for PCA: {df_explanatory.shape}")

# --- 4. Scale and Run PCA ---
# Center and scale the data
scaled_data = preprocessing.scale(df_explanatory)

# Create PCA object
pca = PCA()
pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)

# Calculate variance
per_var = np.round(pca.explained_variance_ratio_ * 100, decimals = 1)

print(f"Explained variance by component: {per_var}")

# --- 5. Plot Scree Plot ---
# Create labels (will be PC1, PC2... up to PC12)
labels = [ 'PC' + str(x) for x in range(1, len(per_var)+1)]

plt.figure(1)
plt.bar(x = range(1, len(per_var) + 1), height = per_var, tick_label = labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Components')
plt.title('Scree Plot (Numeric Features)')

# --- 6. Plot PCA Scatter Plot ---
plt.figure(2)
# Create the PCA DataFrame
pca_df = pd.DataFrame(pca_data, columns = labels, index=df_explanatory.index)

plt.scatter(pca_df.PC1, pca_df.PC2)
plt.title('PCA Graph (Numeric Features)')
plt.xlabel(f'PC1 - {per_var[0]}%')
plt.ylabel(f'PC2 - {per_var[1]}%')

# Show both plots
plt.show()

# A 8 composantes on a 80.1%
# Check coeff PC1
# Contribution de var original / chaque var
#Etudier distribution 