# -*- coding: utf-8 -*-
"""
Script d'Analyse Factorielle des Correspondances (AFC) - 2 Dimensions

Objectif : Analyser la relation entre la Tonalité (key) et le Niveau de Popularité (streams catégorisés)
"""

import pandas as pd
import numpy as np
from prince import CA
import matplotlib.pyplot as plt
import seaborn as sns

FILE_NAME = "real_dataset.csv"

def run_2d_correspondence_analysis():
    """
    Charge, prépare les données pour 2 dimensions, exécute l'AFC
    et affiche le Biplot 2D.
    """
    print(f"--- 1. Chargement et Préparation des Données ({FILE_NAME}) ---")

    try:
        df = pd.read_csv(FILE_NAME)
    except FileNotFoundError:
        print(f"ERREUR : Le fichier '{FILE_NAME}' n'a pas été trouvé.")
        return

    # 1. Nettoyage des données pour les variables d'intérêt
    df_ca = df[['key', 'streams']].copy()
    df_ca.replace('', np.nan, inplace=True)
    df_ca.dropna(subset=['key', 'streams'], inplace=True)
    
    # Assurez-vous que 'streams' est numérique (peut être une chaîne d'abord)
    df_ca['streams'] = pd.to_numeric(df_ca['streams'], errors='coerce')
    df_ca.dropna(subset=['streams'], inplace=True)
    
    # Rétablir les 'streams' en entier pour les seuils
    df_ca['streams'] = df_ca['streams'].astype(int)

    print(f"Nombre d'observations après nettoyage (key et streams non nuls) : {len(df_ca)}")

    # 2. Discrétisation de la variable 'streams' (Création de 3 catégories)
    
    # Utilisation des quantiles (25%, 75%) pour définir les seuils
    low_quantile = df_ca['streams'].quantile(0.33)
    high_quantile = df_ca['streams'].quantile(0.66)
    
    bins = [df_ca['streams'].min() - 1, low_quantile, high_quantile, df_ca['streams'].max()]
    labels = ['Faible_Streams', 'Moyen_Streams', 'Élevé_Streams']
    
    df_ca['streams_category'] = pd.cut(
        df_ca['streams'], 
        bins=bins, 
        labels=labels, 
        include_lowest=True, 
        duplicates='drop'
    )
    
    # Si le découpage n'a pas créé au moins 3 catégories, arrêter (très rare)
    if df_ca['streams_category'].nunique() < 2:
        print("\nERREUR : Le découpage de 'streams' n'a pas permis de créer assez de catégories pour l'AFC (moins de 2 catégories).")
        return


    # 3. Création du Tableau de Contingence
    # Lignes (key) x Colonnes (streams_category)
    contingency_table = pd.crosstab(df_ca['key'], df_ca['streams_category'])

    print("\n--- 2. Tableau de Contingence (Tonalité vs Catégorie de Streams) ---")
    print(contingency_table)
    
    # Le nombre de dimensions maximales est 2
    N_comp_max = min(contingency_table.shape[0] - 1, contingency_table.shape[1] - 1)
    
    if N_comp_max < 2:
        print(f"ALERTE: Le tableau n'a que {N_comp_max} dimensions possibles. On continue avec {N_comp_max} composantes.")
        N_components = N_comp_max
    else:
        N_components = 2

    # 4. Exécution de l'Analyse Factorielle des Correspondances
    print(f"\n--- 3. Exécution de l'AFC avec {N_components} Dimensions ---")
    
    ca = CA(n_components=N_components, n_iter=10, engine='sklearn')
    ca = ca.fit(contingency_table)

    # 5. Analyse de la Variance Expliquée
    print("\n--- 4. Analyse de la Variance Expliquée ---")
    
    eigenvalues = ca.eigenvalues_
    total_inertia = eigenvalues.sum()
    explained_variance_ratio = eigenvalues / total_inertia
    
    print("Variance Expliquée par les Composantes :")
    for i, ratio in enumerate(explained_variance_ratio):
        if i < N_components:
            print(f"Dimension {i+1}: {ratio:.4f} ({ratio*100:.2f}%)")

    if N_components == 2:
        cumulative_variance = np.cumsum(explained_variance_ratio)
        print(f"\nVariance Cumulée des 2 premières dimensions : {cumulative_variance[1]:.4f} ({cumulative_variance[1]*100:.2f}%)")
    
    # 6. Visualisation du Biplot (2D)
    print("\n--- 5. Visualisation du Biplot (2D) ---")
    
    row_coords = ca.row_coordinates(contingency_table)
    col_coords = ca.column_coordinates(contingency_table)
    
    # Extraction des 2 dimensions
    x_coords_rows = row_coords.iloc[:, 0]
    y_coords_rows = row_coords.iloc[:, 1]
    x_coords_cols = col_coords.iloc[:, 0]
    y_coords_cols = col_coords.iloc[:, 1]
    
    # Préparation de la figure
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 10))

    # Trace des Tonalités (Lignes - key)
    ax.scatter(x_coords_rows, y_coords_rows, label='Tonalités (key)', 
               marker='o', s=100, color='blue', alpha=0.8)
    for i, txt in enumerate(contingency_table.index):
        ax.annotate(txt, (x_coords_rows.iloc[i] + 0.01, y_coords_rows.iloc[i]), 
                    fontsize=10, color='blue', weight='bold')

    # Trace des Catégories de Streams (Colonnes)
    ax.scatter(x_coords_cols, y_coords_cols, label='Catégories de Streams', 
               marker='s', s=150, color='red', alpha=0.9)
    for i, txt in enumerate(contingency_table.columns):
        ax.annotate(txt, (x_coords_cols.iloc[i] + 0.01, y_coords_cols.iloc[i]), 
                    fontsize=12, color='red', weight='heavy')

    # Axes de référence
    ax.axhline(0, color='grey', linestyle='--', linewidth=1)
    ax.axvline(0, color='grey', linestyle='--', linewidth=1)

    # Configuration du graphique
    ax.set_title("Biplot de l'AFC : Tonalité vs Popularité (2D)", fontsize=16)
    
    dim1_ratio = explained_variance_ratio[0]
    dim2_ratio = explained_variance_ratio[1]
    
    ax.set_xlabel(f"Dimension 1 ({dim1_ratio*100:.2f}%)", fontsize=12)
    ax.set_ylabel(f"Dimension 2 ({dim2_ratio*100:.2f}%)", fontsize=12)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Ajuster les limites
    all_x = np.concatenate([x_coords_rows, x_coords_cols])
    all_y = np.concatenate([y_coords_rows, y_coords_cols])
    margin = 0.2
    ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
    
    # Enregistrement et affichage
    output_filename = "biplot_afc_2D.png"
    fig.savefig(output_filename)
    print(f"\nLe Biplot 2D a été enregistré sous : {output_filename}")
    
    plt.show() # Ceci pourrait encore générer l'avertissement non-interactif, mais le fichier est sauvé.

    # 7. Interprétation
    print("\n--- 6. Interprétation du Biplot 2D ---")
    print("1. Les points proches les uns des autres (tonalités et catégories de streams) sont associés.")
    print("2. L'axe 1 (Horizontal) capture la plus grande partie de la relation. L'axe 2 (Vertical) capture la deuxième plus grande partie, orthogonale à l'axe 1.")
    print("3. Cherchez quelles tonalités sont proches de 'Élevé_Streams' et quelles tonalités sont proches de 'Faible_Streams' pour identifier des tendances de popularité.")


if __name__ == "__main__":
    run_2d_correspondence_analysis()