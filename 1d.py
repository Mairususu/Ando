# -*- coding: utf-8 -*-
"""
Script d'Analyse Factorielle des Correspondances (AFC) - Version Finale

Objectif : Analyser la relation entre la Tonalité (key) et le Mode (mode)
          dans le jeu de données real_dataset.csv.
"""

import pandas as pd
import numpy as np
from prince import CA # Correspondence Analysis
import matplotlib.pyplot as plt
import seaborn as sns

# Nom du fichier de données
FILE_NAME = "real_dataset.csv"

def run_correspondence_analysis():
    """
    Charge les données, prépare le tableau de contingence, exécute l'AFC
    et affiche les résultats et le Biplot en s'adaptant au nombre de dimensions calculables.
    """
    print(f"--- 1. Chargement et Préparation des Données ({FILE_NAME}) ---")

    try:
        df = pd.read_csv(FILE_NAME)
    except FileNotFoundError:
        print(f"ERREUR : Le fichier '{FILE_NAME}' n'a pas été trouvé.")
        return
    except Exception as e:
        print(f"Une erreur est survenue lors du chargement du fichier : {e}")
        return

    # 1. Nettoyage des données
    df_ca = df[['key', 'mode']].copy()
    df_ca.replace('', np.nan, inplace=True)
    df_ca.dropna(subset=['key', 'mode'], inplace=True)

    print(f"Nombre d'observations après nettoyage (key et mode non nuls) : {len(df_ca)}")

    if len(df_ca) < 10:
        print("Alerte : Trop peu d'observations pour une AFC pertinente après nettoyage.")
        return

    # 2. Création du Tableau de Contingence
    contingency_table = pd.crosstab(df_ca['key'], df_ca['mode'])

    print("\n--- 2. Tableau de Contingence (Tonalité vs Mode) ---")
    print(contingency_table)

    # 3. Exécution de l'Analyse Factorielle des Correspondances
    print("\n--- 3. Exécution de l'AFC ---")
    
    # On demande 2 composantes
    ca = CA(n_components=2, n_iter=10, engine='sklearn')
    ca = ca.fit(contingency_table)

    # 4. Analyse de la Variance Expliquée
    print("\n--- 4. Analyse de la Variance Expliquée ---")
    
    eigenvalues = ca.eigenvalues_
    total_inertia = eigenvalues.sum()
    explained_variance_ratio = eigenvalues / total_inertia
    
    # Le nombre de dimensions réellement calculées
    N_comp = len(eigenvalues)
    
    print(f"Nombre de dimensions calculables : {N_comp}")
    
    print("Variance Expliquée par les Composantes :")
    for i, ratio in enumerate(explained_variance_ratio):
        print(f"Dimension {i+1}: {ratio:.4f} ({ratio*100:.2f}%)")

    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    if N_comp > 0:
        # Affichage dynamique de la variance cumulée
        if N_comp == 1:
            print(f"Variance Cumulée de la seule dimension disponible : {cumulative_variance[0]:.4f} ({cumulative_variance[0]*100:.2f}%)")
        else: # N_comp >= 2
            # L'indice [1] correspond à la somme des dimensions 1 et 2
            print(f"Variance Cumulée des 2 premières dimensions : {cumulative_variance[1]:.4f} ({cumulative_variance[1]*100:.2f}%)")
    else:
        print("Avertissement : Aucune dimension n'a pu être calculée.")
        return

    # 5. Visualisation du Biplot
    print("\n--- 5. Visualisation du Biplot ---")
    
    row_coords = ca.row_coordinates(contingency_table)
    col_coords = ca.column_coordinates(contingency_table)
    
    # CORRIGÉ : Utilisation de .iloc pour extraire la colonne par position
    # Définition des coordonnées X et Y en fonction du nombre de dimensions disponibles
    x_coords_rows = row_coords.iloc[:, 0]
    x_coords_cols = col_coords.iloc[:, 0]
    
    # Si N_comp est 1, on force la coordonnée Y à zéro (projection 1D)
    if N_comp >= 2:
        y_coords_rows = row_coords.iloc[:, 1]
        y_coords_cols = col_coords.iloc[:, 1]
    else:
        y_coords_rows = np.zeros_like(x_coords_rows, dtype=float)
        y_coords_cols = np.zeros_like(x_coords_cols, dtype=float)
        print("NOTE : Seule la Dimension 1 est valide. Le biplot est une projection 1D.")
        

    # Préparation de la figure
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 10))

    row_color = 'skyblue'
    col_color = 'darkorange'

    # Trace des coordonnées des Tonalités (Lignes)
    ax.scatter(x_coords_rows, y_coords_rows, label='Tonalités (key)', 
               marker='o', s=100, color=row_color, alpha=0.8)
    for i, txt in enumerate(contingency_table.index):
        ax.annotate(txt, (x_coords_rows.iloc[i] + 0.01, y_coords_rows[i]), 
                    fontsize=10, color=row_color, weight='bold')

    # Trace des coordonnées des Modes (Colonnes)
    ax.scatter(x_coords_cols, y_coords_cols, label='Mode', 
               marker='s', s=150, color=col_color, alpha=0.9)
    for i, txt in enumerate(contingency_table.columns):
        ax.annotate(txt, (x_coords_cols.iloc[i] + 0.01, y_coords_cols[i]), 
                    fontsize=12, color=col_color, weight='heavy')

    # Ajout des axes de référence au centre (0,0)
    ax.axhline(0, color='grey', linestyle='--', linewidth=1)
    ax.axvline(0, color='grey', linestyle='--', linewidth=1)

    # Configuration du graphique
    ax.set_title("Biplot de l'AFC : Relation entre Tonalité et Mode", fontsize=16)
    dim1_ratio = explained_variance_ratio[0]
    
    # Labels d'axes
    ax.set_xlabel(f"Dimension 1 ({dim1_ratio*100:.2f}%)", fontsize=12)
    if N_comp >= 2:
        dim2_ratio = explained_variance_ratio[1]
        ax.set_ylabel(f"Dimension 2 ({dim2_ratio*100:.2f}%)", fontsize=12)
    else:
        ax.set_ylabel("Dimension 2 (Non disponible - Valeur de 0 forcée)", fontsize=12)
    
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Ajuster les limites
    all_x = np.concatenate([x_coords_rows, x_coords_cols])
    all_y = np.concatenate([y_coords_rows, y_coords_cols])
    margin = 0.2
    
    # Assurez-vous d'avoir au moins une marge minimale même si tous les points sont à 0 sur un axe.
    min_x, max_x = all_x.min(), all_x.max()
    min_y, max_y = all_y.min(), all_y.max()
    
    if min_x == max_x: # Cas où tous les points sont centrés (devrait pas arriver pour Dim 1)
        min_x, max_x = -margin, margin
    if min_y == max_y: # Cas où la Dimension 2 est forcée à zéro (N_comp=1)
        min_y, max_y = -margin, margin
        
    ax.set_xlim(min_x - margin, max_x + margin)
    ax.set_ylim(min_y - margin, max_y + margin)
    
   # Enregistrement de la figure pour la visualisation
    output_filename = "biplot_afc_1D.png"
    fig.savefig(output_filename)
    print(f"\nLe Biplot (projection 1D) a été enregistré sous : {output_filename}")
    
    # Tentative d'affichage (peut générer l'avertissement dans certains environnements)
    plt.show()

    # 6. Interprétation
    print("\n--- 6. Interprétation du Biplot ---")
    print("Étant donné qu'une seule dimension est significative, l'interprétation se concentre sur l'axe horizontal (Dimension 1).")
    print("1. L'axe 1 sépare les modalités de 'key' associées au 'Major' (à droite) de celles associées au 'Minor' (à gauche), ou inversement.")
    print("2. Les tonalités plus éloignées du centre (0,0) sont les plus discriminantes dans cette relation.")


if __name__ == "__main__":
    run_correspondence_analysis()