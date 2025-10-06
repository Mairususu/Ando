Wesh la team§

C'est le baitounet a l'appareil!

Wesh Baitounet, comment ça va ?

Bien et toi, utilisatuer inconnu?

Caca

Caca toi même, je ne te permet pas

on est très matures dit donc

ouiiii

Rim Tu tournes 

Allez vous faire foutre 

hhhhaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa



Pour Bait:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
#from sklearn.datasets import load_wine

                #EXO1

#1 and 2

data=pd.read_csv("heptathlon_2025.csv", sep=";")
#print(data)
data_subset = data.iloc[:, 2:]
data_subset= data_subset.apply(pd.to_numeric).to_numpy()
#print(data_subset)
#print(data_subset.dtype)
mean = np.mean(data_subset, axis=0)
std = np.std(data_subset, axis=0, ddof=0)
data_standardized = (data_subset - mean) / std  #standardisé = centré réduit
C_stand= np.cov(data_standardized, rowvar= False)
C_unstand=np.cov(data_subset, rowvar= False)
print("Cov standarlized : \n", C_stand)
print("Cov unstandarlized : \n", C_unstand)


#3

eigen= np.linalg.eig(C_stand)
eigenval= eigen[0]
eigenvect= eigen[1]
#print("eigen values:\n", eigenval)
print("eigen vectors:\n", eigenvect)
sort= (np.argsort(eigenval))[::-1]
eigenval_sorted=eigenval[sort]
eigenvect_sorted = eigenvect[:, sort]
print(" sorted eigen values:\n", eigenval_sorted)

#4
total_variance = np.sum(eigenval_sorted)
explained_variance_ratio = eigenval_sorted / total_variance
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.7)
plt.ylabel('Proportion of Variance Explained')
plt.xlabel('Principal Component')
plt.title('Variance Explained by Each Principal Component')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
#The last 2 or 3 variables could be removed

#5
eigenval_sorted2=eigenval_sorted[:6]
eigenvect_sorted2=eigenvect_sorted[:, :6]
sqrt_eigenval = np.sqrt(eigenval_sorted2)  # racine carrée des valeurs propres
correlations = eigenvect_sorted2 * sqrt_eigenval
print("correlations:",correlations)
#
