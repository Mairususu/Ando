import pandas as pd
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing 
import matplotlib.pyplot as plt 

# Turnin JSON file into Dataframe 
with open('onepiece_characters.json') as f:
    data = json.load(f)

# Use pd.json_normalize to convert the JSON to a DataFrame
df = pd.json_normalize(data)

# Rename the columns for clarity
df.columns = ['Name', 'Bounty', 'Age', 'Crew', 'Fruit']

print(df.head())
print(df.shape)

# Clean Bounty, age and fruit
df['Bounty'] = df['Bounty'].str.replace('.', '', regex=False).astype(float)
df['Age'] = df['Age'].str.extract(r'(\d+)').astype(float)
df['Fruit'] = df['Fruit'].apply(lambda x: 1 if pd.notna(x) and x.strip() != '' else 0)

# Drop non numeric columns 
df = df.drop(columns=['Crew'])
df = df.drop(columns=['Name'])
df = df.dropna()

print(df.head())
print(df.shape)
print(df.dtypes)

# Center and scale data 
scaled_data = preprocessing.scale(df)

# Create PCA object : 
pca = PCA()
pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)
per_var = np.round(pca.explained_variance_ratio_ * 100, decimals = 1)

# Create labels for the scree plot (PC1, PC2...)
labels = [ 'PC' + str(x) for x in range(1, len(per_var)+1)]

# Create barplot 
plt.bar(x = range(1, len(per_var) + 1), height = per_var, tick_label = labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Components')
plt.title('Scree Plot')


pca_df = pd.DataFrame(pca_data, columns = labels)
plt.scatter(pca_df.PC1, pca_df.PC2)
plt.title('My PCA graph')
plt.xlabel('PC1 - {0}%'.format(per_var[0]))
plt.ylabel('PC2 - {0}%'.format(per_var[1]))
for sample in pca_df.index : 
    plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))

plt.show()


# Interpretation : 
# The PCA of the variables Bounty, Age, and Fruit shows that the first two components explain about 75% of the total variance.
# The first axis (PC1) is strongly associated with bounty values, reflecting the power or notoriety of characters.
# Characters with high bounties are located on the right-hand side, while those with low bounties cluster on the left.
# The second axis (PC2) appears to differentiate characters based on the presence or absence of a Devil Fruit.
# Overall, this suggests that bounty evolution in One Piece is primarily explained by a combination of power 
# (reflected by bounty) and possession of a Fruit, with age playing a smaller role.