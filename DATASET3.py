import pandas as pd
df = pd.read_csv("DATASET.csv", encoding="latin1")
df = df.dropna().reset_index(drop=True) # On enleve les lignes avec une valeur manquante et on rectifie les indices 
#print(df_clean["key"].unique())

for k in range(len(df)):
    if(df.loc[k,"mode_x"]=="Major"):
        df.loc[k,"mode_x"]=1
    else :
        df.loc[k,"mode_x"]=0

df = df.rename(columns={'mode_x': 'Major'}) # On renomme la colonne mode par Major: Si le morceau est en majeur on met 1 et si c'est en mineur on met 0
df = df.drop(columns=['track_name','artist(s)_name', 'release_date', 'streams_per_day']) # suppression de certaines colonnes


df.to_csv("adapt3.csv")