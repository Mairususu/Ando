import pandas as pd
df = pd.read_csv("DATASET.csv", encoding="latin1")
df = df.dropna().reset_index(drop=True) # On enleve les lignes avec une valeur manquante et on rectifie les indices 
#print(df_clean["key"].unique())
for k in range(len(df)): 
    if(df.loc[k,"mode_x"]=="Major"):
        df.loc[k,"mode_x"]=1
    else :
        df.loc[k,"mode_x"]=0

df = df.drop(columns=['track_name','artist(s)_name','released_year','released_month','released_day','in_spotify_playlists','in_spotify_charts','streams', 'liveness_%', 'instrumentalness_%','speechiness_%','streams_per_day','release_date','months_since_release']) # suppression de la colonne key  
# df = df.drop(columns=['released_year']) # suppression de la colonne key  
df = df.rename(columns={'mode_x': 'Major'}) # On renomme la colonne mode par Major: Si le morceau est en majeur on met 1 et si c'est en mineur on met 0
#print(df.dtypes) #pour voir si on a des int partout


df.to_csv("adapt4.csv")

