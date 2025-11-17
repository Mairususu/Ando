import pandas as pd
df = pd.read_csv("spotify-2023.csv", encoding="latin1")
df = df.dropna().reset_index(drop=True) # On enleve les lignes avec une valeur manquante et on rectifie les indices 
#print(df_clean["key"].unique())
new_col = ["A", "A#", "B", "C#", "D", "D#", "E", "F", "F#","G", "G#"]
for col in new_col:
    df[col] = 0

for k in range(len(df)):
    key = df.loc[k, "key"]
    if key in new_col:
      df.loc[k, key] = 1  
    if(df.loc[k,"mode"]=="Major"):
        df.loc[k,"mode"]=1
    else :
        df.loc[k,"mode"]=0

df = df.rename(columns={'mode': 'Major'}) # On renomme la colonne mode par Major: Si le morceau est en majeur on met 1 et si c'est en mineur on met 0
df = df.drop(columns=['key','track_name','artist(s)_name']) # suppression de la colonne key  

#print(df.dtypes) #pour voir si on a des int partout
df['streams'] = df['streams'].str.replace(',', '', regex=False) #Si on a par exemple 1,050 on le converti en 1050
df['streams'] = pd.to_numeric(df['streams'], errors='coerce')

df['in_deezer_playlists'] = df['in_deezer_playlists'].str.replace(',', '', regex=False)
df['in_deezer_playlists'] = pd.to_numeric(df['in_deezer_playlists'], errors='coerce')

df['in_shazam_charts'] = df['in_shazam_charts'].str.replace(',', '', regex=False)
df['in_shazam_charts'] = pd.to_numeric(df['in_shazam_charts'], errors='coerce')

print(df.in_shazam_charts)
df.to_csv("adapt2.csv")

# for k in range (len(df)):
#    if (df.loc[k, "key"] == "A"): 
#       df.loc[k,"A"] =1
#    elif (df.loc[k, "key"] == "A#"): 
#        df[k,"A#"] =1
#    elif (df.loc[k, "key"] == "B"):
#        df.loc[k,"B"] =1
#    elif (df.loc[k, "key"]== "C#"): 
#        df.loc[k,"C#"] =1 
#    elif (df.loc[k, "key"] == "D"): 
#        df.loc[k,"D"] =1
#    elif (df.loc[k, "key"] == "D#"): 
#        df.loc[k,"D#"] =1    
#    elif (df.loc[k, "key"]== "E"): 
#        df.loc[k,"E"] =1
#    elif (df.loc[k, "key"] == "F"): 
#        df.loc[k,"F"] =1 
#    elif (df.loc[k, "key"] == "F#"): 
#       df.loc[k,"F#"] =1    
#    elif (df.loc[k, "key"] == "G"): 
#        df.loc[k,"G"] =1
#    elif (df.loc[k, "key"] == "G#"): 
#        df.loc[k,"G#"] =1  


#['B' 'C#' 'F' 'A' 'D' 'F#' 'G#' 'G' 'E' 'A#' 'D#']
#df_clean.to_csv("spotify_numeric_clean.csv", index=False, encoding="utf-8") # Sauvegarder le CSV nettoyé
