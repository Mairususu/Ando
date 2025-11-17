import pandas as pd
df = pd.read_csv("spotify-2023.csv", encoding="latin1")
df_clean = df.dropna() # On enleve les lignes avec une ou plusieurs valeurs manquantes
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
df = df.drop(columns=['key']) # suppression de la colonne key  
df.to_csv("adapt.csv")

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
