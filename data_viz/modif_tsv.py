import pandas as pd

# Charger le fichier TSV
df = pd.read_csv(r"G:\fMRI_FINAL_FOLDER\BOLD_DATA\pixar_events.tsv", sep='\t')

# Ajouter la colonne 'video_name' avec la valeur 'Party Cloudy' pour chaque ligne
df['video_name'] = 'Partly Cloudy'

# Sauvegarder le fichier TSV modifié
df.to_csv(r"G:\fMRI_FINAL_FOLDER\BOLD_DATA\pixar_events_bis.tsv", sep='\t', index=False)