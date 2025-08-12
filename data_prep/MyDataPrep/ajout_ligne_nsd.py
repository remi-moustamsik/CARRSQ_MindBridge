import csv
import os

fichier = r"D:\MiB\MindBridge\data\natural-scenes-dataset\nsddata\experiments\nsd\nsd_stim_info_merged.csv"
colonnes = ['cocoID', 'cocoSplit']
nouvelles_lignes = 819

# Lire la dernière ligne existante
if os.path.exists(fichier):
    with open(fichier, 'r', newline='') as f:
        reader = csv.DictReader(f)
        lignes = list(reader)
        if lignes:
            dernier_id = int(lignes[-1]['cocoID'])
            dernier_score = int(lignes[-1]['Score'])

# Ajouter les nouvelles lignes
with open(fichier, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=colonnes)

    # Écrire l'en-tête si le fichier est vide
    if os.stat(fichier).st_size == 0:
        writer.writeheader()

    for i in range(1, nouvelles_lignes + 1):
        nouvelle_ligne = {
            'cocoID': dernier_id + i,
            'cocoSplit': dernier_id,  # ou une valeur fixe
        }
        writer.writerow(nouvelle_ligne)
