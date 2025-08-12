import nibabel as nib

img = nib.load(r"C:\Users\Rémi\Desktop\australie\codes\Lucas\Results\sub-03\sub-03_Risk-control_z-map.nii.gz")
data = img.get_fdata()
n_volumes = data.shape[2]  # 4e dimension = nombre de volumes
print(f"Nombre de volumes : {n_volumes}")

import csv

# Nom du fichier de sortie
filename = "sub-01_task-Risk2_events.tsv"

# Données (tu peux les remplir dynamiquement)
events = [
    ["onset", "duration", "trial_type"],
    [10.0, 6.0, "risk"],
    [20.0, 6.0, "risk"],
    [30.0, 6.0, "risk"],
    [40.0, 6.0, "risk"],
    [50.0, 6.0, "risk"],
    [70.0, 6.0, "neutral"],
    [80.0, 6.0, "neutral"],
    [90.0, 6.0, "neutral"],
    [100.0, 6.0, "neutral"],
    [110.0, 6.0, "neutral"],
    # ... ajoute le reste ici
]

# Écriture du fichier TSV
with open(filename, "w", newline="") as tsvfile:
    writer = csv.writer(tsvfile, delimiter="\t")
    writer.writerows(events)

print(f"Fichier {filename} créé avec succès.")
