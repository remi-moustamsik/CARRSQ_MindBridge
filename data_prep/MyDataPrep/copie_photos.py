import shutil
import glob
import os

# Dossier source et destination
source_folder = r"E:\MiB\MiB2\data\natural-scenes-dataset\webdataset_avg_split\train\subj03"
destination_folder = r"E:\MiB\MiB2\data\natural-scenes-dataset\webdataset_avg_split\train\subj07"

# S'assurer que le dossier destination existe
os.makedirs(destination_folder, exist_ok=True)

# Parcourir tous les fichiers .png et les copier
for file_path in glob.glob(os.path.join(source_folder, "*.jpg")):
    shutil.copy(file_path, destination_folder)

print("Tous les fichiers .png ont été copiés !")
