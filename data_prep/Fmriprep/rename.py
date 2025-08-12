# import os
# import re


# # Chemin du dossier contenant les dossiers à renommer
# root_dir = r"G:\fMRI_FINAL_FOLDER\ds000228"

# # Parcours tous les dossiers du répertoire racine
# for name in os.listdir(root_dir):
#     match = re.match(r"sub-pixar(\d+)", name)
#     if match:
#         new_name = f"sub-{match.group(1)}"
#         src = os.path.join(root_dir, name)
#         dst = os.path.join(root_dir, new_name)
#         print(f"Renomme {src} -> {dst}")
#         os.rename(src, dst)

import os
import re

# Chemin du dossier contenant les dossiers à renommer
root_dir = r"G:\fMRI_FINAL_FOLDER\ds000228"

# Parcours tous les dossiers du répertoire racine
for name in os.listdir(root_dir):
    match = re.match(r"sub-(\d+)", name)
    if match:
        sub_dir = os.path.join(root_dir, name)
        # Parcours tous les fichiers dans le dossier (récursivement)
        for dirpath, dirnames, filenames in os.walk(sub_dir):
            for filename in filenames:
                if "pixar" in filename:
                    new_filename = filename.replace("pixar", "")
                    src_file = os.path.join(dirpath, filename)
                    dst_file = os.path.join(dirpath, new_filename)
                    print(f"Renomme fichier {src_file} -> {dst_file}")
                    os.rename(src_file, dst_file)