import numpy as np

coco_file = r"E:\MiB\MindBridge\data\natural-scenes-dataset\webdataset_avg_split\train\subj03\sample00000000452.coco73k.npy"

# Charge le fichier
data = np.load(coco_file)

# Affiche son contenu
print(data)
