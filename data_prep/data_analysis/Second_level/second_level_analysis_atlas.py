from nilearn.glm.second_level import SecondLevelModel
from nilearn.plotting import plot_stat_map, find_xyz_cut_coords
from nilearn.image import threshold_img, math_img, load_img, iter_img # Ajout de iter_img ici
from nilearn.reporting import get_clusters_table
from nilearn.regions import connected_regions
from nilearn import datasets
import pandas as pd
import matplotlib.pyplot as plt 
from nilearn.image import resample_to_img
from nilearn.plotting import plot_anat, show

# Téléchargement de l'atlas de Harvard-Oxford pour les régions corticales
atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
atlas_filename = atlas.maps
labels = atlas.labels

# Liste des cartes z de premier niveau pour tous les sujets
z_maps = [
    'C:/Users/hennecol/Documents/FMRI experiment/FINAL FOLDER/Lucas/Results/sub-03/sub-03_Risk-NonRisk_z-map.nii.gz',
    'C:/Users/hennecol/Documents/FMRI experiment/FINAL FOLDER/Lucas/Results/sub-05/sub-05_Risk-NonRisk_z-map.nii.gz',
    'C:/Users/hennecol/Documents/FMRI experiment/FINAL FOLDER/Lucas/Results/sub-06/sub-06_Risk-NonRisk_z-map.nii.gz',
    'C:/Users/hennecol/Documents/FMRI experiment/FINAL FOLDER/Lucas/Results/sub-07/sub-07_Risk-NonRisk_z-map.nii.gz',
    'C:/Users/hennecol/Documents/FMRI experiment/FINAL FOLDER/Lucas/Results/sub-08/sub-08_Risk-NonRisk_z-map.nii.gz'
]

# Création d'un DataFrame pour les covariables de second niveau, si nécessaire
second_level_design_matrix = pd.DataFrame([1] * len(z_maps), columns=['intercept'])

# Initialisation et ajustement du modèle de second niveau
second_level_model = SecondLevelModel(smoothing_fwhm=8.0)
second_level_model = second_level_model.fit(z_maps, design_matrix=second_level_design_matrix)

# Définition et calcul du contraste de second niveau
second_level_contrast = [1]
z_map_second_level = second_level_model.compute_contrast(second_level_contrast, output_type='z_score')

# Seuil de la carte z
thresholded_map = threshold_img(z_map_second_level, threshold=3.1)

# Extraction des régions connectées
regions_img, index = connected_regions(thresholded_map, min_region_size=100)



# Préparation de l'affichage avec les marqueurs pour chaque région
display = plot_stat_map(z_map_second_level, threshold=1.96, display_mode='ortho', cut_coords=None, title='Group level: Risk vs NonRisk')

# Itération sur chaque région pour trouver les coordonnées et les afficher
for i, region_img in enumerate(iter_img(regions_img), start=1):
    display_region = plot_stat_map(region_img, title=f"Région connectée {i}", display_mode='ortho', cut_coords=None, colorbar=True)
    coordinates = find_xyz_cut_coords(region_img)
    # Ajout de marqueurs pour chaque région détectée

    print(f"Région {i}: Coordonnées {coordinates}")
    
    # Utilisation de resample_to_img pour assurer la compatibilité spatiale
    region_resampled = resample_to_img(region_img, atlas_filename, interpolation='nearest')
    
    # Identification de la région de l'atlas la plus proche
    region_data = region_resampled.get_fdata()
    unique_regions = set(region_data.astype(int).ravel())  # Convertit en entiers et trouve des valeurs uniques
    for region_label in unique_regions:
        if region_label == 0:  # Ignorer le fond
            continue
        print(f"Région {i} correspond à: {labels[region_label]}")

# Visualisation de la carte statistique seuillée
plt.show()

# Sauvegarde de la figure (ajustez le chemin selon vos besoins)
display.savefig('C:/Users/hennecol/Documents/FMRI experiment/FINAL FOLDER/Lucas/Results/group_level_Risk-NonRisk_thresholded_1.96_5sub_ortho.png')