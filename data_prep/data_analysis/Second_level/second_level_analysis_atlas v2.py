from nilearn.glm.second_level import SecondLevelModel
from nilearn.plotting import plot_stat_map, find_xyz_cut_coords
from nilearn.image import threshold_img, math_img, load_img, iter_img # Ajout de iter_img ici
from nilearn.reporting import get_clusters_table
from nilearn.regions import connected_regions
from nilearn import datasets
import pandas as pd
import matplotlib.pyplot as plt 
from nilearn.image import resample_to_img
from nilearn.plotting import plot_anat, show, plot_roi, show, plot_stat_map
import nibabel as nib
import numpy as np
from nilearn import plotting
from nilearn.plotting import view_img
from nilearn.image import threshold_img, math_img, load_img, iter_img, resample_to_img, new_img_like




# Download the Harvard-Oxford atlas for cortical regions
atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
#atlas = datasets.fetch_atlas_aal()
atlas_filename = atlas.maps
labels = atlas.labels
print(labels)

# List of first-level z-cards for all subjects
# z_maps = [
#     'C:/Users/hennecol/Documents/FMRI experiment/FINAL FOLDER/Lucas/Results/sub-03/sub-03_Risk-NonRisk_z-map.nii.gz',
#     'C:/Users/hennecol/Documents/FMRI experiment/FINAL FOLDER/Lucas/Results/sub-05/sub-05_Risk-NonRisk_z-map.nii.gz',
#     'C:/Users/hennecol/Documents/FMRI experiment/FINAL FOLDER/Lucas/Results/sub-06/sub-06_Risk-NonRisk_z-map.nii.gz',
#     'C:/Users/hennecol/Documents/FMRI experiment/FINAL FOLDER/Lucas/Results/sub-07/sub-07_Risk-NonRisk_z-map.nii.gz',
#     'C:/Users/hennecol/Documents/FMRI experiment/FINAL FOLDER/Lucas/Results/sub-08/sub-08_Risk-NonRisk_z-map.nii.gz'
# ]
z_maps = [
    r"C:\Users\Rémi\Desktop\australie\codes\Lucas\Results\sub-03\sub-03_Risk-NonRisk_z-map.nii.gz",
    r"C:\Users\Rémi\Desktop\australie\codes\Lucas\Results\sub-05\sub-05_Risk-NonRisk_z-map.nii.gz",
    r"C:\Users\Rémi\Desktop\australie\codes\Lucas\Results\sub-06\sub-06_Risk-NonRisk_z-map.nii.gz",
    r"C:\Users\Rémi\Desktop\australie\codes\Lucas\Results\sub-07\sub-07_Risk-NonRisk_z-map.nii.gz",
    r"C:\Users\Rémi\Desktop\australie\codes\Lucas\Results\sub-08\sub-08_Risk-NonRisk_z-map.nii.gz"
]



# Creation of a DataFrame for second-level covariates, if required
second_level_design_matrix = pd.DataFrame([1] * len(z_maps), columns=['intercept'])

# Initialization and adjustment of the second-level model
second_level_model = SecondLevelModel(smoothing_fwhm=8.0)
second_level_model = second_level_model.fit(z_maps, design_matrix=second_level_design_matrix)

# Definition and calculation of second-level contrast
second_level_contrast = [1]
z_map_second_level = second_level_model.compute_contrast(second_level_contrast, output_type='z_score')

#Define threshold
threshold_ = 3.1

# z map threshold
thresholded_map = threshold_img(z_map_second_level, threshold=threshold_)

# Extraction of connected regions
regions_img, index = connected_regions(thresholded_map, min_region_size=500)



# Creates an average image for subsequent use as a display medium
# subject_images =['C:/Users/hennecol/Documents/FMRI experiment/FINAL FOLDER/BOLD_DATA/BIDS/derivatives/sub-03/anat/sub-03_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz',
#                  'C:/Users/hennecol/Documents/FMRI experiment/FINAL FOLDER/BOLD_DATA/BIDS/derivatives/sub-05/anat/sub-05_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz',
#                   'C:/Users/hennecol/Documents/FMRI experiment/FINAL FOLDER/BOLD_DATA/BIDS/derivatives/sub-06/anat/sub-06_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz',
#                    'C:/Users/hennecol/Documents/FMRI experiment/FINAL FOLDER/BOLD_DATA/BIDS/derivatives/sub-07/anat/sub-07_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz',
#                    'C:/Users/hennecol/Documents/FMRI experiment/FINAL FOLDER/BOLD_DATA/BIDS/derivatives/sub-08/anat/sub-08_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz']

# # Initialiser la somme des images avec des zéros
# first_image = nib.load(subject_images[0])
# mean_image_data = np.zeros(first_image.shape)

# # Accumuler les données de toutes les images
# for subject_image_path in subject_images:
#     image_data = nib.load(subject_image_path).get_fdata()
#     mean_image_data += image_data

# # Calculer la moyenne
# mean_image_data /= len(subject_images)

# # Créer et sauvegarder l'image moyenne
# mean_image_nifti = nib.Nifti1Image(mean_image_data, affine=first_image.affine)
# nib.save(mean_image_nifti, 'C:/Users/hennecol/Documents/FMRI experiment/FINAL FOLDER/BOLD_DATA/BIDS/mean_T1w_image.nii.gz')

# mean_image_path = 'C:/Users/hennecol/Documents/FMRI experiment/FINAL FOLDER/BOLD_DATA/BIDS/mean_T1w_image.nii.gz'

# plotting.plot_anat(mean_image_path, title='Image Moyenne T1', display_mode='ortho', dim=-1, cut_coords=(0, 0, 0))
# plotting.show()



# Visualisation of the ROIs and the atlas
#anat_img_path = 'C:/Users/hennecol/Documents/FMRI experiment/FINAL FOLDER/BOLD_DATA/BIDS/mean_T1w_image.nii.gz'

# # Préparation de l'affichage de l'image anatomique de fond
# display = plot_anat(anat_img_path, title='Superposition des régions connectées et de l\'atlas', draw_cross=False, dim=-1)

# # Rééchantillonnage de l'atlas sur l'image anatomique
# atlas_resampled = resample_to_img(atlas_filename, anat_img_path, interpolation='nearest')

# # Ajouter l'atlas comme contours sur l'image
# display.add_contours(atlas_resampled, levels=[0.5], colors='g')

# # Pour chaque région connectée, ajouter un overlay
# for region_img in iter_img(regions_img):
#     # Rééchantillonne chaque région connectée pour correspondre à l'image anatomique
#     region_resampled = resample_to_img(region_img, anat_img_path, interpolation='nearest')
#     # Superposer les régions connectées avec une transparence
#     display.add_overlay(region_resampled, cmap='autumn', alpha=0.5)

# show()




# Preparing the display with markers for each region
display = plot_stat_map(z_map_second_level, threshold=threshold_, display_mode='ortho', cut_coords=None, title='Group level: Risk vs NonRisk')



# # Iteration on each region to find and display coordinates
# for i, region_img in enumerate(iter_img(regions_img), start=1):
#     coordinates = find_xyz_cut_coords(region_img)
    
#     # Use resample_to_img to ensure spatial compatibility
#     region_resampled = resample_to_img(region_img, atlas_filename, interpolation='nearest')
    
#     # Identifying the nearest atlas region
#     region_data = region_resampled.get_fdata()
#     unique_regions = set(region_data.astype(int).ravel())
    
#     # Prepare to collect region names for display
#     region_names = [labels[region_label] for region_label in unique_regions if region_label != 0]
#     region_names_str = ", ".join(region_names)  # Concatenate all region names into one string
    
#     # Only proceed if there are any regions to display
#     if region_names:
#         region_name_for_file = "_".join(region_names[0].split()).lower()  # Use the first region name for file naming
#         filename = f"C:/Users/hennecol/Documents/FMRI experiment/FINAL FOLDER/Lucas/Results/2nd level analysis/Risk-NonRisk_thresholded_{threshold_}region_{i}_{region_name_for_file}.png"
        
#         # Display the region with all relevant names in the title
#         display_region = plot_stat_map(region_img, title=f"Région {i}: {region_names_str}", display_mode='ortho', cut_coords=coordinates, colorbar=True)
        
#         # Save the region's image with the modified title
#         display_region.savefig(filename)
#         print(f"Région {i} ({region_names_str}): Coordonnées {coordinates} - Sauvegardée sous {filename}")

for i, region_img in enumerate(iter_img(regions_img), start=1):
    coordinates = find_xyz_cut_coords(region_img)
    print(coordinates)
    # Assurez-vous de la compatibilité spatiale
    region_resampled = resample_to_img(region_img, atlas_filename, interpolation='nearest')
    
    # Identification de la région de l'atlas la plus proche
    region_data = region_resampled.get_fdata()
    print(region_data)
    unique_regions = set(region_data.astype(int).ravel())
    
    # Collecte des noms de régions pour l'affichage
    region_names = [labels[region_label] for region_label in unique_regions if region_label != 0]
    region_names_str = ", ".join(region_names)  # Concaténation de tous les noms de régions en une chaîne
    
    if region_names:
        region_name_for_file = "_".join(region_names[0].split()).lower()  # Utilisez le premier nom de région pour le nommage de fichier
        # filename = f"C:/Users/hennecol/Documents/FMRI experiment/FINAL FOLDER/Lucas/Results/2nd level analysis/Risk-NonRisk_thresholded_{threshold_}region_{i}_{region_name_for_file}.png"
        filename = r"C:\Users\Rémi\Desktop\australie\codes\Lucas\Results\2nd level analysis/Risk-NonRisk_thresholded_{threshold_}region_{i}_{region_name_for_file}.png"
        
        # Affichage de la région avec tous les noms pertinents dans le titre
        display_region = plot_stat_map(region_img, title=f"Région {i}: {region_names_str}", display_mode='ortho', cut_coords=coordinates, colorbar=True)
        display_region.add_contours(region_img, levels=[0.5], colors=['purple'])

        
        for region_label in unique_regions:
            if region_label != 0:  # Ignorer l'arrière-plan
                # Extraction de la région spécifique de l'atlas
                region_of_interest = math_img(f'img == {region_label}', img=atlas_filename)
                
                # Ajout des contours de cette région à l'affichage
                display_region.add_contours(region_of_interest, levels=[0.5], colors=['yellow'])
        
        # Sauvegarde de l'image de la région avec le titre modifié
        #display_region.savefig(filename)
        #print(f"Région {i} ({region_names_str}): Coordonnées {coordinates} - Sauvegardée sous {filename}")


# Viewing the thresholded statistical map
plt.show()

# # # Save figure (adjust path to suit your needs)
# display.savefig(f'C:/Users/hennecol/Documents/FMRI experiment/FINAL FOLDER/Lucas/Results/group_level_Risk-NonRisk_thresholded_{threshold_}_sub_ortho_500-voxels.png')


# Sauvegarde de la figure (ajustez le chemin selon vos besoins)

