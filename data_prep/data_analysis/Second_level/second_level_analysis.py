from nilearn.glm.second_level import SecondLevelModel
from nilearn.plotting import plot_stat_map
import pandas as pd
from nilearn import plotting

# Liste des cartes z de premier niveau pour tous les sujets
z_maps = [
    'C:/Users/hennecol/Documents/FMRI experiment/FINAL FOLDER/Lucas/Results/sub-03/sub-03_Risk-NonRisk_z-map.nii.gz',
    'C:/Users/hennecol/Documents/FMRI experiment/FINAL FOLDER/Lucas/Results/sub-05/sub-05_Risk-NonRisk_z-map.nii.gz',
    'C:/Users/hennecol/Documents/FMRI experiment/FINAL FOLDER/Lucas/Results/sub-06/sub-06_Risk-NonRisk_z-map.nii.gz',
    'C:/Users/hennecol/Documents/FMRI experiment/FINAL FOLDER/Lucas/Results/sub-07/sub-07_Risk-NonRisk_z-map.nii.gz',
    'C:/Users/hennecol/Documents/FMRI experiment/FINAL FOLDER/Lucas/Results/sub-07/sub-07_Risk-NonRisk_z-map.nii.gz'
]

# Création d'un DataFrame pour les covariables de second niveau, si nécessaire
# Par exemple, l'âge ou le sexe des sujets
# Si vous n'avez pas de covariables, vous pouvez passer `None` au modèle
second_level_design_matrix = pd.DataFrame(
    [1] * len(z_maps),  # Ici, un vecteur de 1 pour un effet moyen
    columns=['intercept']
)

# Initialisation du modèle de second niveauje ne co
second_level_model = SecondLevelModel(smoothing_fwhm=8.0)

# Ajustement du modèle
second_level_model = second_level_model.fit(z_maps, design_matrix=second_level_design_matrix)

# Définition du contraste de second niveau (ici, l'effet moyen à travers les sujets)
second_level_contrast = [1]  # Un vecteur avec un 1 pour 'intercept' si vous testez l'effet moyen

# Calcul du contraste de second niveau
z_map_second_level = second_level_model.compute_contrast(second_level_contrast, output_type='z_score')

# Visualisation de la carte statistique de second niveau
display = plot_stat_map(z_map_second_level, threshold=1.96, display_mode='z', cut_coords=3, title='Group level: Risk vs NonRisk')
display.savefig('C:/Users/hennecol/Documents/FMRI experiment/FINAL FOLDER/Lucas/Results/group_level_Risk-NonRisk_threshold1.96.png')

# Visualisation
plotting.show()

