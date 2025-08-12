from nilearn import plotting, input_data
from nilearn.glm.first_level import FirstLevelModel
from nilearn.plotting import plot_stat_map
import pandas as pd
import numpy as np
import nibabel as nib
from nilearn import plotting
from nilearn.plotting import plot_design_matrix
import matplotlib.pyplot as plt


sub_nb = '08'

# Loading data
bold_files = [f'C:/Users/hennecol/Documents/FMRI experiment/FINAL FOLDER/BOLD_DATA/BIDS/derivatives/sub-{sub_nb}/func/sub-{sub_nb}_task-carrsq_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz', f'C:/Users/hennecol/Documents/FMRI experiment/FINAL FOLDER/BOLD_DATA/BIDS/derivatives/sub-{sub_nb}/func/sub-{sub_nb}_task-carrsq_run-2_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz']
confounds_files = [f'C:/Users/hennecol/Documents/FMRI experiment/FINAL FOLDER/BOLD_DATA/BIDS/derivatives/sub-{sub_nb}/func/sub-{sub_nb}_task-carrsq_run-1_desc-confounds_timeseries.tsv', f'C:/Users/hennecol/Documents/FMRI experiment/FINAL FOLDER/BOLD_DATA/BIDS/derivatives/sub-{sub_nb}/func/sub-{sub_nb}_task-carrsq_run-2_desc-confounds_timeseries.tsv']
events_files = [f'C:/Users/hennecol/Documents/FMRI experiment/FINAL FOLDER/BOLD_DATA/BIDS/sub-{sub_nb}/func/sub-{sub_nb}_task-carrsq_run-1_events.tsv', f'C:/Users/hennecol/Documents/FMRI experiment/FINAL FOLDER/BOLD_DATA/BIDS/sub-{sub_nb}/func/sub-{sub_nb}_task-carrsq_run-2_events.tsv']

# Confounder loading and preparation function
def prepare_confounds(confounds_file):
    confounds = pd.read_csv(confounds_file, delimiter='\t')
    selected_confounds = confounds[['global_signal','white_matter', 'csf', 'csf_wm', 'framewise_displacement', 'cosine00', 'cosine01','a_comp_cor_00','a_comp_cor_01', 'a_comp_cor_02', 'a_comp_cor_03','a_comp_cor_04', 'a_comp_cor_05', 'trans_x', 'trans_y', 'trans_z','rot_x', 'rot_y', 'rot_z', 'tcompcor']]
    selected_confounds = selected_confounds.bfill().fillna(0)
    return selected_confounds

#['framewise_displacement', 'a_comp_cor_00','a_comp_cor_01', 'a_comp_cor_02', 'a_comp_cor_03','a_comp_cor_04', 'a_comp_cor_05', 'trans_x', 'trans_y', 'trans_z','rot_x', 'rot_y', 'rot_z', 'global_signal', 'white_matter']

# Preparing confounders for each run
confounds_run1 = prepare_confounds(confounds_files[0])
confounds_run2 = prepare_confounds(confounds_files[1])

# Loading events for each run
events_run1 = pd.read_csv(events_files[0], delimiter='\t')
events_run2 = pd.read_csv(events_files[1], delimiter='\t')


# Initializing the GLM model
t_r = 1.0  # Repeat time, adjust according to your data
glm_model = FirstLevelModel(t_r=t_r, slice_time_ref=0.5, hrf_model='glover')

# GLM model fitting for each run
glm_model = glm_model.fit(run_imgs=bold_files, 
                          events=[events_run1, events_run2], 
                          confounds=[confounds_run1, confounds_run2])


#Plots the design matrix
design_matrix = glm_model.design_matrices_[1]  # Pour le premier run, utilisez [1] pour le second, etc.
print(design_matrix, type(design_matrix))

plot_design_matrix(design_matrix)
plt.title('Matrice de Design GLM')
plt.show()



#Definition of the variable to which Risk is compared: Change 'control' or 'NonRisk'.
risk_type_vs = 'NonRisk'

# Defining the contrast of interest
contrast_def = f'Risk - {risk_type_vs}'  # Ajustez selon vos conditions

# Contrast calculation
z_map = glm_model.compute_contrast(contrast_def, output_type='z_score')

z_map_filename = f'C:/Users/hennecol/Documents/FMRI experiment/FINAL FOLDER/Lucas/Results/sub-{sub_nb}/sub-{sub_nb}_Risk-{risk_type_vs}_z-map.nii.gz'
nib.save(z_map, z_map_filename)

#A changer:
threshold_ = 3.1

display = plot_stat_map(z_map, threshold=threshold_, display_mode='z', cut_coords=3, title=f'Risk vs {risk_type_vs}')
display.savefig(f'C:/Users/hennecol/Documents/FMRI experiment/FINAL FOLDER/Lucas/Results/sub-{sub_nb}/sub-{sub_nb}_Risk-{risk_type_vs}_threshold{threshold_}.png')

# Visualisation
plotting.show()