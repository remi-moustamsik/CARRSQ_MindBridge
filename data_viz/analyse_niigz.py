import nibabel as nib
import pandas as pd

img = nib.load(r"G:\fMRI_FINAL_FOLDER\BOLD_DATA\nsdgeneralROI\extracted_voxels\sub-pixar001_task-pixar_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold_extracted.nii.gz")
print('nb_volumes:' ,img.shape)  # (x, y, z, n_volumes)

data = pd.read_csv(r"G:\fMRI_FINAL_FOLDER\BOLD_DATA\pixar_events_bis.tsv", sep='\t')
dur_tot = 0
for i in data['duration']:
    dur_tot += i
print('Total duration of all events:', dur_tot)  # Total duration of all events in seconds