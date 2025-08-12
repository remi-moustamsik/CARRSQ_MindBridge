# Permet d'obtenir lh.nsdgeneral.mgz et rh.nsdgeneral.mgz à partir de l'image fonctionnelle prétraitée

freesurfer_home=/mnt/d/freesurfer
export FREESURFER_HOME="$freesurfer_home"
source $FREESURFER_HOME/SetUpFreeSurfer.sh

# Pour l'hémisphère gauche
mri_vol2surf \
  --mov "/mnt/d/ds000228/derivatives/sub-pixar001/func/sub-pixar001_task-pixar_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz" \
  --regheader fsaverage \
  --hemi lh \
  --o lh.nsdgeneral.mgz \
  --projfrac 0.5

# Pour l'hémisphère droit
mri_vol2surf \
  --mov "/mnt/d/ds000228/derivatives/sub-pixar001/func/sub-pixar001_task-pixar_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz" \
  --regheader fsaverage \
  --hemi rh \
  --o rh.nsdgeneral.mgz \
  --projfrac 0.5

