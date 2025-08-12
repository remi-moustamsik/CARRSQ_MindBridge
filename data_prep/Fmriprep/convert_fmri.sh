#!/bin/bash

# === CONFIGURATION A MODIFIER ===

# Chemin absolu vers votre dataset BIDS
BIDS_DIR="/mnt/c/Users/Rémi/Desktop/CentraleSupelec/Cours/stage/australie/ds000228"

# Chemin absolu vers le dossier de sortie
OUT_DIR="/mnt/g/fMRI_FINAL_FOLDER/fmriprep_out"

# Chemin vers votre fichier de licence FreeSurfer
FS_LICENSE="/mnt/g/fMRI_FINAL_FOLDER/opt/freesurfer/license.txt"
export FS_LICENSE="/mnt/g/fMRI_FINAL_FOLDER/opt/freesurfer/license.txt"

# Nombre de threads à utiliser (adapter selon votre machine)
THREADS=4

# Mémoire maximale (en Mo)
MEM_MB=8000

# === COMMANDE fMRIPrep AVEC DOCKER ===

docker run -ti --rm \
  -v ${BIDS_DIR}:/data:ro \
  -v ${OUT_DIR}:/out \
  -v ${FS_LICENSE}:/opt/freesurfer/license.txt \
  nipreps/fmriprep:latest \
  /data /out participant \
  --fs-license-file /opt/freesurfer/license.txt \
  --output-spaces MNI152NLin2009cAsym \
  --nthreads ${THREADS} --mem_mb ${MEM_MB}
