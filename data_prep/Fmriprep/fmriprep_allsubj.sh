#!/bin/bash
#Template provided by Daniel Levitas of Indiana University
#Edits by Andrew Jahn, University of Michigan, 07.22.2020

#User inputs:
# bids_root_dir="C:\Users\Rémi\Documents\FMRI experiment\FINAL FOLDER\BOLD_DATA\BIDS"
bids_root_dir="G:/BOLD_DATA_2/BIDS"

nthreads=8
mem=16 #gb
container=docker #docker or singularity

#Begin:

#Convert virtual memory from gb to mb
mem=`echo "${mem//[!0-9]/}"` #remove gb at end
mem_mb=`echo $(((mem*1000)-5000))` #reduce some memory for buffer space during pre-processing

#export TEMPLATEFLOW_HOME=$HOME/.cache/templateflow
# export FS_LICENSE="C:\Users\hennecol\Documents\FMRI experiment\FINAL FOLDER\2 analysis and first preprocessing\texte.txt"
export FS_LICENSE="C:\Users\Rémi\Desktop\australie\codes\license.txt"
export PATH="/c/Users/Rémi/AppData/Local/Programs/Python/Python312/:$PATH"
export BIDS_ROOT_DIR="$bids_root_dir"

# Boucle sur les sujets de 2 à 155
for i in 3
do
  subj="0${i}"
  echo "Traitement du sujet $subj"
  python.exe -m fmriprep_docker "$bids_root_dir" "$bids_root_dir\derivatives" \
    participant \
    --participant-label $subj \
    --skip-bids-validation \
    --md-only-boilerplate \
    --fs-license-file "C:\Users\Rémi\Desktop\australie\codes\license.txt" --fs-no-reconall \
    --output-spaces MNI152NLin2009cAsym:res-2 \
    --nthreads $nthreads \
    --stop-on-first-crash \
    --mem_mb $mem_mb
done