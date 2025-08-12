#!/bin/bash
#Template provided by Daniel Levitas of Indiana University
#Edits by Andrew Jahn, University of Michigan, 07.22.2020

#User inputs:
# bids_root_dir="C:\Users\Rémi\Documents\FMRI experiment\FINAL FOLDER\BOLD_DATA\BIDS"
bids_root_dir="D:\fMRI_FINAL_FOLDER\BOLD_DATA\BOLD_DATA_2\BIDS"

nthreads=8
mem=16 #gb
container=docker #docker or singularity

#Begin:

#Convert virtual memory from gb to mb
mem=`echo "${mem//[!0-9]/}"` #remove gb at end
mem_mb=`echo $(((mem*1000)-5000))` #reduce some memory for buffer space during pre-processing

export FS_LICENSE="D:\fMRI_FINAL_FOLDER\opt\freesurfer\license.txt"
export BIDS_ROOT_DIR="$bids_root_dir"

# Omly for one subject, adaptable for more
for i in 3
do
  subj="0${i}"
  echo "Traitement du sujet $subj"
  python.exe -m fmriprep_docker "$bids_root_dir" "$bids_root_dir\derivatives" \
    participant \
    --participant-label $subj \
    --skip-bids-validation \
    --md-only-boilerplate \
    --fs-license-file "D:\fMRI_FINAL_FOLDER\opt\freesurfer\license.txt" --fs-no-reconall \
    --output-spaces MNI152NLin2009cAsym:res-2 \
    --nthreads $nthreads \
    --stop-on-first-crash \
    --mem_mb $mem_mb
done