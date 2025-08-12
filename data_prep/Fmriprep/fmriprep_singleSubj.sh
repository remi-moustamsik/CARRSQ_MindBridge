#!/bin/bash
#Template provided by Daniel Levitas of Indiana University
#Edits by Andrew Jahn, University of Michigan, 07.22.2020

#User inputs:
bids_root_dir="G:\BOLD_DATA_2\BIDS"

subj=03
nthreads=8
mem=16 #gb
container=docker #docker or singularity

#Begin:

#Convert virtual memory from gb to mb
mem=`echo "${mem//[!0-9]/}"` #remove gb at end
mem_mb=`echo $(((mem*1000)-5000))` #reduce some memory for buffer space during pre-processing

export FS_LICENSE="D:/fMRI_FINAL_FOLDER/opt/freesurfer/license.txt"
export BIDS_ROOT_DIR="$bids_root_dir"

#Run fmriprep
python.exe -m fmriprep_docker "$bids_root_dir" "$bids_root_dir\derivatives" \
  participant \
  --participant-label $subj \
  --skip-bids-validation \
  --md-only-boilerplate \
  --fs-license-file "G:/fMRI_FINAL_FOLDER/opt/freesurfer/license.txt"  --fs-no-reconall \
  --output-spaces MNI152NLin2009cAsym:res-2 \
  --nthreads $nthreads \
  --stop-on-first-crash \
  --mem_mb $mem_mb

