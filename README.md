# CARRSQ_MindBridge: Specializing Cross-Subject Brain Decoding for Road Scene Understanding

![teaser](assets/CARRS-Q-Logo-PMS-200289.jpg)

[Rémi Moustamsik Billah](https://remi-moustamsik.github.io/home/), [Collaborators], [CARRSQ, QUT]

## Overview
![method]

> CARRSQ_MindBridge is an extension of **[MindBridge](https://github.com/littlepure2333/MindBridge/)**, adapted for the specialization of **visual reconstruction of road scenes** from brain signals (fMRI).
> 
> The goal is to assess whether a brain decoding model, originally designed for general-purpose visual reconstruction, can be fine-tuned to detect, reconstruct, and interpret **road safety-specific elements**, such as:
> - the presence of pedestrians, cyclists, or specific vehicles
> 
> This project investigates:
> 1. **Thematic specialization** – adapting a cross-subject model to images from driving contexts.
> 2. **Robustness** – evaluating generalization to new subjects with limited data.
> 3. **Relevance for road safety** – analyzing the ability to reconstruct critical details for decision-making.

---

## Installation

1. Clone this repository:
```bash
git clone https://github.com/remi-moustamsik/CARRSQ_MindBridge.git
cd CARRSQ_MindBridge
```

## Project Structure

The project is organized into three main directories:

- **MindBridge** – contains the code for training new subjects and performing image reconstruction, adapted from the original MindBridge project.
- **data_prep** – scripts and utilities to format the data so it can be used by MindBridge.
- **data_viz** – tools for visualizing the different types of data used in the project.

The data is only available on the provided hard drive.

## How to use the script

### MindBridge

How to use the model on the HPC:

First, you must download the conda environment at this path:

Then, you must import the .yml file to the HPC home folder \\hpc-fs\home.

Then you must type: conda env create -f mindbridge-history.yml

Your conda environment is created!

Then, you must transfer this folder to the HPC : TO COMPLETE.

Once it is done, you can run two types of jobs:

-          interactive jobs running live, but it may take some time for the scheduler to find a place for you in the queue

-          scripts jobs which don’t run live and will run automatically when there is a place in the queue

Let’s start with interactive jobs which are useful to debug. Mindbridge is resource intensive, and you will need a GPU A100 and 32Gb of ram to run the different scripts. Here is the command to ask for a interactive job for 1:00 hour, with a GPU A100 and 32Gb of ram.

“qsub -I -S /bin/bash -l walltime=1:00:00,ncpus=1,ngpus=1,mem=32GB,gputype=A100”

Then, you must execute the following commands:

“conda activate mindbridge”: to activate the conda environment

“cd MindBridge/”: to change the working directory

“module load cuda/11.7.0” : to load the correct version of cuda

“bash scripts/mon_script.sh” : to execute your bash script

There are a few scripts that you can execute (description taken from MindBridge’s github page):

-          train_single.sh : this script trains the per-subject-per-model version of MindBridge

-          train_bridge.sh : this script contains training MindBridge on multi-subjects (e.g. subj01, 02, 05, 07)

-          adapt_bridge.sh : once the MindBridge is trained on some known "source subjects" (e.g. subj01, 02, 05), you can adapt the MindBridge to a new "target subject" (e.g. subj07) based on limited data volume (e.g. 4000 data points)

-          inference.sh : this script will reconstruct one subject's images (e.g. subj01) on the test set from a MindBridge model (e.g. subj01, 02, 05, 07), then calculate all the metrics. The evaluated metrics will be saved in a csv file. You can indicate which MindBridge model and which subject in the script.

Training the model on a large dataset may take a few hours, so we recommend sending a job script. The model file of a job script is adapt_bridge_hpc.pbs.

To send the job script, you just have to type qsub adapt_bridge_hpc.pbs.

Here is the link to QUT’s tutorial on how to use the HPC : https://qutvirtual4.qut.edu.au/group/staff/research/conducting/facilities/advanced-research-computing-storage/supercomputing/getting-started-with-hpc

How to train the model on a new subject:

-          Prepare your data using the files in the MindBridge/MyDataPrep folder.

-          Export your data to the HPC (don’t forget to put a 0 in front of the subject number)

-          Go into the file src/options.py and add the subject number to:

o   subj-list

o   subj-source

o   subj-taget

o   subj-load

o   subj-test

-          Run the adapt_bridge script

How to recreate images:

Go to scrips/inference.sh.

Enter the model name, its number in subj_load and subj_test, and the weights obtained during training ('last' for the last calculations or 'best' for the best).

Then go to your model’s folder in train_logs and create a folder named “recon_on_subXX” with XX the number of your subject.

You can now run the scripts/inference.sh script.
