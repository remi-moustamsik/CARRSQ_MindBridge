from nilearn import image, datasets, input_data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import nibabel as nib
import os
import cv2

from glmsingle import GLM_single
import numpy as np
import pandas as pd
from moviepy import VideoFileClip

import numpy as np
from collections import defaultdict
from scipy.stats import zscore

import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Script de preparation des données pour MindBridge")
    parser.add_argument('--model_name', type=str, required=True, help='Nom du modèle')
    parser.add_argument('--autoencoder_name', type=str, default=None, help='Nom de l\'autoencodeur')
    parser.add_argument('--ckpt_from', type=str, default='last', help='Checkpoint à charger')
    parser.add_argument('--subj_test', type=int, required=True, help='Numéro du sujet à tester')
    parser.add_argument('--batch_size', type=int, default=1, help='Taille du batch')
    parser.add_argument('--gpu_id', type=int, default=0, help='ID du GPU à utiliser')
    parser.add_argument('--data_path', type=str, default='data', help='Chemin vers les données')
    return parser.parse_args()

subject_numbers = [1]

from nilearn.image import resample_to_img

# Function to apply the probmap to an fMRI file
def apply_probmap(fmri_file, probmap_data, output_dir):
    # Charger les données fMRI
    fmri_img = nib.load(fmri_file)
    fmri_data = fmri_img.get_fdata()
    
    # Apply a threshold to binarize the probmap (for example, 0.125)
    roi_mask = probmap_data > 0.125

    # Extract voxels of interest
    extracted_voxels = fmri_data[roi_mask]

    # Show dimensions of extracted fMRI data
    print(f"Dimensions des données fMRI extraites pour {fmri_file}: {extracted_voxels.shape}")

    # Save the extracted voxels in a new NIfTI file (optional)
    affine = fmri_img.affine
    extracted_img = nib.Nifti1Image(extracted_voxels.reshape(-1, 1, 1, fmri_data.shape[-1]), affine)
    
    # Build exit path
    output_path = os.path.join(output_dir, os.path.basename(fmri_file).replace('.nii.gz', '_extracted.nii.gz'))
    nib.save(extracted_img, output_path)

    print(f"Voxels extraits sauvegardés à: {output_path}")




# TR = 1  # Repeat time in seconds, adjust according to your data
TR = 2  # Repeat time in seconds, adjust according to your data

delay_seconds = 6  # Time to see the effect of stimulations
delay_points = delay_seconds / TR  # Number of data points corresponding to the delay


def label_time_series(num_sub, events, time_series):

    l =[]
    # Go through each segment to extract features
    for i in range(1):
        # Initializing the list to store features and labels
        labels = []
        ts = []
        video_name = []
        for _, row in events[f'events_0{num_sub}'][i].iterrows():
            onset_time = row['onset'] + delay_seconds
            trial_type = row['trial_type']


            #adjusted_onset_time = onset_time
            start_index = int(onset_time / TR)  # Convert onset to data point index
            # end_index = start_index + int(row['nb_images'])  # Determine the end index of the segment
            end_index = start_index + int(row['duration'] / TR)
            
            segment = time_series[f'time_series_0{num_sub}'][i][:, start_index:end_index]
            ts.append(segment)
            labels.append(trial_type)
            video_name.append(row['video_name'])

        data = {'labels' : labels, 'time_series' : ts, 'video_name' : video_name}
        # data = {'labels' : labels, 'time_series' : ts}

        df = pd.DataFrame(data)
        l.append(df)

    return l



# Définir les paramètres globaux
stimdur = 0.1  # Durée du stimulus en secondes (exemple)
tr = 1  # Temps de répétition en secondes (exemple)

# Fonction pour créer des onsets pour chaque point des vidéos 'Risk' et 'NonRisk'
def create_onsets_durations(run_dfs):
    onsets = []
    current_idx = 0
    for run_df in run_dfs:
        for i, row in run_df.iterrows():
            # if row['labels'] in ['NonRisk', 'Risk']:
            if row['labels'] in ['ToM', 'Pain']:

                onsets.extend(list(range(current_idx, current_idx + row['pts_nb'])))
            current_idx += row['pts_nb']
    print("Onsets:", onsets)
    return onsets

# Fonction pour créer une matrice de design avec un régresseur pour chaque point des vidéos 'Risk' et 'NonRisk'
def create_design_matrix(run_dfs):
    total_length = sum(run_df['pts_nb'].sum() for run_df in run_dfs)
    # num_trials = sum(row['pts_nb'] for run_df in run_dfs for i, row in run_df.iterrows() if row['labels'] in ['NonRisk', 'Risk'])
    num_trials = sum(row['pts_nb'] for run_df in run_dfs for i, row in run_df.iterrows() if row['labels'] in ['ToM', 'Pain'])

    design_matrix = np.zeros((total_length, num_trials))
    onsets = create_onsets_durations(run_dfs)
    
    for i, onset in enumerate(onsets):
        design_matrix[onset, i] = 1
    
    return design_matrix



# Apply a z-score on the betas
def apply_zscore(betas):
    return zscore(betas, axis=0)


# Préparer les données fMRI en concaténant toutes les séries temporelles pour chaque run
def prepare_fmri_data(run_dfs):
    fmri_runs = []
    for run_df in run_dfs:
        all_time_series = np.concatenate(run_df['time_series'].values, axis=1)
        fmri_runs.append(all_time_series)
    return np.concatenate(fmri_runs, axis=1)


def dataprep(fmri_file_path, events, video_folder, img_file_path, subject_numbers):
    fmri_file_path[f'fmri_file_paths_0{i}']= [rf"G:\fMRI_FINAL_FOLDER\BOLD_DATA\nsdgeneralROI\extracted_voxels\sub-pixar001_task-pixar_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold_extracted.nii.gz"]
    time_series = {}
    for i in subject_numbers:
        # Data extraction for each fMRI run
        fmri_run1_path = fmri_file_path[f'fmri_file_paths_0{i}'][0]
        # fmri_run2_path = fmri_file_paths[f'fmri_file_paths_0{i}'][1]
        
        # Loading fMRI files
        fmri_run1 = nib.load(fmri_run1_path).get_fdata()
        # fmri_run2 = nib.load(fmri_run2_path).get_fdata()
        
        # Using squeeze to remove size 1 dimensions
        fmri_run1_squeezed = np.squeeze(fmri_run1) 
        # fmri_run2_squeezed = np.squeeze(fmri_run2) 
        
        # Concatenation of the time series of the two runs
        time_series[f'time_series_0{i}'] = [fmri_run1_squeezed]
        # time_series[f'time_series_0{i}'] = [fmri_run1_squeezed, fmri_run2_squeezed]
        print(f'time_series_0{i} shape: {time_series[f"time_series_0{i}"][0].shape}')  # Print the shape of the time series
    events = {}
    for i in subject_numbers:
        # events1 = pd.read_csv(f'C:/Users/hennecol/Documents/FMRI experiment/FINAL FOLDER/2 analysis and first preprocessing/events/events_video_name/sub_0{i}-run1_video_name.tsv', sep='\t')
        # events2 = pd.read_csv(f'C:/Users/hennecol/Documents/FMRI experiment/FINAL FOLDER/2 analysis and first preprocessing/events/events_video_name/sub_0{i}-run2_video_name.tsv', sep='\t')
        events1 = pd.read_csv(r"G:\fMRI_FINAL_FOLDER\BOLD_DATA\pixar_events_bis.tsv", sep='\t')
        # events[f'events_0{i}'] = [events1, events2]
        events[f'events_0{i}'] = [events1]
    events['events_01'][0]
    time_series_labels = {}
    for i in subject_numbers:
        time_series_labels[f'time_series_0{i}'] = label_time_series(i, events, time_series)
    # Add a column with the number of points of the time series
    for subj in time_series_labels:
        for i in range(1):
            time_series_labels[subj][i]['pts_nb'] = time_series_labels[subj][i]['time_series'].apply(lambda x: x.shape[1])
    # Créer les matrices de design pour chaque sujet
    design_matrices = {
        subject: create_design_matrix(run_dfs)
        for subject, run_dfs in time_series_labels.items()
    }
    # Préparer les données fMRI pour chaque sujet
    fmri_data = {
        subject: prepare_fmri_data(run_dfs)
        for subject, run_dfs in time_series_labels.items()
    }

    # Exécuter GLMSingle pour chaque sujet
    # Initialiser GLMSingle avec les options supportées
    options = dict()
    glm = GLM_single(options)
    # Ajuster GLM pour chaque sujet
    results = {}
    for subject in fmri_data:
        results[subject] = glm.fit(
            [design_matrices[subject]],  # Matrices de design correspondantes
            [fmri_data[subject]],  # Données fMRI pour le sujet
            stimdur,  # Durée du stimulus
            tr  # Temps de répétition
        )
    results_d = results['time_series_01'][fitting_type]
    fitting_type = "typeb"
    R2 = results_d['R2']
    R2run = results_d['R2run']
    #glmbadness = results_d['glmbadness']
    #rrbadness = results_d['rrbadness']
    HRFindex = results_d['HRFindex']
    HRFindexrun = results_d['HRFindexrun']
    #xvaltrend = results_d['xvaltrend']
    meanvol = results_d['meanvol']
    #FRACvalue = results_d['FRACvalue']
    #scaleoffset = results_d['scaleoffset']
    results_betas = {}

    for num in subject_numbers:
        key = f'time_series_0{num}'
        new_key = f'subj_0{num}'
        
        # Vérifie si la clé existe dans `results` pour éviter les erreurs
        if key in results and fitting_type in results[key] and 'betasmd' in results[key][fitting_type]:
            results_betas[new_key] = results[key][fitting_type]['betasmd']

    for key in results_betas:
    results_betas[key] = np.squeeze(results_betas[key])

    # Initialiser un dictionnaire pour stocker les métriques de chaque voxel
    metrics = {}

    # Parcourir les résultats pour chaque sujet
    for subject, results_list in results.items():
        r_squared = []
        p_values = []
        betas = []
        
        # Parcourir les résultats pour chaque voxel
        for result in results_list:
            r_squared.append(result.rsquared)
            p_values.append(result.pvalues)
            betas.append(result.params)
        
        # Convertir les listes en DataFrame
        metrics_df = pd.DataFrame({
            'R_squared': r_squared,
            'P_values': p_values,
            'Betas': betas
        })
        
        # Stocker le DataFrame dans le dictionnaire des métriques
        metrics[subject] = metrics_df

    # Exemple d'accès aux métriques pour un sujet spécifique
    subject_metrics = metrics['time_series_01']
    print(subject_metrics)

    # Enregistrer les métriques dans un fichier CSV (facultatif)
    subject_metrics.to_csv('subject_metrics_time_series_01.csv', index=False)

    # Apply the z-score on the betas for each subject
    results_betas_zscored = {
        subject: apply_zscore(betas)
        for subject, betas in results_betas.items()
    }

    # Example to check the contents of the new dictionary with z-score applied
    for key, value in results_betas_zscored.items():
        print(f'{key}: {value}')

    # Show a small part of the results to check
    for key in results_betas_zscored:
        print(f'{key} z-scored betas:', results_betas_zscored[key][:5])  # Shows first 5 items to check

    # Transposer les données dans results_betas
    for key in results_betas_zscored:
        results_betas_zscored[key] = np.transpose(results_betas_zscored[key])

    # Creates a mask to know the indices of the last 2 points of each video, corresponding to a blank screen so we must delete them

    # Dictionary to store indices
    indices_suppr = {}

    for subj_num in subject_numbers:
        key = f'time_series_0{subj_num}'
        new_key = f'subj_0{subj_num}'
        
        if key in time_series_labels:
            # List to store clues for the current topic
            subj_indices = []
            
            # Continuous index for points
            current_index = 0
            
            # Processing of the two runs
            for i in range(1):
                df = time_series_labels[key][i]
                
                # Add a 'next_label' column to store the label of the next line
                df['next_label'] = df['labels'].shift(-1)
                
                # Add a 'pts_nb_extract' column initialized with 'pts_nb'
                df['pts_nb_extract'] = df['pts_nb']
                
                # Filter lines where 'labels' is 'No Risk' or 'Risk'
                filtered_df = df[df['labels'].isin(['ToM', 'Pain'])]

                # Calculate continuous indices
                for idx, row in filtered_df.iterrows():
                    pts_nb = row['pts_nb']
                    line_indices = list(range(current_index, current_index + pts_nb))
                    current_index += pts_nb
                    
                    # Store the indices of the last 2 points only if the next line is 'control' or does not exist, as we have split some videos into Risk and NonRisk, we remove the last 2 points only if the following line is control
                    if row['next_label'] == 'control' or pd.isna(row['next_label']):
                        subj_indices.extend(line_indices[-2:])
                        
                        # Update 'pts_nb_extract' to 'pts_nb - 2' for this row
                        df.at[idx, 'pts_nb_extract'] = pts_nb - 2
            
            # Store in dictionary
            indices_suppr[new_key] = subj_indices

    # Dictionary to store filtered results
    results_betas_filtered = {}

    # Browse each topic in indices_suppr
    for subj_key, indices_to_remove in indices_suppr.items():
        # Get data from results_betas for this topic
        data = results_betas_zscored[subj_key]
        
        # Convert indices to be removed into a set for faster searching
        indices_to_remove_set = set(indices_to_remove)
        
        # Create a mask for the lines to keep (True = keep, False = delete)
        mask = np.array([i not in indices_to_remove_set for i in range(data.shape[0])])
        
        # Apply mask to get filtered data
        filtered_data = data[mask]
        
        # Store filtered data in the new dictionary
        results_betas_filtered[subj_key] = filtered_data

    # Loop through each subject in results_betas
    for subj_num in subject_numbers:
        key = f'subj_0{subj_num}'
        data = results_betas_filtered[key]

        # Ensure the directory exists for each subject
        directory = rf"E:\MiB\MindBridge\data\natural-scenes-dataset\webdataset_avg_split\train\subj01{subj_num}4"
        os.makedirs(directory, exist_ok=True)

        # Loop through each line in the data matrix
        for j in range(data.shape[0]):
            line = data[j, :]
            
            # Duplicate the line 3 times
            duplicated_lines = np.tile(line, (3, 1))
            
            # Save to .npy file with the specified naming convention
            file_name = os.path.join(directory, f"sample00000000{j}.nsdgeneral.npy")
            np.save(file_name, duplicated_lines)
    
    # Nouveau dictionnaire pour stocker les lignes filtrées
    time_series_R_NR = {}

    # Parcourir chaque sujet dans time_series_labels
    for subj_num in subject_numbers:
        key = f'time_series_0{subj_num}'
        
        if key in time_series_labels:
            # Liste pour stocker les DataFrames filtrés pour chaque run
            filtered_runs = []
            
            # Traitement des deux runs
            for i in range(1):
                df = time_series_labels[key][i]
                
                # Filtrer les lignes où 'labels' est 'NonRisk' ou 'Risk'
                filtered_df = df[df['labels'].isin(['ToM', 'Pain'])]
                
                # Ajouter le DataFrame filtré à la liste
                filtered_runs.append(filtered_df)
            
            # Stocker la liste des DataFrames filtrés dans le nouveau dictionnaire
            time_series_R_NR[f'subj_0{subj_num}'] = filtered_runs

    video_folder = r"G:\fMRI_FINAL_FOLDER\BOLD_DATA\PARTLY CLOUDY.mp4"

    grouped_videos_per_subject = {f'subj_0{subj_num}': {} for subj_num in subject_numbers}

    for subj_key, runs in time_series_R_NR.items():
        for run in runs:
            for idx, row in run.iterrows():
                video_name = row['video_name']
                pts_nb_extract = row['pts_nb_extract']
                if video_name in grouped_videos_per_subject[subj_key]:
                    grouped_videos_per_subject[subj_key][video_name] += pts_nb_extract
                else:
                    grouped_videos_per_subject[subj_key][video_name] = pts_nb_extract

    for subj_num in subject_numbers:
        subj_key = f'subj_0{subj_num}'
        save_folder = fr"E:\MiB\MindBridge\data\natural-scenes-dataset\webdataset_avg_split\train\subj1{subj_num}"
        os.makedirs(save_folder, exist_ok=True)
        
        frame_index = 0
        
        for video_name, total_frames in grouped_videos_per_subject[subj_key].items():
            # video_path = os.path.join(video_folder, video_name)
            video_path = os.path.join(video_folder)
            
            if not os.path.exists(video_path):
                print(f"Error: Video file {video_name} not found at {video_path}.")
                continue
            
            try:
                clip = VideoFileClip(video_path)
                duration = clip.duration
                frame_times = [t * duration / total_frames for t in range(total_frames)]
                
                frames_extracted = 0
                for t in frame_times:
                    frame = clip.get_frame(t)
                    frame_name = os.path.join(save_folder, f"sample00000000{frame_index}.jpg")
                    frame_index += 1
                    frames_extracted += 1
                    
                    # Convert frame to image
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(frame_name, frame_bgr)
                
                print(f"Extracted {frames_extracted} frames out of {total_frames} requested from {video_name}.")
            
            except Exception as e:
                print(f"Error processing video {video_name}: {e}")

    # Chemin vers le dossier contenant les images
    file_path = r"E:\MiB\MindBridge\data\natural-scenes-dataset\webdataset_avg_split\train\subj11"

    # Parcourir le dossier
    for filename in os.listdir(img_file_path):
        if filename.endswith(".jpg"):  # Vérifier si le fichier est une image .jpg
            image_path = os.path.join(file_path, filename)
            img = cv2.imread(image_path)  # Charger l'image
            if img is not None:
                resized_img = cv2.resize(img, (256, 256))  # Redimensionner l'image à 256x256
                cv2.imwrite(image_path, resized_img)  # Sauvegarder l'image redimensionnée
    
    # Path to the CSV file
    csv_file_path = r"E:\MiB\MindBridge\data\natural-scenes-dataset\nsddata\experiments\nsd\nsd_stim_info_merged.csv"

    # Load the CSV file
    df = pd.read_csv(csv_file_path)

    # Directory to save the .npy files
    save_directory = r'E:\MiB\MindBridge\data\natural-scenes-dataset\webdataset_avg_split\train\subj11'

    # Ensure the directory exists
    os.makedirs(save_directory, exist_ok=True)

    # Process the rows starting from the index 73002
    for index, row in df.iloc[73000:].iterrows():
        coco_id = row['cocoId']
        sample_number = row['Unnamed: 0']
        # Compute the correct index for the filename
        file_index = coco_id - 1000000
        file_name = f'sample00000000{file_index}.coco73k.npy'
        file_path = os.path.join(save_directory, file_name)
        
        # Save the sample number in a .npy file
        np.save(file_path, [sample_number])
        np.save(file_path, [sample_number])

def main():
    # Apply the resampled probmap to each fMRI file
    # output_dir = 'C:/Users/hennecol/Documents/FMRI experiment/FINAL FOLDER/BOLD_DATA/nsdgeneralROI/extracted_voxels'
    output_dir = r"G:\fMRI_FINAL_FOLDER\BOLD_DATA\nsdgeneralROI\extracted_voxels"

    fmri_file_paths= {}
    for i in subject_numbers:
        # fmri_file_paths[f'fmri_file_paths_0{i}']= [f'C:/Users/hennecol/Documents/FMRI experiment/FINAL FOLDER/BOLD_DATA/BIDS/derivatives/sub-0{i}/func/sub-0{i}_task-carrsq_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz', f'C:/Users/hennecol/Documents/FMRI experiment/FINAL FOLDER/BOLD_DATA/BIDS/derivatives/sub-0{i}/func/sub-0{i}_task-carrsq_run-2_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz']
        fmri_file_paths[f'fmri_file_paths_0{i}']= [rf"C:\Users\Rémi\Desktop\CentraleSupelec\Cours\stage\australie\ds000228\derivatives\sub-pixar00{i}\func\sub-pixar00{i}_task-pixar_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"]
    fmri_file_paths

    # Path of the resampled probmap
    # probmap_resampled_path = 'C:/Users/hennecol/Documents/FMRI experiment/FINAL FOLDER/BOLD_DATA/nsdgeneralROI/nsdgeneral_probmap_resampled.nii.gz'
    probmap_resampled_path = r"G:\fMRI_FINAL_FOLDER\BOLD_DATA\nsdgeneralROI\func1pt8mm\nsdgeneral_probmap_resampled.nii.gz"

    # Load resampled probmap
    probmap_resampled_img = nib.load(probmap_resampled_path)
    probmap_resampled_data = probmap_resampled_img.get_fdata()


    for subject, files in fmri_file_paths.items():
        for fmri_file in files:
            apply_probmap(fmri_file, probmap_resampled_data, output_dir)

    # Préparer les arguments pour dataprep
    fmri_file_path = {"fmri_file_paths_01": [fmri_file]}
    events = {}  # À remplir selon tes besoins
    video_folder = r"G:\fMRI_FINAL_FOLDER\BOLD_DATA\PARTLY CLOUDY.mp4"
    img_file_path = r"path\to\images"
    subject_numbers = [i for i in range(2, 156) ]

    # Appel de dataprep
    dataprep(fmri_file_path, events, video_folder, img_file_path, subject_numbers)

if __name__ == "__main__":
    main()