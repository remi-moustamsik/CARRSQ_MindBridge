import gzip
import shutil

input_file = r"G:\fMRI_FINAL_FOLDER\BOLD_DATA\ds000228\sub-001\func\sub-pixar001_task-pixar_bold.nii\sub-pixar001_task-pixar_bold.nii"
output_file = input_file + ".gz"

with open(input_file, 'rb') as f_in, gzip.open(output_file, 'wb') as f_out:
    shutil.copyfileobj(f_in, f_out)

print("Compression terminée :", output_file)