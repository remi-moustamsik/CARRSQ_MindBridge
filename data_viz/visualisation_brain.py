import numpy as np

bold_responses = np.load(r"E:\MiB\MindBridge\data\natural-scenes-dataset\webdataset_avg_split\train\subj02\sample000000300.nsdgeneral.npy")  # Shape (n_trials, n_voxels)
print(bold_responses)

# voxel_coords = np.load('subj01_voxel_coords.npy')  # Shape (n_voxels, 3)
# print(voxel_coords.shape)

# # Supposons que le cerveau soit de taille (128, 128, 128) en voxels
# brain_volume = np.zeros((128, 128, 128))

# # Choisir une réponse cérébrale pour un stimulus donné
# stimulus_idx = 0  # premier stimulus
# response = bold_responses[stimulus_idx]  # taille (n_voxels,)

# for i, (x, y, z) in enumerate(voxel_coords):
#     brain_volume[int(x), int(y), int(z)] = response[i]

# import matplotlib.pyplot as plt

# plt.imshow(brain_volume[:, :, 64], cmap='gray')  # slice au milieu
# plt.colorbar()
# plt.title('IRM fonctionnelle reconstituée (slice z=64)')
# plt.show()
