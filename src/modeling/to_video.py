import numpy as np
import pandas as pd
import os
import pydicom
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
from IPython.display import HTML

patients_dir = '../../data/processed/train'
patient_ids = os.listdir(patients_dir)


def plot_ynw(patient_dcm_paths):
    dcm_arrays = []
    # lb = -90
    # ub = 165
    lb = -1024
    ub = 99999

    plt.figure()
    for j, patient_dcm_path in enumerate(patient_dcm_paths):

        final = np.load(patient_dcm_path)
        final[final < lb] = lb
        final[final > ub] = ub
        plt.subplot(5, 3, j + 1)
        plt.imshow(final, plt.cm.bone)
        dcm_arrays.append(final)
    print(len(dcm_arrays))
    plt.show()


for patient_id in patient_ids:
    patient_path = os.path.join(patients_dir, patient_id)

    dcm_names = np.array([dcm_name[:-4] for dcm_name in os.listdir(patient_path)], dtype=int)
    dcm_names = sorted(list(dcm_names))

    patient_dcm_paths = [f'{patients_dir}/{patient_id}/{dcm_num}.npy' for dcm_num in dcm_names]

# dcm_arrays = np.array(dcm_arrays)
#
# fig = plt.figure()
#
# ims = []

# for image in range(0, dcm_arrays.shape[0], 1):
#     im = plt.imshow(dcm_arrays[image,:, :],
#                     animated=True, cmap=plt.cm.bone)
#     plt.axis("off")
#     ims.append([im])
#
# ani = animation.ArtistAnimation(fig, ims, interval=100)
