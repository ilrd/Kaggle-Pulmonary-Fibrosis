from tensorflow import keras
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Activation, Input, Flatten
from tensorflow.keras import Model

# As we see, Sex and SmokingStatus as categorical features, so let's create their numerical versions:
CATEGORICAL_FEATURES = ['Sex', 'SmokingStatus']
from sklearn.preprocessing import LabelEncoder

labels_decoded = dict()


def show_window(ct_img, w=None, m=None):
    ct_img = ct_img.copy()
    if w and m:
        l_bound = int(m - w / 2)
        r_bound = int(m + w / 2)
        ct_img[ct_img < l_bound] = l_bound
        ct_img[ct_img > r_bound] = r_bound
    else:
        w = abs(ct_img.min()) + ct_img.max()
        m = w / 2

    plt.figure(figsize=(9, 9))
    plt.title(f'Window with size: {w} and mean {m}')
    plt.imshow(ct_img, plt.cm.bone)


csv_data_dir = '../../data/raw'

train_df = pd.read_csv(f'{csv_data_dir}/train.csv')
records = []

for patient in train_df['Patient'].unique():
    num_of_records = np.sum(train_df['Patient'] == patient)

    for _ in range(num_of_records):
        records.append(num_of_records)

train_df['Records'] = records

for cf in CATEGORICAL_FEATURES:
    print('Train data:\n', train_df[cf].value_counts(), '\n\n')
    le = LabelEncoder()
    train_df[cf + '_n'] = le.fit_transform(train_df[cf])

    ul = train_df[cf].unique()
    labels_decoded[cf] = dict(zip(ul, le.transform(train_df[cf].unique())))

print(train_df.head(), '\n')
print('Categorical labels and their encoded vesions:\n', labels_decoded)


# Import data generators










train_path = f'../../data/processed/train'

# # Simple CNN
# i = Input(shape=())
