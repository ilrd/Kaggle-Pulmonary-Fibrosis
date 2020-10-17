from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from models import clustering_autoencoder
from tensorflow.keras import callbacks
import datetime

np.random.seed(4)

gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)

window = [-1024, -769]


# ==================================#

def get_data(nums=None, window=None, shuffle=True, full_shuffle=False, names=False, iter=True):
    data_dir = '../../data/clustering/train'

    all_images = np.array(os.listdir(data_dir))
    if full_shuffle:
        np.random.shuffle(all_images)

    if nums:
        img_names = all_images[nums[0]:nums[1]]
    else:
        img_names = all_images

    while True:
        if shuffle:
            np.random.shuffle(img_names)
        for j, dcm_img_name in enumerate(img_names, start=1):

            dcm_img_path = os.path.join(data_dir, dcm_img_name)
            dcm_img_array = np.load(dcm_img_path)
            if window:
                lb = window[0]
                ub = window[1]
                dcm_img_array[dcm_img_array < lb] = lb
                dcm_img_array[dcm_img_array > ub] = ub
                dcm_img_array = (dcm_img_array - lb) / (ub - lb)

            dcm_img_array = np.reshape(dcm_img_array, newshape=(1, 512, 512, 1))
            if not names:
                yield dcm_img_array
            else:
                yield dcm_img_array, dcm_img_name

        if not iter:
            break


# ==================================#

class BatchLoss(callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.model.history.history['batch_loss'] = []

    def on_batch_end(self, batch, logs=None):
        batch_loss = logs.get('loss')
        self.model.history.history['batch_loss'].append(batch_loss)


class RestoreDefaults(callbacks.Callback):
    def on_train_begin(self, logs=None):
        default_lr = 0.001
        tf.keras.backend.set_value(self.model.optimizer.lr, default_lr)


# ==================================#

def fit_model(model, train_gen, val_gen, samples, epochs=5, train_callbacks=None):
    steps_per_epoch = samples[1] - samples[0]
    train_gen = ((train_sample, train_sample) for train_sample in train_gen)
    val_gen = ((val_sample, val_sample) for val_sample in val_gen)
    history = model.fit(train_gen, batch_size=1, epochs=epochs, validation_data=val_gen,
                        steps_per_epoch=steps_per_epoch, callbacks=train_callbacks,
                        validation_steps=5)
    return history


# ==================================#
# Training, testing

def training_loop(autoencoder):
    train_samples = [0, 100]
    val_samples = [0, 2000]
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_filepath = f'modeling/tmp/clustering_checkpoint3.h5'

    train_callbacks = [
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=3,
            min_delta=0.01,
            cooldown=0,
            min_lr=1e-6,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
        ),
        callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
        ),
        BatchLoss(),
        # RestoreDefaults(),
    ]

    gen = get_data(nums=train_samples, window=window, full_shuffle=True)
    val_datagen = get_data(nums=val_samples, window=window, full_shuffle=True)

    train_history = fit_model(autoencoder, gen, val_datagen, train_samples, 15, train_callbacks).history

    return train_history


autoencoder, encoder, decoder = clustering_autoencoder()
# training_loop(autoencoder)


# ==================================#

def plot_example():
    gen = get_data(nums=[2000, 3000], window=window, full_shuffle=True)

    x1 = next(gen)

    # autoencoder = keras.models.load_model('modeling/tmp/clustering_checkpoint3.h5')
    #
    # x1 = np.reshape(x1, newshape=(1, 512, 512, 1))
    #
    # decoded1 = autoencoder.predict(x1)
    # x1 = np.reshape(x1, newshape=(512, 512))
    # decoded1 = np.reshape(decoded1, newshape=(512, 512))
    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(x1)
    # plt.subplot(122)
    # plt.imshow(decoded1)

    autoencoder = keras.models.load_model('modeling/tmp/clustering_checkpoint3.h5')

    x1 = np.reshape(x1, newshape=(1, 512, 512, 1))

    decoded1 = autoencoder.predict(x1)
    x1 = np.reshape(x1, newshape=(512, 512))
    decoded1 = np.reshape(decoded1, newshape=(512, 512))
    plt.figure()
    plt.subplot(121)
    plt.imshow(x1)
    plt.subplot(122)
    plt.imshow(decoded1)


plot_example()


# ==================================#

# plt.figure()
# plt.plot(full_train_history[0]['batch_loss'][:])
#
# plt.figure()
# plt.plot(full_train_history[0]['val_loss'][:], label='val_loss')
# plt.plot(full_train_history[0]['loss'][:], label='loss')


# ==================================#
# K-means
autoencoder = keras.models.load_model('modeling/tmp/clustering_checkpoint3.h5')
encoder = autoencoder.layers[1]
preproc_gen = get_data(nums=[0, 1], window=window, shuffle=False, names=False, iter=False)
enc1 = encoder.predict(preproc_gen)
from sklearn.cluster import KMeans

nums = [0, 500]
preproc_gen = get_data(nums=nums, window=window, shuffle=False, names=False, iter=False)

scores = []

X = encoder.predict(preproc_gen)

for n_clusters in range(10, 11):
    clustering_model = KMeans(n_clusters=n_clusters)

    clustering_model.fit(X)
    scores.append(clustering_model.score(X))
    print(n_clusters, 'clusters done')

# plt.figure()
# plt.plot(scores)
# ==================================#

# PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA()
X_pca = pca.fit_transform(X_scaled)
per_var = np.round(pca.explained_variance_ratio_ * 100, 1)
labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]
plt.figure()
plt.bar(x=range(1, X.shape[1] + 1), height=per_var[:], tick_label=labels[:])
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()

pca_df = pd.DataFrame(X_pca, columns=labels)

plt.figure()
plt.scatter(pca_df.PC1, pca_df.PC2)
plt.show()
# ==================================#
# Processing the data

# train_dir = '../../data/processed/train'
# imgs = 0
# for (dcm_img_array, name) in preproc_gen:
#     dcm_img_ds = np.array([dcm_img_array, np.empty(shape=(1,), dtype=np.float32)])
#
#     dcm_img_ds[1] = encoder.predict(dcm_img_array)
#
#     patient_ID, j = name[:-4].split('-')
#
#     if not os.path.isdir(f'{train_dir}/{patient_ID}'):
#         os.mkdir(f'{train_dir}/{patient_ID}')
#
#     np.save(f'{train_dir}/{patient_ID}/{j}', dcm_img_ds)
#     imgs += 1
#     if imgs >= nums[1] - nums[0]:
#         break

# ==================================#
