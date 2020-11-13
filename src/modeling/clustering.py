from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from models import clustering_autoencoder
from tensorflow.keras import callbacks
import datetime
from random import randint

# Setting seed
SEED = 4
np.random.seed(SEED)

gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)


# ==================================#
# Data generator
def get_data(ids=None, window=None, ids_shuffle=False, iterate=True, seed=None, batch_size=1):
    data_dir = '../../data/processed/train'

    patient_ids = np.array(os.listdir(data_dir))

    if ids_shuffle:
        if seed:
            np.random.seed(seed)
        else:
            np.random.seed(randint(5, 500))
        np.random.shuffle(patient_ids)

    if ids:
        patient_ids = patient_ids[ids[0]:ids[1] - 1]

    batch = []
    while True:
        for patient_id in patient_ids:
            img_names = os.listdir(os.path.join(data_dir, patient_id))

            for j, img_name in enumerate(img_names, start=1):
                img_path = os.path.join(data_dir, patient_id, img_name)
                img_array = np.load(img_path)

                if window:
                    lb = window[0]
                    ub = window[1]
                    img_array[img_array < lb] = lb
                    img_array[img_array > ub] = ub
                    img_array = (img_array - lb) / (ub - lb)

                img_array = np.reshape(img_array, newshape=(512, 512, 1))
                batch.append(img_array)

                if len(batch) == batch_size:
                    yield np.array(batch)
                    batch = []

        if not iterate:
            break


# ==================================#
# Custom callback
class BatchLoss(callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.model.history.history['batch_loss'] = []

    def on_batch_end(self, batch, logs=None):
        batch_loss = logs.get('loss')
        self.model.history.history['batch_loss'].append(batch_loss)


# ==================================#

def fit_model(model, train_gen, val_gen, batch_size=1, epochs=5, train_callbacks=None):
    train_gen = ((train_sample, train_sample) for train_sample in train_gen)
    val_gen = ((val_sample, val_sample) for val_sample in val_gen)
    history = model.fit(train_gen, batch_size=1, epochs=epochs, validation_data=val_gen,
                        steps_per_epoch=130 * 15 // batch_size, callbacks=train_callbacks,
                        validation_steps=20 * 15 // batch_size)
    return history


# ==================================#
# Training

def training_loop(autoencoder, window):
    train_samples = [0, 130]
    val_samples = [130, 150]
    log_dir = "logs/clustering_fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_filepath = f'./saved_models/clustering_models/clustering_checkpoint.h5'

    train_callbacks = [
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=2,
            min_delta=0.01,
            cooldown=0,
            min_lr=1e-5,
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
    ]

    batch_size = 5
    train_gen = get_data(ids=train_samples, window=window, ids_shuffle=True, seed=SEED, batch_size=batch_size)
    val_gen = get_data(ids=val_samples, window=window, ids_shuffle=True, seed=SEED, batch_size=batch_size)

    train_history = fit_model(autoencoder, train_gen, val_gen, batch_size, 10, train_callbacks).history

    return train_history


window = [-100, 155]


# autoencoder = clustering_autoencoder()
# full_train_history = training_loop(autoencoder, window=window)

# ==================================#
# Plot training history
def plot_history(history):
    plt.figure()
    plt.plot(history['batch_loss'][:])

    plt.figure()
    plt.plot(history['val_loss'][:], label='val_loss')
    plt.plot(history['loss'][:], label='loss')
    plt.legend()
    plt.show()


# ==================================#

# Plot example of encoded-decoded image along with the original
def plot_example(autoencoder, window):
    gen = get_data(ids=[150, 175], window=window, ids_shuffle=True, batch_size=1, iterate=False)

    x = next(gen)

    x = np.reshape(x, newshape=(1, 512, 512, 1))

    decoded1 = autoencoder.predict(x)
    x = np.reshape(x, newshape=(512, 512))
    decoded1 = np.reshape(decoded1, newshape=(512, 512))
    plt.figure()
    plt.subplot(121)
    plt.imshow(x, plt.cm.bone)
    plt.subplot(122)
    plt.imshow(decoded1, plt.cm.bone)


# plot_example(autoencoder, window=window)


# ==================================#
# K-means clustering based on encoded vector of the images
autoencoder = keras.models.load_model('./saved_models/clustering_models/clustering_15ep_1024dim.h5')
encoder = autoencoder.layers[1]
preproc_gen = get_data(ids=[0, 90], window=window, iterate=False)
from sklearn.cluster import KMeans

scores = []

X = encoder.predict(preproc_gen)

for n_clusters in range(5, 25):
    kmeans = KMeans(n_clusters=n_clusters)

    kmeans.fit(X)
    scores.append(kmeans.score(X))
    print(n_clusters, 'clusters done')

plt.figure()
plt.plot(scores)
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
plt.bar(x=range(1, 101), height=per_var[:100], tick_label=labels[:100])
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()

pca_df = pd.DataFrame(X_pca, columns=labels)

plt.figure()
plt.scatter(pca_df.PC1, pca_df.PC2)
plt.show()
