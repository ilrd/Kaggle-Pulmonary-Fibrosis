from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from time import perf_counter
import os
import tensorflow.keras.backend as K
from datagens import create_datagen
from models import DcmCsvModel

gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)

# ==================================#
# Training data paths
CSV_PATH = '../../data/processed/train.csv'
DCM_PATH = '../../data/processed/train'


# ==================================#
# Create loss func
def loss_wrapper(is_patient_record):
    # is_patient_record - (146) shape array with 1s and 0s that show if there is a patient record for a corresp. week
    is_patient_record = tf.cast(is_patient_record, tf.float32)

    # @tf.function
    def dynamic_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        mse = tf.keras.losses.MSE(is_patient_record * y_true, is_patient_record * y_pred)

        cost = mse
        return cost

    return dynamic_loss


# ==================================#
# GradientTape
def train_step_wrapper(optimizer):
    # @tf.function
    def my_train_step(model, inputs, outputs, is_patient_record):
        y_true = outputs
        with tf.GradientTape() as tape:
            y_pred = model(inputs, training=True)
            loss = loss_wrapper(is_patient_record)(y_true, y_pred)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return loss

    return my_train_step


# ==================================#
# Custom training loop
def train(model, epochs=5, steps_per_epoch=1, shuffle=True):
    datagen_lungs = create_datagen(shuffle, window=[-800, -655])
    datagen_tissue = create_datagen(shuffle, window=[-100, 155])

    history = {}
    history['loss'] = []
    history['val_loss'] = []
    history['batch_loss'] = []
    history['val_batch_loss'] = []

    steps_per_epoch = steps_per_epoch

    for epoch in range(1, epochs + 1):
        loss = 0
        if 15 >= epoch >= 1:
            optimizer = keras.optimizers.Adam(lr=0.001)

        elif 30 > epoch >= 15:
            optimizer = keras.optimizers.Adam(lr=0.0001)

        elif 35 > epoch >= 30:
            optimizer = keras.optimizers.Adam(lr=0.00001)

        else:
            optimizer = keras.optimizers.Adam(lr=0.000001)

        train_step = train_step_wrapper(optimizer=optimizer)

        for current_batch in range(steps_per_epoch):
            begin = perf_counter()
            batch_X, batch_y, is_patient_record = next(datagen_tissue)
            loss_ = float(train_step(model, batch_X, batch_y, is_patient_record))

            batch_X, batch_y, is_patient_record = next(datagen_lungs)
            loss_ += float(train_step(model, batch_X, batch_y, is_patient_record))
            end = perf_counter()

            loss += loss_
            print(f"Epoch {epoch}, step {current_batch}:\n"
                  f"Epoch avg loss: {loss / (current_batch + 1)}, batch loss: {loss_}")

            history['batch_loss'].append(loss_)

            print(f'batch was processed in {end - begin} seconds\n')

        history['loss'].append(loss / steps_per_epoch)

    return history


# ==================================#
# Training

model = DcmCsvModel()
fit_history = train(model, 30, steps_per_epoch=176, shuffle=True)

# Saving the model
model.save('saved_models/saved_model', save_format='tf')


# Plot the predicted FVC values for the weeks, as well as true values and a common-sense baseline
def plot_prediction(model):
    datagen_tissue = create_datagen(True, window=[-100, 155])
    batch_X, batch_y, is_patient_record = next(datagen_tissue)

    y_pred = model.predict(batch_X)[0] * 832.5021066817238 + 2690.479018721756
    y_pred_baseline = np.array([np.array(list([batch_X[0][0][1]]) * 146)])[0] * 832.5021066817238 + 2690.479018721756

    plt.plot(y_pred, label='y_pred')
    plt.plot(y_pred_baseline, label='y_baseline')
    plt.plot(batch_y[0] * 832.5021066817238 + 2690.479018721756, label='y_true')
    plt.legend()


# plot_prediction(model)


# Plot history of a given metric
def plot_history(fit_history, metric='loss'):
    plt.figure(figsize=(14, 14))
    plt.title(metric)
    plt.plot(fit_history[metric])


# plot_history(fit_history)


# ==================================#
# Score

def score(fvc_true, fvc_pred, sigma):
    sigma_clip = np.maximum(sigma, 70)
    delta = np.abs(fvc_true - fvc_pred)
    delta = np.minimum(delta, 1000)
    sq2 = np.sqrt(2)
    metric = np.mean((delta / sigma_clip) * sq2 + np.log(sigma_clip * sq2))
    return np.mean(-metric)


def get_submission_score(model):
    const_scores = []
    model_scores = []
    fit_gen = create_datagen(window=[-100, 155])
    for i in range(10):
        [csv_X, dcm_X], y, dmat = next(fit_gen)

        y_pred = np.array([np.array(list([csv_X[0][1]]) * 146)])[0]

        y_true = (y * dmat)[0] * 832.5021066817238 + 2690.479018721756
        y_pred = (y_pred * dmat)[0] * 832.5021066817238 + 2690.479018721756

        y_true = y_true[y_true != 2690.479]
        y_pred = y_pred[y_pred != 2690.479]

        const_scores.append(score(y_true, y_pred, 100))

        y_pred = model.predict([csv_X, dcm_X], steps=1)[0]

        y_pred = (y_pred * dmat)[0] * 832.5021066817238 + 2690.479018721756
        y_pred = y_pred[y_pred != 2690.479]
        model_scores.append(score(y_true, y_pred, 100))
        print(i, 'done', end='\r', flush=True)

    print(np.mean(const_scores), 'baseline_score')
    print(np.mean(model_scores), 'model_score')


get_submission_score(model)

# ==================================#
# Training with EfficientNet (implemented in my Kaggle notebook)

from tensorflow.keras.layers import *


def train_eff_net():
    import efficientnet.tfkeras as efn
    from tensorflow.keras.models import load_model
    from tensorflow.keras import Model

    efnb5_model = load_model('efficientnetb5-50epochs.h5')

    dcm_input = Input((512, 512, 15))
    x = Conv2D(filters=128, kernel_size=7, padding='same')(dcm_input)
    x = Conv2D(filters=256, kernel_size=5, padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=1, kernel_size=3, padding='same')(x)
    x = efnb5_model.layers[1](x)
    x = MaxPool2D(2, 2)(x)
    x = Conv2D(filters=512, kernel_size=3, padding='same')(x)
    x = MaxPool2D(2, 2)(x)
    x = Conv2D(filters=256, kernel_size=3, padding='same')(x)
    x = MaxPool2D(2, 2)(x)
    x = Conv2D(filters=128, kernel_size=3, padding='same')(x)
    x = Flatten()(x)

    csv_input = Input(shape=(7,))
    csv_x = Dense(16, activation='relu')(csv_input)
    csv_x = GaussianNoise(0.1)(csv_x)

    x = Concatenate()([x, csv_x])
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(200, activation='relu')(x)
    x = Dense(146)(x)

    efn_model = Model([csv_input, dcm_input], x)

    efn_model.layers[4].trainable = False

    fit_history = train(efn_model, 35, True)

    efn_model.save('efn_model_35epochs.h5')
