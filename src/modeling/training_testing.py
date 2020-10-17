from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from time import perf_counter
import os
import tensorflow.keras.backend as K
from datagens import create_datagen

gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)

# To debug:
# tf.debugging.set_log_device_placement(True)

# ==================================#
# Global variables
CSV_PATH = '../../data/processed/train.csv'
DCM_PATH = '../../data/processed/train'
num_of_patients = len(os.listdir(DCM_PATH))

# ==================================#
# Importing model class
from models import DcmCsvModel

model = DcmCsvModel()


# ==================================#
# Create loss func
def loss_wrapper(is_patient_record):  # is_patient_record - 146 len 1D array with 1s and 0s
    is_patient_record = tf.cast(is_patient_record, tf.float32)

    def dynamic_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        mse = K.mean(K.square(y_true - is_patient_record * y_pred), axis=-1)

        cost = mse
        return cost

    return dynamic_loss


# ==================================#
# GradientTape
def train_step_wrapper(optimizer):
    @tf.function
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
# Training with GradientTape
def train(model, epochs=5, debug=False, shuffle=True):
    datagen = create_datagen(shuffle)

    history = {}
    history['loss'] = []
    history['val_loss'] = []
    history['batch_loss'] = []
    history['val_batch_loss'] = []

    steps_per_epoch = num_of_patients

    for epoch in range(1, epochs + 1):
        if not debug:
            loss = 0
            if epoch == 1:
                optimizer = keras.optimizers.Adam(lr=0.001)

            elif 5 > epoch >= 2:
                optimizer = keras.optimizers.Adam(lr=0.0001)

            elif 8 > epoch >= 5:
                optimizer = keras.optimizers.Adam(lr=0.00001)

            else:
                optimizer = keras.optimizers.Adam(lr=0.000001)

            train_step = train_step_wrapper(optimizer=optimizer)

            for current_batch in range(steps_per_epoch):
                batch_X, batch_y, is_patient_record = next(datagen)
                # is_patient_record = np.ones((1, 146))

                begin = perf_counter()
                loss_ = float(train_step(model, batch_X, batch_y, is_patient_record))
                # loss_ = model.train_on_batch(batch_X, batch_y)

                end = perf_counter()

                loss += loss_
                print(f"Epoch {epoch}, step {current_batch}:\n"
                      f"Epoch avg loss: {loss / (current_batch + 1)}, batch loss: {loss_}")

                history['batch_loss'].append(loss_)

                print(f'batch was processed in {end - begin} seconds\n')

            history['loss'].append(loss / steps_per_epoch)

        else:
            loss = 0


            for current_batch in range(steps_per_epoch):
                batch_X, batch_y, is_patient_record = next(datagen)
                # is_patient_record = np.ones((1, 146))

                if epoch == 1:
                    optimizer = keras.optimizers.Adam(lr=0.001)
                    model.compile(optimizer=optimizer, loss=loss_wrapper(is_patient_record), run_eagerly=True)

                elif 5 > epoch >= 2:
                    optimizer = keras.optimizers.Adam(lr=0.001)
                    model.compile(optimizer=optimizer, loss=loss_wrapper(is_patient_record), run_eagerly=True)


                elif 8 > epoch >= 5:
                    optimizer = keras.optimizers.Adam(lr=0.0001)
                    model.compile(optimizer=optimizer, loss=loss_wrapper(is_patient_record), run_eagerly=True)


                else:
                    optimizer = keras.optimizers.Adam(lr=0.00001)
                    model.compile(optimizer=optimizer, loss=loss_wrapper(is_patient_record), run_eagerly=True)



                begin = perf_counter()
                loss_ = model.train_on_batch(batch_X, batch_y)

                end = perf_counter()

                loss += loss_
                print(f"Epoch {epoch}, step {current_batch}:\n"
                      f"Epoch avg loss: {loss / (current_batch + 1)}, batch loss: {loss_}")

                history['batch_loss'].append(loss_)

                print(f'batch was processed in {end - begin} seconds\n')

            history['loss'].append(loss / steps_per_epoch)

    return history


# ==================================#
# Testing
fit_history = train(model, 3, False, True)

plt.figure(figsize=(14, 14))
plt.title('Batch loss')
plt.plot(fit_history["batch_loss"][:])
#
fit_gen = create_datagen(1)
[csv_X, dcm_X, dcm_X_num], y, dmat = next(fit_gen)

y_pred = model.predict([csv_X, dcm_X, dcm_X_num], steps=1)


plt.figure(figsize=(14, 14))
plt.title('y_pred on first patient')
plt.plot(y_pred[0])

plt.figure(figsize=(14, 14))
plt.title('y_pred vs y_true')
plt.plot((y_pred * dmat)[0], label='y_pred')
plt.plot(y[0], label='y_true')
plt.show()

# 0.0004509228480436342
