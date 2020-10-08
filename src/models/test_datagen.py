from tensorflow import keras
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.layers import (
    Dense, Conv2D, MaxPool2D, Activation,
    Input, Flatten, Lambda, Concatenate, Reshape,
)
from tensorflow.keras import Model
from keras_nalu.nalu import NALU
import tensorflow as tf
from datagens import DcmDataGenerator, CsvDataGenerator
from time import perf_counter
import tensorflow.keras.backend as K

gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)

# To debug:
# tf.debugging.set_log_device_placement(True)

# ==================================#
# Global variables
CSV_PATH = '../../data/processed/train.csv'

# ==================================#

dcm_inp = Input(shape=(512, 512, 40))
dcm_hid = Conv2D(32, 3, activation='relu', padding='same')(dcm_inp)
dcm_hid = MaxPool2D(2, 2)(dcm_hid)
# dcm_hid = Conv2D(32, 3, activation='relu', padding='same')(dcm_hid)
dcm_hid = MaxPool2D(2, 2)(dcm_hid)
# dcm_hid = Conv2D(64, 3, activation='relu', padding='same')(dcm_hid)
dcm_hid = MaxPool2D(2, 2)(dcm_hid)
dcm_hid = Flatten()(dcm_hid)
dcm_hid = Dense(1, activation='relu')(dcm_hid)  # 256

csv_inp = Input(shape=(7,))
csv_hid = Dense(128, activation='selu')(csv_inp)

conc = Concatenate()([csv_hid, dcm_hid])
conc_dense = Dense(1024, activation='selu')(conc)  # 128
# conc_dense = Dense(512, activation='selu')(conc_dense)  # 128
out = Dense(146)(conc_dense)

# out = Reshape((8, 2))(conc_dense)

inputs = [csv_inp, dcm_inp]
model = Model(inputs, out)


# ==================================#

def loss_wrapper(patient_record):  # patient_records - 146 len 1D array with 1s and 0s
    patient_record = tf.cast(patient_record, tf.float32)

    def dynamic_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        # ypred_np = (patient_record * y_pred).numpy()
        # y_np = y_true.numpy()
        mse = K.mean(K.square(y_true - patient_record * y_pred), axis=-1)
        min_max_penalizer = K.square(K.max(y_true) - K.max(y_pred)) + K.square(K.min(y_true) - K.min(y_pred))
        print(K.mean(y_pred).numpy(), y_pred.numpy().max(), y_pred.numpy().min())
        cost = mse + min_max_penalizer
        return cost

    return dynamic_loss


# ==================================#
# Creating datagen
def merge_datagens(csv_gen, dcm_gen):
    csv_flow = csv_gen.flow()
    dcm_flow = dcm_gen.flow()
    while True:
        csv_outp = next(csv_flow)
        dcm_outp = next(dcm_flow)
        # yield [csv_outp[0], dcm_outp[0]], csv_outp[1][:, :, 0], csv_outp[1][:, :, 1]
        # Without dicom images
        yield [csv_outp[0], np.zeros(shape=dcm_outp[0].shape)], csv_outp[1][:, :, 0], csv_outp[1][:, :, 1]

        # Without "weeks"
        # yield [csv_outp[0][:,1:], dcm_outp[0]], csv_outp[1][:,:,1]


def create_datagen(batch_size=1):
    csv_datagen = CsvDataGenerator(CSV_PATH, normalize=True, batch_size=batch_size)
    dcm_datagen = DcmDataGenerator('../../data/processed/train', batch_size=batch_size)

    merged_gen = merge_datagens(csv_datagen, dcm_datagen)

    return merged_gen


# ==================================#
# Making dynamically-changing loss func

# have to define manually a dict to store all epochs scores


def dynamic_loss_training(model, batch_size=1, num_of_patients=52):
    datagen = create_datagen(batch_size=batch_size)

    history = {}
    history['history'] = {}
    history['history']['loss'] = []
    history['history']['val_loss'] = []
    history['history']['batch_loss'] = []
    history['history']['val_batch_loss'] = []

    # first compiling with mse

    # define number of iterations in training and test
    steps_per_epoch = round(num_of_patients / batch_size)
    # test_iter = round(testX.shape[0] / batch_size)
    for epoch in range(1, 2):

        # train iterations
        loss = 0
        for current_batch in range(steps_per_epoch):
            batch_X, batch_y, patient_record = next(datagen)

            model.compile(loss=loss_wrapper(patient_record), optimizer=keras.optimizers.RMSprop(lr=0.1 / 10 ** epoch),
                          metrics=['mae'], run_eagerly=True)

            loss_, mae = model.train_on_batch(batch_X, batch_y)
            print(f"Epoch {epoch}, batch {current_batch}:\n"
                  f"Epoch avg loss - {loss / (current_batch + 1)}, batch loss - {loss_}, mae - {mae}")
            history['history']['batch_loss'].append(loss_)

            loss += loss_

        history['history']['loss'].append(loss / steps_per_epoch)

        # # test iterations
        # val_loss = 0
        # for i in range(test_iter):
        #     start = i * batch_size
        #     end = i * batch_size + batch_size
        #     batchX = testX[start:end, ]
        #     batchy = testy[start:end, ]
        #
        #     val_loss_ = model.test_on_batch(batchX, batchy)
        #
        #     val_loss += val_loss_
        #
        # history['history']['val_loss'].append(val_loss / test_iter)

    return model, history


# ==================================#
# Testing
model, history = dynamic_loss_training(model, 1)
fit_gen = create_datagen(1)
[csv_X, dcm_X], y, dmat = next(fit_gen)

y_pred = model.predict([csv_X, dcm_X], steps=1)
print(np.mean(y_pred))
# plt.figure(figsize=(14, 14))
# plt.plot(y_pred[0])
# plt.plot(y[0])

plt.figure(figsize=(14, 14))
plt.plot(history['history']["batch_loss"][150:])

plt.figure(figsize=(14, 14))
plt.plot(y_pred[0])
plt.plot(y[0])
plt.show()
