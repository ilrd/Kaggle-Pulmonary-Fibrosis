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

gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)

# ==================================#
# Global variables
CSV_PATH = '../../data/processed/train.csv'

# ==================================#

dcm_inp = Input(shape=(512, 512, 40))
dcm_hid = Conv2D(32, 3, activation='relu', padding='same')(dcm_inp)
dcm_hid = MaxPool2D(2, 2)(dcm_hid)
dcm_hid = Conv2D(32, 3, activation='relu', padding='same')(dcm_hid)
dcm_hid = MaxPool2D(2, 2)(dcm_hid)
dcm_hid = Conv2D(64, 3, activation='relu', padding='same')(dcm_hid)
dcm_hid = MaxPool2D(2, 2)(dcm_hid)
dcm_hid = Flatten()(dcm_hid)
dcm_hid = Dense(256, activation='relu')(dcm_hid)  # 256

csv_inp = Input(shape=(7,))
csv_hid = Dense(16, activation='relu')(csv_inp)

conc = Concatenate()([dcm_hid, csv_hid])
conc_dense = Dense(512, activation='relu')(conc)  # 128
conc_dense = Dense(256, activation='relu')(conc_dense)  # 128
out = Dense(146)(conc_dense)

# out = Reshape((8, 2))(conc_dense)

inputs = [csv_inp, dcm_inp]
model = Model(inputs, out)

# ==================================#

from keras.losses import mean_squared_error, binary_crossentropy
import tensorflow.keras.backend as K


# ==================================#

def my_custom_loss_wrapper(patient_record):  # patient_records - 146 len 1D array with 1s and 0s
    # patient_record = tf.cast(patient_record, tf.float32)
    patient_record = K.variable(0, )

    def my_custom_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        mse = K.mean(K.square(y_true - patient_record * y_pred), axis=-1)

        # mse = patient_records*mean_squared_error(y_true, y_pred)
        return mse

    return my_custom_loss


current_batch = K.variable(0.)


# ==================================#
class PerBatchHistory(keras.callbacks.Callback):
    def __init__(self):
        self.losses = []
        self.val_losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))


class DynamicLossCallback(keras.callbacks.Callback):

    def __init__(self, current_batch):
        self.current_batch = current_batch

    def on_batch_end(self, batch, logs=None):
        K.set_value(self.current_batch, batch)


# Callbacks to variables
loss_callback = DynamicLossCallback(current_batch)
history = PerBatchHistory()

callbacks = [loss_callback, history]


# ==================================#
# Creating datagen
def merge_datagens(csv_gen, dcm_gen):
    global patient_record1
    csv_flow = csv_gen.flow()
    dcm_flow = dcm_gen.flow()
    while True:
        csv_outp = next(csv_flow)
        dcm_outp = next(dcm_flow)
        patient_record1 = csv_outp[1][:, :, 1]
        yield [csv_outp[0], dcm_outp[0]], csv_outp[1][:, :, 0]
        # Without dicom images
        # yield [csv_outp[0], np.zeros(shape=dcm_outp[0].shape)], csv_outp[1]

        # Without "weeks"
        # yield [csv_outp[0][:,1:], dcm_outp[0]], csv_outp[1][:,:,1]


def create_datagen(batch_size=1):
    csv_datagen = CsvDataGenerator(CSV_PATH, normalize=True, batch_size=batch_size)
    dcm_datagen = DcmDataGenerator('../../data/processed/train', batch_size=batch_size)

    merged_gen = merge_datagens(csv_datagen, dcm_datagen)

    return merged_gen


# ==============================='===#
# Creating patient_records (1s and 0s for each patient)

patients_df = pd.read_csv(CSV_PATH)
patient_ids = patients_df['Patient'].unique()
num_of_patients = len(patient_ids)

patient_records = np.empty(shape=(num_of_patients, 146), dtype=np.float32)
patient_record1 = np.empty(shape=(146,), dtype=np.float32)

for j, ID in enumerate(patient_ids):
    patient_csv = patients_df[patients_df['Patient'] == ID]
    patient_csv.reset_index(inplace=True)

    weeks_FVC = patient_csv.loc[1:, ['Weeks', 'FVC']]
    for i in range(-12, 134):
        if not np.any(weeks_FVC['Weeks'] == i):
            weeks_FVC = weeks_FVC.append({'Weeks': i, 'FVC': 0}, ignore_index=True)

    weeks_FVC.sort_values('Weeks', inplace=True)
    weeks_FVC.drop(columns='Weeks', inplace=True)

    patient_records[j,] = weeks_FVC.applymap(lambda x: 0 if x == 0 else 1).values.flatten()


# ==================================#

def compile_fit(datagen, batch_size=1, to_fit=True):
    model.compile(optimizer='adam', loss=my_custom_loss_wrapper(patient_records[int(current_batch)]), metrics=['mae'],
                  experimental_run_tf_function=False)
    # model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    if to_fit:
        model.fit(datagen, epochs=1, steps_per_epoch=num_of_patients / batch_size, callbacks=callbacks)

    return model


# ==================================#
# Testing

fit_gen = create_datagen(batch_size=1)
model = compile_fit(fit_gen, 1, True)
[csv_X, dcm_X], y = next(fit_gen)

# y_pred = model.predict([csv_X, dcm_X], steps=1)
# plt.figure(figsize=(14, 14))
# plt.plot(y_pred[0])
# plt.plot(y[0])

# plt.figure(figsize=(14, 14))
# plt.plot(history.losses)
