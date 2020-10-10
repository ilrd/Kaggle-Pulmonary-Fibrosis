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
# Creating the model
class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()

        # self.csv_inp = Input(shape=(7,))
        self.csv_hid1 = Dense(64, activation='relu')

        # self.dcm_inp = Input(shape=(512, 512, 40))
        self.dcm_hid1 = Conv2D(32, 3, activation='relu', padding='same')
        self.dcm_hid2 = MaxPool2D(2, 2)
        # self.dcm_hid = Conv2D(32, 3, activation='relu', padding='same')
        self.dcm_hid3 = MaxPool2D(2, 2)
        # self.dcm_hid = Conv2D(64, 3, activation='relu', padding='same')
        self.dcm_hid4 = MaxPool2D(2, 2)
        self.dcm_hid5 = Flatten()
        self.dcm_hid6 = Dense(1, activation='relu')

        self.conc = Concatenate()
        self.conc_dense1 = Dense(128, activation='relu')
        self.conc_dense2 = Dense(64, activation='relu')
        self.out = Dense(146)

    def call(self, inputs, **kwargs):
        csv_inp, dcm_inp = inputs

        csv_x = self.csv_hid1(csv_inp)

        dcm_x = self.dcm_hid1(dcm_inp)
        dcm_x = self.dcm_hid2(dcm_x)
        # dcm_x = self.dcm_hid(dcm_x)
        dcm_x = self.dcm_hid3(dcm_x)
        # dcm_x = self.dcm_hid(dcm_x)
        dcm_x = self.dcm_hid4(dcm_x)
        dcm_x = self.dcm_hid5(dcm_x)
        dcm_x = self.dcm_hid6(dcm_x)

        conc_x = self.conc([csv_x, dcm_x])
        conc_x = self.conc_dense1(conc_x)
        conc_x = self.conc_dense2(conc_x)
        return self.out(conc_x)


model = MyModel()


# ==================================#

def loss_wrapper(patient_record):  # patient_records - 146 len 1D array with 1s and 0s
    patient_record = tf.cast(patient_record, tf.float32)

    def dynamic_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        # ypred_np = (patient_record * y_pred).numpy()
        # y_np = y_true.numpy()
        mse = K.mean(K.square(y_true - patient_record * y_pred), axis=-1)
        # ==#
        y_pred_sleft = K.concatenate((K.constant([[0]]), y_pred))
        y_pred_sright = K.concatenate((y_pred, K.constant([[0]])))
        y_perc = (K.abs(y_pred_sright / y_pred_sleft - 1)).numpy()[0][1:-1]
        y_perc[y_perc > 1] = 1
        y_perc[0 > y_perc] = 0
        # ==#
        neighbor_difference_penalizer = K.mean(
            K.variable(np.array([K.exp(K.variable(np.array([(i - 1.1) * 10]))) for i in y_perc])))
        print(K.mean(y_pred).numpy(), y_pred.numpy().max(), y_pred.numpy().min())
        cost = mse + neighbor_difference_penalizer
        # cost = mse
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
    csv_datagen = CsvDataGenerator(CSV_PATH, to_normalize=True, batch_size=batch_size)
    dcm_datagen = DcmDataGenerator('../../data/processed/train', batch_size=batch_size)

    merged_gen = merge_datagens(csv_datagen, dcm_datagen)

    return merged_gen


# ==================================#
# Making dynamically-changing loss func

# have to define manually a dict to store all epochs scores


def dynamic_loss_training(model, batch_size=1, epochs=5, num_of_patients=52):
    datagen = create_datagen(batch_size=batch_size)

    history = {}
    history['history'] = {}
    history['history']['loss'] = []
    history['history']['val_loss'] = []
    history['history']['batch_loss'] = []
    history['history']['val_batch_loss'] = []

    # batch_X, batch_y, patient_record = next(datagen)

    # define number of iterations in training and test
    steps_per_epoch = round(num_of_patients / batch_size)
    # test_iter = round(testX.shape[0] / batch_size)
    for epoch in range(1, epochs + 1):

        # train iterations
        loss = 0
        for current_batch in range(steps_per_epoch):
            batch_X, batch_y, patient_record = next(datagen)
            if epoch < 3:
                model.compile(loss=loss_wrapper(patient_record), optimizer=keras.optimizers.Adam(lr=0.01),
                              metrics=['mae'], run_eagerly=True)

            if epoch == 3:
                model.compile(loss=loss_wrapper(patient_record), optimizer=keras.optimizers.Adam(lr=0.001),
                              metrics=['mae'], run_eagerly=True)

            if epoch >= 4:
                model.compile(loss=loss_wrapper(patient_record), optimizer=keras.optimizers.Adam(lr=0.0001),
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
model, history = dynamic_loss_training(model, 1, 4)
plt.figure(figsize=(14, 14))
plt.plot(history['history']["batch_loss"][:])

fit_gen = create_datagen(1)
[csv_X, dcm_X], y, dmat = next(fit_gen)

y_pred = model.predict([csv_X, dcm_X], steps=1)
print(np.mean(y_pred))
# plt.figure(figsize=(14, 14))
# plt.plot(y_pred[0])
# plt.plot(y[0])


plt.figure(figsize=(14, 14))
plt.plot((y_pred)[0])

plt.figure(figsize=(14, 14))
plt.plot((y_pred * dmat)[0])
plt.plot(y[0])
plt.show()

# 0.0004509228480436342
