from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from time import perf_counter
import os
import tensorflow.keras.backend as K
from tensorflow.keras.layers import (
    Dense, Conv2D, MaxPool2D, BatchNormalization,
    Flatten, Reshape, Input, UpSampling2D, SeparableConv2D,
)
from tensorflow.keras import Model

np.random.seed(3)

gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)

encoded_dims = 4096

inputs = Input(shape=(512, 512, 1))
x = Conv2D(32, (3, 3), padding='same')(inputs)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPool2D(2, 2)(x)

x = SeparableConv2D(64, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(64, (2, 2), activation='relu', padding='same')(x)
x = MaxPool2D(2, 2)(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(32, (2, 2), activation='relu', padding='same')(x)
x = MaxPool2D(2, 2)(x)

x = Conv2D(16, (2, 2), activation='relu', padding='same')(x)
x = Conv2D(4, (2, 2), padding='same')(x)
x = Conv2D(1, (2, 2), padding='same')(x)
x = Flatten()(x)
encoded = Dense(encoded_dims)(x)

encoder = Model(inputs=inputs, outputs=encoded)

encoded_inputs = Input(shape=(encoded_dims,))
x = Dense(encoded_dims)(encoded_inputs)
x = Reshape((64, 64, 1))(x)
x = Conv2D(4, (2, 2), padding='same')(x)
x = Conv2D(16, (2, 2), padding='same')(x)
x = UpSampling2D(2)(x)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D(2)(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = SeparableConv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D(2)(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(32, (3, 3), padding='same')(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

decoder = Model(inputs=encoded_inputs, outputs=decoded)

x = encoder(inputs)
x = decoder(x)

autoencoder = Model(inputs=inputs, outputs=x)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')


def get_data(nums=None, window=None, shuffle=True, training=True, full_shuffle=False):
    data_dir = '../../data/clustering/train'

    all_images = np.array(os.listdir(data_dir))
    if full_shuffle:
        np.random.shuffle(all_images)

    if nums:
        img_names = all_images[nums[0]:nums[1]]
    else:
        img_names = all_images

    if training:
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
                yield dcm_img_array
    else:
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
            yield dcm_img_array


# from tensorflow.keras import callbacks
#
# train_callbacks = [
#
# ]


def train(model, datagen, nums, epochs=5):
    history = {}
    history['loss'] = []
    history['val_loss'] = []
    history['batch_loss'] = []
    history['val_batch_loss'] = []

    steps_per_epoch = nums[1] - nums[0] - 1

    for epoch in range(1, epochs + 1):
        loss = 0

        for current_batch in range(1, steps_per_epoch + 1):
            batch_X = next(datagen)

            batch_X = np.reshape(batch_X, newshape=(1, 512, 512, 1))

            begin = perf_counter()
            loss_ = model.train_on_batch(batch_X, batch_X)

            end = perf_counter()

            loss += loss_
            print(f"Epoch {epoch}, step {current_batch}:\n"
                  f"Epoch avg loss: {loss / current_batch}, batch loss: {loss_}")

            history['batch_loss'].append(loss_)

            print(f'batch was processed in {end - begin} seconds\n')

        history['loss'].append(loss / steps_per_epoch)

    return history


def fit_model(model, train_gen, val_gen, nums):
    steps_per_epoch = nums[1] - nums[0] - 1
    train_gen = ((train_sample, train_sample) for train_sample in train_gen)
    val_gen = ((val_sample, val_sample) for val_sample in val_gen)
    history = model.fit(train_gen, batch_size=1, epochs=3, validation_data=val_gen,
                        steps_per_epoch=steps_per_epoch)
    return history


nums = [0, 30]
gen = get_data(nums=nums, window=[-128, 127], full_shuffle=True)
# train_history = train(autoencoder, gen, nums=nums, epochs=20)
val_datagen = get_data(nums=[0, 5], window=[-128, 127], full_shuffle=True, training=False)

train_history = fit_model(autoencoder, gen, val_datagen, nums=nums)
x1 = next(gen)
x1 = np.reshape(x1, newshape=(1, 512, 512, 1))

encoded1 = encoder.predict(x1)
decoded1 = decoder.predict(encoded1)
x1 = np.reshape(x1, newshape=(512, 512))
decoded1 = np.reshape(decoded1, newshape=(512, 512))
plt.figure()
plt.subplot(121)
plt.imshow(x1)
plt.subplot(122)
plt.imshow(decoded1)

plt.figure()
plt.plot(train_history.history['loss'][:])

test_gen = get_data(nums=[0, 200], window=[-128, 127], training=False, full_shuffle=True)

test_loss = 0
for test_sample in test_gen:
    y_true = test_sample
    y_pred = np.reshape(autoencoder.predict(y_true.reshape((1, 512, 512, 1))), (512, 512))
    y_true = test_sample.reshape((512,512))
    test_loss_ = np.sum(tf.losses.binary_crossentropy(y_true, y_pred).numpy())
    test_loss += test_loss_

print(test_loss / 200 / 512, '-test loss')
print(train_history.history['loss'][-1], '-train loss')

test_gen = get_data(nums=[0, 200], window=[-128, 127], training=False, full_shuffle=True)

xt1 = next(test_gen)
xt1 = np.reshape(xt1, newshape=(1, 512, 512, 1))

encodedt1 = encoder.predict(xt1)
decodedt1 = decoder.predict(encodedt1)
xt1 = np.reshape(xt1, newshape=(512, 512))
decodedt1 = np.reshape(decodedt1, newshape=(512, 512))

plt.figure(figsize=(15, 14))
plt.subplot(121)
plt.imshow(xt1)
plt.subplot(122)
plt.imshow(decodedt1)

# test_samples = list(test_gen)

# xt1 = test_samples[9]

#
# encodedt1 = encoder.predict(xt1)
# decodedt1 = decoder.predict(encodedt1)
# xt1 = np.reshape(xt1, newshape=(512, 512))
# decodedt1 = np.reshape(decodedt1, newshape=(512, 512))
#
# plt.figure(figsize=(15,14))
# plt.subplot(121)
# plt.imshow(xt1)
# plt.subplot(122)
# plt.imshow(decodedt1)
