from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow import keras
import tensorflow.keras.backend as K
import tensorflow as tf


class DcmCsvModel(Model):
    def __init__(self):
        super(DcmCsvModel, self).__init__()

        # Dicom images analysis
        self.dcm_conv1 = Conv2D(64, 3, activation='relu', padding='same')
        self.dcm_conv2 = Conv2D(64, 3, activation='relu', padding='same')
        self.dcm_maxpool1 = MaxPool2D(4, 4)
        self.dcm_conv3 = Conv2D(128, 3, activation='relu', padding='same')
        self.dcm_conv4 = Conv2D(128, 2, activation='relu', padding='same')
        self.dcm_maxpool2 = MaxPool2D(4, 4)
        self.dcm_flatten = Flatten()
        self.dcm_dense1 = Dense(1024, activation='relu')
        self.dcm_dense2 = Dense(128, activation='relu')

        # Csv data analysis
        self.csv_dense1 = Dense(256, activation='relu')
        self.csv_dense2 = Dense(512, activation='relu')
        self.csv_dense3 = Dense(256, activation='relu')

        # Both csv and dcm data analysis
        self.conc1 = Concatenate()  # concats last csv_dense and dcm_dense outputs
        self.conc_dense1 = Dense(512, activation='relu')
        self.conc_dense2 = Dense(1024, activation='relu')

        self.out = Dense(146)

    def call(self, inputs, **kwargs):
        csv_inp, dcm_inp = inputs

        # Dcm part
        dcm_x = self.dcm_conv1(dcm_inp)
        dcm_x = self.dcm_conv2(dcm_x)
        dcm_x = self.dcm_maxpool1(dcm_x)

        dcm_x = self.dcm_conv3(dcm_x)
        dcm_x = self.dcm_conv4(dcm_x)
        dcm_x = self.dcm_maxpool2(dcm_x)

        dcm_x = self.dcm_flatten(dcm_x)

        dcm_x = self.dcm_dense1(dcm_x)
        dcm_x = self.dcm_dense2(dcm_x)

        # Csv part
        csv_x = self.csv_dense1(csv_inp)
        csv_x = self.csv_dense2(csv_x)
        csv_x = self.csv_dense3(csv_x)

        # Conc part
        conc_x = self.conc1([csv_x, dcm_x])
        conc_x = self.conc_dense1(conc_x)
        conc_x = self.conc_dense2(conc_x)

        return self.out(conc_x)


def SSIMLoss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))


def clustering_autoencoder():  # For clustering dcm images
    encoded_dims = 256

    inputs = Input(shape=(512, 512, 1))
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPool2D(8, 8)(x)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPool2D(4, 4)(x)

    x = Conv2D(16, (5, 5), activation='relu', padding='same')(x)
    x = Conv2D(4, (7, 7), activation='relu', padding='same')(x)
    x = Conv2D(1, (9, 9), activation='relu', padding='same')(x)
    x = Flatten()(x)
    encoded = Dense(encoded_dims, activation='relu')(x)

    encoder = Model(inputs=inputs, outputs=encoded)

    encoded_inputs = Input(shape=(encoded_dims,))
    x = Dense(encoded_dims, activation='relu')(encoded_inputs)
    x = Reshape((16, 16, 1))(x)
    x = Conv2D(4, (5, 5), activation='relu', padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D(4)(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D(8)(x)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    decoder = Model(inputs=encoded_inputs, outputs=decoded)

    x = encoder(inputs)
    x = decoder(x)

    autoencoder = Model(inputs=inputs, outputs=x)

    autoencoder.compile(optimizer=keras.optimizers.Adam(lr=0.003), loss=SSIMLoss)

    return autoencoder
