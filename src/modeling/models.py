from tensorflow.keras.layers import (
    Dense, Conv2D, MaxPool2D, SeparableConv2D, Reshape,
    Concatenate, Flatten, Dropout, Input, SeparableConv2D, UpSampling2D,
)
from tensorflow.keras import Model
from tensorflow import keras


class DcmCsvModel(Model):
    def __init__(self):
        super(DcmCsvModel, self).__init__()

        self.csv_hid1 = Dense(4, activation='relu')

        self.dcm_conv1 = SeparableConv2D(32, 3, activation='relu', padding='same')
        self.dcm_conv2 = SeparableConv2D(64, 3, activation='relu', padding='same')
        self.dcm_maxpool1 = MaxPool2D(4, 4)
        self.dcm_conv3 = SeparableConv2D(64, 3, activation='relu', padding='same')
        self.dcm_conv4 = SeparableConv2D(64, 2, activation='relu', padding='same')
        self.dcm_maxpool2 = MaxPool2D(4, 4)
        self.dcm_conv5 = SeparableConv2D(64, 2, activation='relu', padding='same')
        self.dcm_conv6 = Conv2D(32, 2, activation='relu', padding='same')
        self.dcm_maxpool3 = MaxPool2D(4, 4)
        self.dcm_conv7 = Conv2D(16, 2, activation='relu', padding='same')
        self.dcm_conv8 = Conv2D(4, 2, activation='relu', padding='same')
        self.dcm_maxpool4 = MaxPool2D(2, 2)
        self.dcm_conv9 = Conv2D(1, 2, padding='same')
        self.dcm_flatten = Flatten()
        self.dcm_conc = Concatenate()  # concats dcm_num_inp and dcm_flatten
        self.dcm_dense1 = Dense(8, activation='relu')
        self.dcm_dense2 = Dense(4)

        self.conc1 = Concatenate()  # concats csv_hid1 and dcm_dense
        self.conc_dense1 = Dense(8, activation='relu')
        # self.conc_drop = Dropout(0.25)
        self.conc_dense2 = Dense(4, activation='relu')
        self.conc2 = Concatenate()  # concats csv_hid1 and conc_dense2
        self.out = Dense(146, activation='relu')

    def call(self, inputs, **kwargs):
        csv_inp, dcm_inp, dcm_num_inp = inputs

        csv_x = self.csv_hid1(csv_inp)

        dcm_x = self.dcm_conv1(dcm_inp)
        dcm_x = self.dcm_conv2(dcm_x)
        dcm_x = self.dcm_maxpool1(dcm_x)
        dcm_x = self.dcm_conv3(dcm_x)
        dcm_x = self.dcm_conv4(dcm_x)
        dcm_x = self.dcm_maxpool2(dcm_x)
        dcm_x = self.dcm_conv5(dcm_x)
        dcm_x = self.dcm_conv6(dcm_x)
        dcm_x = self.dcm_maxpool3(dcm_x)
        dcm_x = self.dcm_conv7(dcm_x)
        dcm_x = self.dcm_conv8(dcm_x)
        dcm_x = self.dcm_maxpool4(dcm_x)
        dcm_x = self.dcm_conv9(dcm_x)
        dcm_x = self.dcm_flatten(dcm_x)
        dcm_x_conc = self.dcm_conc([dcm_num_inp, dcm_x])
        dcm_x = self.dcm_dense1(dcm_x_conc)
        dcm_x = self.dcm_dense2(dcm_x)

        conc_x = self.conc1([csv_x, dcm_x])
        conc_x = self.conc_dense1(conc_x)
        # conc_x = self.conc_drop(conc_x)
        conc_x = self.conc_dense2(conc_x)
        conc_x = self.conc2([csv_x, conc_x])
        return self.out(conc_x)


def clustering_autoencoder():  # Deprecated.
    encoded_dims = 64

    inputs = Input(shape=(512, 512, 1))
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPool2D(8, 8)(x)

    x = SeparableConv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (2, 2), activation='relu', padding='same')(x)
    x = MaxPool2D(8, 8)(x)

    x = Conv2D(16, (2, 2), activation='relu', padding='same')(x)
    x = Conv2D(4, (2, 2), activation='relu', padding='same')(x)
    x = Conv2D(1, (2, 2), padding='same')(x)
    x = Flatten()(x)
    x = Dense(encoded_dims)(x)
    encoded = Dense(10, activation='sigmoid')(x)

    encoder = Model(inputs=inputs, outputs=encoded)

    encoded_inputs = Input(shape=(10,))
    x = Dense(encoded_dims)(encoded_inputs)
    x = Reshape((8, 8, 1))(x)
    x = Conv2D(4, (2, 2), padding='same')(x)
    x = Conv2D(16, (2, 2), activation='relu', padding='same')(x)
    x = UpSampling2D(8)(x)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = SeparableConv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D(8)(x)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    decoder = Model(inputs=encoded_inputs, outputs=decoded)

    x = encoder(inputs)
    x = decoder(x)

    autoencoder = Model(inputs=inputs, outputs=x)

    autoencoder.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='binary_crossentropy')

    return autoencoder, encoder, decoder

