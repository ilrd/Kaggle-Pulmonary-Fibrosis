from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, BatchNormalization, Concatenate, Flatten, Dropout
from tensorflow.keras import Model


class DcmCsvModel(Model):
    def __init__(self):
        super(DcmCsvModel, self).__init__()

        self.csv_hid1 = Dense(128, activation='relu')

        self.dcm_conv1 = Conv2D(32, 3, activation='relu', padding='same')
        self.norm1 = BatchNormalization()
        self.dcm_maxpool1 = MaxPool2D(2, 2)
        self.dcm_conv2 = Conv2D(32, 3, activation='relu', padding='same')
        self.norm2 = BatchNormalization()
        self.dcm_maxpool2 = MaxPool2D(2, 2)
        self.dcm_conv3 = Conv2D(1, 3, activation='relu', padding='same')
        self.norm3 = BatchNormalization()
        self.dcm_maxpool3 = MaxPool2D(2, 2)
        self.dcm_flatten = Flatten()
        self.dcm_dense = Dense(1, activation='relu')

        self.conc = Concatenate()
        self.conc_dense1 = Dense(256, activation='relu')
        # self.conc_drop = Dropout(0.25)
        self.conc_dense2 = Dense(128, activation='relu')
        self.out = Dense(146)

    def call(self, inputs, **kwargs):
        csv_inp, dcm_inp = inputs

        csv_x = self.csv_hid1(csv_inp)

        dcm_x = self.dcm_conv1(dcm_inp)
        dcm_x = self.norm1(dcm_x)
        dcm_x = self.dcm_maxpool1(dcm_x)
        dcm_x = self.dcm_conv2(dcm_x)
        dcm_x = self.norm2(dcm_x)
        dcm_x = self.dcm_maxpool2(dcm_x)
        dcm_x = self.dcm_conv3(dcm_x)
        dcm_x = self.norm3(dcm_x)
        dcm_x = self.dcm_maxpool3(dcm_x)
        dcm_x = self.dcm_flatten(dcm_x)
        dcm_x = self.dcm_dense(dcm_x)

        conc_x = self.conc([csv_x, dcm_x])
        conc_x = self.conc_dense1(conc_x)
        # conc_x = self.conc_drop(conc_x)
        conc_x = self.conc_dense2(conc_x)
        return self.out(conc_x)

