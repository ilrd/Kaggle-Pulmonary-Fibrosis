import numpy as np
from tensorflow import keras
import pandas as pd
import os


class DcmDataGenerator(keras.utils.Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """

    def __init__(self, images_path, batch_size=1, dim=(40, 512, 512), to_normalize=True):
        """Initialization
        :param normalize: True to normalize, False otherwise
        :param images_path: path to images location
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        """
        self.list_IDs = os.listdir(images_path)
        self.images_path = images_path
        self.batch_size = batch_size
        self.dim = dim
        self.indexes = np.arange(len(self.list_IDs))
        self.on_epoch_end()
        self.to_normalize = to_normalize

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_IDs))

    def flow(self):
        i = 0
        while True:
            yield self.__getitem__(i % self.__len__())
            i += 1

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = self._generate_X(list_IDs_temp)

        return X, np.array([1])

    def _generate_X(self, list_IDs_temp):
        """Generates data containing batch_size images
        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        # Initialization
        X = np.empty((self.batch_size, *self.dim), dtype=np.float32)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            j_temp = 0
            for j, img in enumerate(os.listdir(path=os.path.join(self.images_path, ID))):
                X[i, j] = self._load_dcm(os.path.join(self.images_path, ID, img))
                j_temp = j
            for j in range(j_temp, 40):
                X[i, j] = np.zeros(self.dim[1:])

        X = np.moveaxis(X, 1, -1)

        return X

    def _load_dcm(self, image_path):
        """Load grayscale image
        :param image_path: path to image to load
        :return: loaded image
        """
        img = np.load(image_path)
        if self.to_normalize:
            img = (img - (-1110.0)) / (6870.0 - (-1110.0))
            # img = (img - np.min(img)) / (np.max(img) - np.min(img))
        return img


class CsvDataGenerator(keras.utils.Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """

    def __init__(self, csv_path, to_fit=True, batch_size=1, to_normalize=True):
        """Initialization
        :param csv_path: path to csv file location
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        """
        self.to_normalize = to_normalize
        self.list_IDs = os.listdir(csv_path[:-4])
        self.csv_path = csv_path
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.list_IDs))
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_IDs))

    def flow(self):
        i = 0
        while True:
            yield self.__getitem__(i % self.__len__())
            i += 1

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = self._generate_X(list_IDs_temp)

        if self.to_fit:
            y = self._generate_y(list_IDs_temp)
            return X, y
        else:
            return X

    def _generate_X(self, list_IDs_temp):
        """Generates data containing batch_size images
        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        # Initialization
        X = np.empty((self.batch_size, 7), dtype=np.float32)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = self._load_X(self.csv_path, ID)

        if self.to_normalize:
            X = (X - (-5)) / (4916 - (-5))

        return X

    def _load_X(self, csv_path, patient_ID):
        """Load csv with patient's weeks and corresponding FVC
        :param csv_path: path to csv file with weeks and FVC file to load
        :return: loaded csv file with weeks and FVC file to load
        """
        patients_df = pd.read_csv(csv_path)
        patient = patients_df[patients_df['Patient'] == patient_ID]
        patient.reset_index(inplace=True)
        X_columns = ['Weeks', 'FVC', 'Age', 'Ex-smoker', 'Never smoked', 'Currently smokes', 'Sex_n']

        X_patient = patient.loc[0, X_columns].to_numpy()

        return X_patient

    def _generate_y(self, list_IDs_temp):
        """Generates data containing batch_size masks
        :param list_IDs_temp: list of label ids to load
        :return: batch if masks
        """
        y = np.empty((self.batch_size, 146, 2), dtype=np.float32)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            y[i,] = self._load_y(self.csv_path, ID)



        return y

    def _load_y(self, csv_path, patient_ID):
        """Load csv with patient's weeks and corresponding FVC
        :param csv_path: path to csv file with weeks and FVC file to load
        :return: loaded csv file with weeks and FVC file to load
        """
        patients_df = pd.read_csv(csv_path)

        patient = patients_df[patients_df['Patient'] == patient_ID]
        patient.reset_index(inplace=True)
        weeks_FVC = patient.loc[1:, ['Weeks', 'FVC']]
        weeks_FVC = self.pad_y(weeks_FVC)
        weeks_FVC.to_numpy()
        return weeks_FVC.to_numpy()

    def pad_y(self, csv_df):
        csv_df['isRecord'] = 1
        for i in range(-12, 134):
            if not np.any(csv_df['Weeks'] == i):
                csv_df = csv_df.append({'Weeks': i, 'FVC': 0, 'isRecord': 0}, ignore_index=True)

        csv_df.sort_values('Weeks', inplace=True)
        csv_df.drop(columns='Weeks', inplace=True)
        if self.to_normalize:
            csv_df.loc[:,'FVC'] = (csv_df.loc[:,'FVC'] - (-5)) / (4916 - (-5))
        return csv_df
