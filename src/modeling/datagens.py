import numpy as np
from tensorflow import keras
import pandas as pd
import os


class DcmDataGenerator(keras.utils.Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """

    def __init__(self, images_path, dim=(15, 512, 512), window=None):
        """Initialization
        :param images_path: path to images location
        :param dim: tuple indicating image dimension in format CHW
        """
        self.list_IDs = os.listdir(images_path)
        self.images_path = images_path
        self.dim = dim
        self.indexes = np.arange(len(self.list_IDs))
        self.on_epoch_end()
        self.window = window

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return len(self.list_IDs)

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_IDs))

    def flow(self, seed):
        np.random.seed(seed)
        i = int(np.random.randint(0, self.__len__(), size=(1,)))
        while True:
            yield self.__getitem__(i % self.__len__())
            i += 1

    def __getitem__(self, index):
        """Generate one patient's data
        :param index: index of the patient
        :return: X_dcm
        """

        # Find list of IDs
        patient_ID = self.list_IDs[index]

        # Generate data
        X_dcm, X_num = self._generate_X(patient_ID)

        return X_dcm, X_num

    def _generate_X_num(self, patient_ID):
        """Generates order number of patient's images
        :param patient_ID: ID of the patient
        :return: order number of patient's images
        """

    def _generate_X(self, patient_ID):
        """Generates data containing patient's images
        :param patient_ID: ID of the patient
        :return: patient's images
        """
        # Initialization
        X_dcm = np.empty((1, *self.dim), dtype=np.float32)
        X_num = np.empty((1, self.dim[0]), dtype=np.float32)

        dcm_names = os.listdir(path=os.path.join(self.images_path, patient_ID))

        # Generate data
        for j, dcm_name in enumerate(dcm_names):
            X_dcm[0, j], X_num[0] = self._load_dcm_num(os.path.join(self.images_path, patient_ID, dcm_name))

        X_dcm = np.moveaxis(X_dcm, 1, -1)

        return X_dcm, X_num

    def _load_dcm_num(self, image_path):
        """Load grayscale image
        :param image_path: path to image to load
        :return: loaded image
        """
        img, num = np.load(image_path, allow_pickle=True)

        if self.window:
            lb = self.window[0]
            ub = self.window[1]
            img[img < lb] = lb
            img[img > ub] = ub
            img = (img - lb) / (ub - lb)

        return img, num


class CsvDataGenerator(keras.utils.Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """

    def __init__(self, csv_path, to_fit=True, to_normalize=True):
        """Initialization
        :param to_normalize: True to normalize, False otherwise
        :param csv_path: path to csv file location
        :param to_fit: True to return X and y, False to return X only
        """
        self.to_normalize = to_normalize
        self.list_IDs = os.listdir(csv_path[:-4])
        self.csv_path = csv_path
        self.to_fit = to_fit
        self.indexes = np.arange(len(self.list_IDs))
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return len(self.list_IDs)

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_IDs))

    def flow(self, seed):
        np.random.seed(seed)
        i = int(np.random.randint(0, self.__len__(), size=(1,)))
        while True:
            yield self.__getitem__(i % self.__len__())
            i += 1

    def __getitem__(self, index):
        """Generate one patient's data
        :param index: index of the patient
        :return: X
        """

        # Find list of IDs
        patient_ID = self.list_IDs[index]

        # Generate data
        X = self._generate_X(patient_ID)

        if self.to_fit:
            y = self._generate_y(patient_ID)
            return X, y
        else:
            return X

    def _generate_X(self, patient_ID):
        """Generates data containing patient's first csv record
        :param patient_ID: ID of the patient
        :return: patient's first csv record
        """
        X = np.empty(shape=(1, 7), dtype=np.float32)
        # Generate data
        X[0] = self._load_X(self.csv_path, patient_ID)

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

    def _generate_y(self, patient_ID):
        """Generates data containing patient's [1:] csv records
        :param patient_ID: ID of the patient
        :return: patient's [1:] csv records
        """
        y = np.empty(shape=(1, 146, 2), dtype=np.float32)
        # Generate data
        y[0] = self._load_y(self.csv_path, patient_ID)

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
        weeks_FVC = weeks_FVC[~weeks_FVC.duplicated(['Weeks'])]

        # weeks_FVC.drop()
        weeks_FVC = self.pad_y(weeks_FVC)
        weeks_FVC = weeks_FVC.to_numpy()

        return weeks_FVC

    def pad_y(self, csv_df):
        csv_df['isRecord'] = 1
        for i in range(-12, 134):
            if not np.any(csv_df['Weeks'] == i):
                csv_df = csv_df.append({'Weeks': i, 'FVC': 0, 'isRecord': 0}, ignore_index=True)

        csv_df.sort_values('Weeks', inplace=True)
        csv_df.drop(columns='Weeks', inplace=True)
        if self.to_normalize:
            csv_df.loc[:, 'FVC'] = (csv_df.loc[:, 'FVC'] - (-5)) / (4916 - (-5))
        return csv_df


# ==================================#
# Creating datagen
def _merge_datagens(csv_gen, dcm_gen, shuffle=True):
    seed = 0
    while True:
        csv_flow = csv_gen.flow(seed)
        dcm_flow = dcm_gen.flow(seed)
        patient_num = 1
        while True:
            csv_data = next(csv_flow)
            dcm_data = next(dcm_flow)
            csv_X = csv_data[0]
            dcm_X_img = dcm_data[0]
            dcm_X_num = dcm_data[1]
            csv_y = csv_data[1][:, :, 0]
            csv_is_patient_record = csv_data[1][:, :, 1]
            yield [csv_X, dcm_X_img, dcm_X_num], csv_y, csv_is_patient_record

            patient_num += 1
            if patient_num > 175:
                break
        if shuffle:
            seed += 1


def create_datagen(shuffle=True):
    """Returns generator that yields [csv_X, dcm_X_img, dcm_X_num], csv_y, csv_is_patient_record"""
    csv_datagen = CsvDataGenerator('../../data/processed/train.csv', to_normalize=True)
    dcm_datagen = DcmDataGenerator('../../data/processed/train')

    merged_gen = _merge_datagens(csv_datagen, dcm_datagen, shuffle=shuffle)

    return merged_gen

# def gen_train_test_split(datagen):
#     datagen.


# gen = create_datagen(shuffle=True)
# x1, y1, is_p_r1 = next(gen)
