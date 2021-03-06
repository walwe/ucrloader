from pathlib import Path
import logging
import pandas as pd
import numpy as np


class UCRLoader(object):
    """UCR dataset loader"""
    def __init__(self, path):
        """
        :param path: Path to the extracted UCR dataset
        """
        self._path = Path(path)
        if not self._path.is_dir():
            logging.error("{} is not a directory.".format(path))
            raise NotADirectoryError("Given UCR directory does not exsits")
        self._data_dirs = {p.name: p for p in sorted(self._path.iterdir()) if p.is_dir()}

    @property
    def names(self):
        """
        Dataset names
        """
        return sorted(self._data_dirs.keys())

    def load(self, name, z_norm=False):
        """
        Load dataset from disk
        :param name: name of dataset
        :return: UCRData object
        """
        try:
            return UCRData.from_path(self._data_dirs[name], z_norm)
        except KeyError:
            logging.error("Data{} not available in {}".format(name, self._path))


class UCRData(object):
    """Single UCR Dataset"""
    def __init__(self, name: str,
                 train_data: np.array, train_labels: np.array,
                 test_data: np.array, test_labels: np.array):
        """
        :param name: Name of dataset
        :param train_data: train data array
        :param train_labels: train label array
        :param test_data: test data array
        :param test_labels: test label array
        """
        self.name = name
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels

        # Calculate unique labels
        self.unique_labels = np.unique(np.concatenate(
            (np.unique(self.train_labels), np.unique(self.test_labels))
        )).tolist()

        self._test_label_adjusted = None
        self._train_label_adjusted = None

    @property
    def test_labels_adjusted(self):
        """
        Adjusted test labels in range [0 - len(labels)-1]
        :return: List of labels
        """
        if self._test_label_adjusted is None:
            self._test_label_adjusted = np.array([self.unique_labels.index(v) for v in self.test_labels])
        return self._test_label_adjusted

    @property
    def train_labels_adjusted(self):
        """
        Adjusted train labels in range [0 - len(labels)-1]
        :return: List of labels
        """
        if self._train_label_adjusted is None:
            self._train_label_adjusted = np.array([self.unique_labels.index(v) for v in self.train_labels])
        return self._train_label_adjusted

    @classmethod
    def from_path(cls, path: str, z_norm=False, nan_padding=True):
        """
        Load single dataset add given path
        :param z_norm: Z-normalize data?
        :param path: Path to single dataset
        :param nan_padding: Replace nan values with zero
        :return: UCRData
        """
        path = Path(path)
        if not path.is_dir():
            logging.error("{} is not a directory. Skipping.".format(path))

        name = path.name
        test_fn = path / "{}_TEST".format(name)
        train_fn = path / "{}_TRAIN".format(name)

        logging.info("Loading {}".format(name))
        train_data = cls._read_file(train_fn)
        test_data = cls._read_file(test_fn)

        train_labels, train_data = cls._split(train_data)
        test_labels, test_data = cls._split(test_data)

        if nan_padding:
            train_data = np.nan_to_num(train_data)
            test_data = np.nan_to_num(test_data)

        if z_norm:
            train_data, test_data = cls._z_norm(train_data, test_data)

        return cls(name, train_data, train_labels, test_data, test_labels)

    @staticmethod
    def _split(data):
        """Split into labels and values. First column are labels"""
        labels = np.array(list(
            map(int, data.values[:, 0])
        ))
        values = data.values[:, 1:]
        return labels, values

    @staticmethod
    def _read_file(file_name):
        df = pd.read_csv(file_name, header=None)
        return df

    @staticmethod
    def _z_norm(train_data, test_data):
        axis = 0
        ddof = 0

        # Calculate mean and std on train data
        mns = np.nanmean(train_data)
        sstd = np.nanstd(train_data)

        def norm(a):
            a = np.asanyarray(a)
            if axis and mns.ndim < a.ndim:
                return ((a - np.expand_dims(mns, axis=axis)) /
                        np.expand_dims(sstd, axis=axis))
            else:
                return (a - mns) / sstd

        return norm(train_data), norm(test_data)

