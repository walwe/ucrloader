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

    def load(self, name):
        """
        Load dataset from disk
        :param name: name of dataset
        :return: UCRData object
        """
        try:
            return UCRData.from_path(self._data_dirs[name])
        except KeyError:
            logging.error("Data{} not available in {}".format(name, self._path))


class UCRData(object):
    """Single UCR Dataset"""
    def __init__(self, name: str, train_data, test_data):
        """
        :param name: Name of dataset
        :param train_data: train data array
        :param test_data: test data array
        """
        self.name = name
        self._train_data = train_data
        self._test_data = test_data

    @classmethod
    def from_path(cls, path: str):
        """
        Load single dataset add given path
        :param path: Path to single dataset
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

        return cls(name, train_data, test_data)

    @staticmethod
    def _read_file(file_name):
        df = pd.read_csv(file_name, header=None)
        return df

    @property
    def train_data(self):
        """Train data as numpy array"""
        return self._train_data.values[:, 1:]

    @property
    def train_labels(self):
        """Train labels as numpy array"""
        return np.array(list(
            map(int, self._train_data.values[:, 0])
        ))

    @property
    def test_data(self):
        """Test data as numpy array"""
        return self._test_data.values[:, 1:]

    @property
    def test_labels(self):
        """Test labels as numpy array"""
        return np.array(list(
            map(int, self._test_data.values[:, 0])
        ))
