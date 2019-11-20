from pathlib import Path
from unittest import TestCase, main
from ucrloader import UCRData, UCRLoader
import numpy as np
from tempfile import TemporaryDirectory


class TestUCRLoader(TestCase):
    def test_loader_empty(self):
        d = UCRData("Test", [], [], [], [])
        self.assertListEqual(d.test_labels, [])
        self.assertListEqual(d.train_labels, [])
        self.assertListEqual(d.test_data, [])
        self.assertListEqual(d.train_data, [])
        self.assertEqual(len(d.unique_labels), 0)

    def test_unique_index(self):

        d = UCRData("Test", [], [1, 2, 3], [], [5, 2, 6, 1])
        self.assertEqual(d.unique_labels, [1, 2, 3, 5, 6])
        self.assertEqual(d.test_labels_adjusted.tolist(), [3, 1, 4, 0])
        self.assertEqual(d.train_labels_adjusted.tolist(), [0, 1, 2])

    def test_load_path(self):
        test_data = np.arange(0, 30).reshape(3, 10)
        train_data = np.arange(10, 50).reshape(4, 10)

        name = "dataset"
        with TemporaryDirectory() as d:
            path = Path(d) / name
            path.mkdir()

            test_file = path / "{}_TEST".format(name)
            train_file = path / "{}_TRAIN".format(name)
            np.savetxt(str(test_file), test_data, delimiter=",")
            np.savetxt(str(train_file), train_data, delimiter=",")

            loader = UCRLoader(d)
            self.assertListEqual(loader.names, [name])
            data = loader.load(name)

            np.testing.assert_array_equal(data.test_data, test_data[:, 1:])
            np.testing.assert_array_equal(data.train_data, train_data[:, 1:])

            np.testing.assert_array_equal(data.test_labels, test_data[:, 0])
            np.testing.assert_array_equal(data.train_labels, train_data[:, 0])

    def test_nan_to_num(self):
        test_data = np.arange(0, 30, dtype=float).reshape(3, 10)
        train_data = np.arange(10, 50, dtype=float).reshape(4, 10)

        test_data[:,-1] = np.nan
        train_data[:, 5] = np.nan
        train_data[:, 6] = np.nan

        name = "dataset"
        with TemporaryDirectory() as d:
            path = Path(d) / name
            path.mkdir()

            test_file = path / "{}_TEST".format(name)
            train_file = path / "{}_TRAIN".format(name)
            np.savetxt(str(test_file), test_data, delimiter=",")
            np.savetxt(str(train_file), train_data, delimiter=",")

            loader = UCRLoader(d)
            self.assertListEqual(loader.names, [name])
            data = loader.load(name)

            np.testing.assert_equal(np.any(np.not_equal(data.test_data, test_data[:, 1:])), True)
            np.testing.assert_equal(np.any(np.not_equal(data.train_data, train_data[:, 1:])), True)

            assert not np.isnan(data.train_data).any()
            assert not np.isnan(data.test_data).any()
            assert np.isnan(train_data).any()
            assert np.isnan(test_data).any()


if __name__ == '__main__':
    main()
