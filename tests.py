from unittest import TestCase, main
from ucrloader import UCRData


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


if __name__ == '__main__':
    main()
