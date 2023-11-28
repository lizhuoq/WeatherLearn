import torch

import unittest

from WeatherLearn.models.pangu.utils.shift_window_mask import window_partition, window_reverse


class TestWindowPartition(unittest.TestCase):
    def test_shape(self):
        x = torch.randn(1, 8, 186, 360, 1)
        window_size = (2, 6, 12)
        windows = window_partition(x, window_size)
        self.assertEqual(windows.shape, (1 * 30, 4 * 31, 2, 6, 12, 1))


class TestWindowReverse(unittest.TestCase):
    def test_shape(self):
        windows = torch.randn(1 * 30, 4 * 31, 2, 6, 12, 1)
        window_size = (2, 6, 12)
        x = window_reverse(windows, window_size, 8, 186, 360)
        self.assertEqual(x.shape, (1, 8, 186, 360, 1))


class TestWindowPartitionReverse(unittest.TestCase):
    def test_equal(self):
        x = torch.randn(1, 8, 186, 360, 1)
        window_size = (2, 6, 12)
        windows = window_partition(x, window_size)
        x_reverse = window_reverse(windows, window_size, 8, 186, 360)
        self.assertTrue((x == x_reverse).all())
