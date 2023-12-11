import unittest

import torch

from weatherlearn.models.fuxi.fuxi import UTransformer


class TestUTransformer(unittest.TestCase):
    def test_shape_window_size_int(self):
        input = torch.randn(1, 2, 180, 360)
        utransformer = UTransformer(2, 1, (90, 180), 1, 7, 2)
        output = utransformer(input)
        self.assertEqual(output.shape, input.shape)

    def test_shape_window_size_tuple(self):
        input = torch.randn(1, 2, 180, 360)
        utransformer = UTransformer(2, 1, (90, 180), 1, (4, 8), 2)
        output = utransformer(input)
        self.assertEqual(output.shape, input.shape)
