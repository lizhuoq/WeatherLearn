import unittest

import torch

from weatherlearn.models.fuxi.fuxi import UTransformer


class TestUTransformer(unittest.TestCase):
    def test_shape(self):
        input = torch.randn(1, 2, 180, 360)
        utransformer = UTransformer(2, 1, [90, 180], 1, 7)
        output = utransformer(input)
        self.assertEqual(output.shape, input.shape)
