import unittest

import torch

from weatherlearn.models.fuxi.fuxi import UpBlock


class TestUpBlock(unittest.TestCase):
    def test_shape(self):
        ub = UpBlock(2, 2, 1)
        x = torch.randn(1, 2, 90, 180)
        output = ub(x)
        self.assertEqual(output.shape, (1, 2, 180, 360))
