import unittest

import torch

from weatherlearn.models.fuxi.fuxi import DownBlock


class TestDownBlock(unittest.TestCase):
    def test_shape(self):
        x = torch.randn(1, 2, 180, 360)
        db = DownBlock(2, 2, 1)
        output = db(x)
        self.assertEqual(output.shape, (1, 2, 90, 180))

