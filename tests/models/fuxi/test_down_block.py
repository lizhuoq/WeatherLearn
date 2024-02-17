import unittest

import torch

from weatherlearn.models.fuxi.fuxi import DownBlock


class TestDownBlock(unittest.TestCase):
    # w, h -> int(w / 2), int(h / 2)
    def test_shape(self):
        x = torch.randn(1, 2, 180, 360)
        db = DownBlock(2, 2, 1)
        output = db(x)
        self.assertEqual(output.shape, (1, 2, 90, 180))

    def test_shape_odd(self):
        x = torch.randn(1, 2, 11, 21)
        db = DownBlock(2, 2, 1)
        output = db(x)
        self.assertEqual(output.shape, (1, 2, 5, 10))
       