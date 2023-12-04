import unittest

import torch

from weatherlearn.models import Fuxi


class TestFuxi(unittest.TestCase):
    def test_shape(self):
        in_chans = out_chans = 1
        embed_dim = 1
        x = torch.randn(1, in_chans, 2, 721, 1440)
        fuxi = Fuxi(in_chans=in_chans, out_chans=out_chans, embed_dim=embed_dim, num_groups=1, num_heads=1)
        output = fuxi(x)
        self.assertEqual(output.shape, (1, out_chans, 721, 1440))
