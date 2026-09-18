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

    def test_depth_configurable(self):
        """depth defaults to 48; smaller depth must still produce correct spatial shape."""
        in_chans = out_chans = 1
        x = torch.randn(1, in_chans, 2, 32, 64)
        fuxi = Fuxi(
            img_size=(2, 32, 64),
            in_chans=in_chans,
            out_chans=out_chans,
            embed_dim=8,
            num_groups=1,
            num_heads=1,
            window_size=4,
            depth=2,
        )
        output = fuxi(x)
        self.assertEqual(output.shape, (1, out_chans, 32, 64))
        # U-Transformer Swin stage depth should match constructor arg
        self.assertEqual(len(fuxi.u_transformer.layer.blocks), 2)

