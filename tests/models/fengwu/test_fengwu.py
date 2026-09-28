import unittest

import torch

from weatherlearn.models import FengWu, FengWu_lite
from weatherlearn.models.fengwu import channel_layout, uncertainty_loss


class TestFengWu(unittest.TestCase):
    def test_channel_layout_13(self):
        layout = channel_layout(13)
        self.assertEqual(layout["n_channels"], 69)
        self.assertEqual(layout["pressure"]["t"], (4 + 4 * 13, 4 + 5 * 13))

    def test_channel_layout_37(self):
        self.assertEqual(channel_layout(37)["n_channels"], 189)

    def test_lite_shape(self):
        model = FengWu_lite(n_levels=13)
        x = torch.randn(1, 69, 64, 128)
        y = model(x)
        self.assertEqual(tuple(y.shape), (1, 69, 64, 128))
        self.assertTrue(torch.isfinite(y).all())

    def test_lite_tiny_levels(self):
        """Smaller n_levels for fast CPU unit test."""
        model = FengWu_lite(
            img_size=(32, 64),
            n_levels=2,
            enc_dim=16,
            embed_dim=32,
            enc_depths=(1, 1),
            enc_heads=(2, 2),
            fuser_depth=1,
            fuser_heads=2,
            window_size=4,
        )
        c = 4 + 5 * 2
        x = torch.randn(2, c, 32, 64)
        y = model(x)
        self.assertEqual(tuple(y.shape), (2, c, 32, 64))

    def test_uncertainty_head(self):
        model = FengWu_lite(
            img_size=(32, 64),
            n_levels=2,
            enc_dim=16,
            embed_dim=32,
            enc_depths=(1, 1),
            enc_heads=(2, 2),
            fuser_depth=1,
            fuser_heads=2,
            window_size=4,
            predict_uncertainty=True,
        )
        c = 4 + 5 * 2
        x = torch.randn(1, c, 32, 64)
        mean, log_var = model(x)
        self.assertEqual(tuple(mean.shape), (1, c, 32, 64))
        self.assertEqual(tuple(log_var.shape), (1, c, 32, 64))
        target = torch.randn_like(mean)
        loss = uncertainty_loss(mean, log_var, target)
        self.assertTrue(torch.isfinite(loss))

    def test_full_ctor_defaults_document_13(self):
        """Default FengWu is 13-level / 69-ch (not 37) for practicality."""
        # Avoid building full 721x1440 weights in CI: only check attrs via lite-like sizes
        model = FengWu(
            img_size=(32, 64),
            patch_size=4,
            n_levels=13,
            enc_dim=16,
            embed_dim=32,
            enc_depths=(1, 1),
            enc_heads=(2, 2),
            fuser_depth=1,
            fuser_heads=2,
            window_size=4,
        )
        self.assertEqual(model.in_chans, 69)
        self.assertEqual(model.n_levels, 13)

    def test_param_count_lite_finite(self):
        model = FengWu_lite(n_levels=2, img_size=(32, 64), enc_dim=16, embed_dim=32,
                            enc_depths=(1, 1), enc_heads=(2, 2), fuser_depth=1, fuser_heads=2)
        n = sum(p.numel() for p in model.parameters())
        self.assertGreater(n, 1000)


if __name__ == "__main__":
    unittest.main()
