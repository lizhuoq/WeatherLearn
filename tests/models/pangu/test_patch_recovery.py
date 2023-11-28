import unittest

import torch

from WeatherLearn.models.pangu.utils.patch_recovery import PatchRecovery3D, PatchRecovery2D


class TestPatchRecovery(unittest.TestCase):
    def test_patch_recovery2d_pangu(self):
        surface = torch.randn(1, 1, 181, 360)
        patchrecovery2d = PatchRecovery2D(
            img_size=(721, 1440),
            patch_size=(4, 4),
            in_chans=1,
            out_chans=4
        )
        surface_output = patchrecovery2d(surface)
        self.assertEqual(surface_output.shape, (1, 4, 721, 1440))

    def test_patch_recovery3d_pangu(self):
        upper_air = torch.randn(1, 1, 7, 181, 360)
        patchrecovery3d = PatchRecovery3D(
            img_size=(7, 721, 1440),
            patch_size=(2, 4, 4),
            in_chans=1,
            out_chans=5
        )
        upper_air_output = patchrecovery3d(upper_air)
        self.assertEqual(upper_air_output.shape, (1, 5, 7, 721, 1440))
        