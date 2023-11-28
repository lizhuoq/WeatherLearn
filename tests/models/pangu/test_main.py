import torch

from WeatherLearn.models.pangu.pangu import EarthAttention3D, UpSample, DownSample

import unittest

class TestMain(unittest.TestCase):
    def test_downsample(self):
        in_dim = 1
        input_resolution = (8, 181, 360)
        output_resolution = (8, 91, 180)
        x = torch.randn(1, 8 * 181 * 360, 1)
        downsample = DownSample(in_dim, input_resolution, output_resolution)
        x_downsample = downsample(x)
        self.assertEqual(x_downsample.shape, (1, 8 * 91 * 180, 2))

    def test_upsample(self):
        in_dim = 2
        out_dim = in_dim // 2
        input_resolution = (8, 91, 180)
        output_resolution = (8, 181, 360)
        upsample = UpSample(in_dim, out_dim, input_resolution, output_resolution)
        x = torch.randn(1, 8 * 91 * 180, 2)
        x_upsample = upsample(x)
        self.assertEqual(x_upsample.shape, (1, 8 * 181 * 360, 1))


