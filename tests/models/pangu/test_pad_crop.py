import torch
from torch import nn

import unittest

from WeatherLearn.models.pangu.utils.pad import get_pad3d, get_pad2d
from WeatherLearn.models.pangu.utils.crop import crop2d, crop3d


class TestPad(unittest.TestCase):
    def test_pad2d(self):
        input_resolution = (181, 360)
        window_size = (6, 12)
        padding = get_pad2d(input_resolution, window_size)
        pad = nn.ZeroPad2d(padding)
        x = torch.randn(1, 1, *input_resolution)
        x_pad = pad(x)
        self.assertEqual(x_pad.shape, (1, 1, 186, 360))

    def test_pad3d(self):
        input_resolution = (8, 181, 360)
        window_size = (2, 6, 12)
        padding = get_pad3d(input_resolution, window_size)
        pad = nn.ZeroPad3d(padding)
        x = torch.randn(1, 1, *input_resolution)
        x_pad = pad(x)
        self.assertEqual(x_pad.shape, (1, 1, 8, 186, 360))


class TestCrop(unittest.TestCase):
    def test_crop2d(self):
        x = torch.randn(1, 1, 186, 360)
        resolution = (181, 360)
        crop_x = crop2d(x, resolution)
        self.assertEqual(crop_x.shape, (1, 1, 181, 360))

    def test_crop3d(self):
        x = torch.randn(1, 1, 8, 186, 360)
        resolution = (8, 181, 360)
        crop_x = crop3d(x, resolution)
        self.assertEqual(crop_x.shape, (1, 1, 8, 181, 360))


class TestPadCrop(unittest.TestCase):
    def test_pad2d_crop2d(self):
        input_resolution = (181, 360)
        window_size = (6, 12)
        padding = get_pad2d(input_resolution, window_size)
        pad = nn.ZeroPad2d(padding)
        x = torch.randn(1, 1, *input_resolution)
        x_pad = pad(x)
        crop_x = crop2d(x_pad, input_resolution)
        self.assertTrue((x == crop_x).all())

    def test_pad_crop3d(self):
        input_resolution = (8, 181, 360)
        window_size = (2, 6, 12)
        padding = get_pad3d(input_resolution, window_size)
        pad = nn.ZeroPad3d(padding)
        x = torch.randn(1, 1, *input_resolution)
        x_pad = pad(x)
        crop_x = crop3d(x_pad, input_resolution)
        self.assertTrue((x == crop_x).all())
