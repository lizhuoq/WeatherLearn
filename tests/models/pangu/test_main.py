import torch

from WeatherLearn.models.pangu.pangu import EarthAttention3D, UpSample, DownSample, EarthSpecificBlock, BasicLayer
from WeatherLearn.models.pangu.utils.shift_window_mask import get_shift_window_mask
from WeatherLearn.models import Pangu

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

    def test_attention_without_mask1(self):
        input_resolution = (8, 186, 360)
        window_size = (2, 6, 12)
        num_heads = 2
        attention = EarthAttention3D(4, input_resolution, window_size, num_heads)
        batch_size = 2
        x = torch.randn(batch_size * 30, 4 * 31, 2 * 6 * 12, 4)
        attn = attention(x)
        self.assertEqual(attn.shape, x.shape)

    def test_attention_without_mask2(self):
        input_resolution = (8, 96, 180)
        window_size = (2, 6, 12)
        num_heads = 2
        attention = EarthAttention3D(4, input_resolution, window_size, num_heads)
        batch_size = 2
        x = torch.randn(batch_size * 15, 4 * 16, 2 * 6 * 12, 4)
        attn = attention(x)
        self.assertEqual(attn.shape, x.shape)

    def test_attention_with_mask(self):
        input_resolution = (8, 186, 360)
        window_size = (2, 6, 12)
        num_heads = 2
        attention = EarthAttention3D(4, input_resolution, window_size, num_heads)
        batch_size = 2
        x = torch.randn(batch_size * 30, 4 * 31, 2 * 6 * 12, 4)
        mask = get_shift_window_mask(input_resolution, window_size, (1, 3, 6))
        attn = attention(x, mask=mask)
        self.assertEqual(x.shape, attn.shape)

    def test_block_with_shift(self):
        dim = 4
        input_resolution = (8, 181, 360)
        num_heads = 2
        block = EarthSpecificBlock(dim, input_resolution, num_heads)
        batch_size = 1
        x = torch.randn(batch_size, 8 * 181 * 360, 4)
        block_x = block(x)
        self.assertEqual(x.shape, block_x.shape)

    def test_block_without_shift(self):
        dim = 4
        input_resolution = (8, 181, 360)
        num_heads = 2
        block = EarthSpecificBlock(dim, input_resolution, num_heads, shift_size=(0, 0, 0))
        batch_size = 1
        x = torch.randn(batch_size, 8 * 181 * 360, 4)
        block_x = block(x)
        self.assertEqual(x.shape, block_x.shape)

    def test_layer1(self):
        dim = 4
        input_resolution = (8, 181, 360)
        depth = 2
        num_heads = 2
        window_size = (2, 6, 12)
        layer = BasicLayer(dim, input_resolution, depth, num_heads, window_size)
        batch_size = 1
        x = torch.randn(batch_size, 8 * 181 * 360, dim)
        layer_x = layer(x)
        self.assertEqual(layer_x.shape, x.shape)

    def test_layer2(self):
        dim = 4
        input_resolution = (8, 91, 180)
        depth = 6
        num_heads = 2
        window_size = (2, 6, 12)
        layer = BasicLayer(dim, input_resolution, depth, num_heads, window_size)
        batch_size = 1
        x = torch.randn(batch_size, 8 * 91 * 180, dim)
        layer_x = layer(x)
        self.assertEqual(layer_x.shape, x.shape)
