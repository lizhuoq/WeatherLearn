import unittest

import torch

from weatherlearn.models.fuxi.fuxi import CubeEmbedding


class TestCubeEmbedding(unittest.TestCase):
    def test_shape(self):
        input = torch.randn(1, 70, 2, 721, 1440)
        cubeembedding = CubeEmbedding((2, 721, 1440), (2, 4, 4), 70, 10)
        output = cubeembedding(input)
        self.assertEqual(output.shape, (1, 10, 1, 180, 360))

    def test_reshape_transpose(self):
        x = torch.randn(2, 4, 6, 180, 360)
        shortcut = x

        x = x.reshape(2, 4, -1).transpose(1, 2)
        self.assertEqual(x.shape, (2, 180 * 360 * 6, 4))

        x = x.transpose(1, 2).reshape(2, 4, 6, 180, 360)

        self.assertTrue((shortcut == x).all())
