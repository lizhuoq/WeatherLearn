import unittest

import torch
from torch import nn

from WeatherLearn.models.layers.patch_embed import PatchEmbed2D, PatchEmbed3D


class PatchEmbed2DTest(unittest.TestCase):
    def test_patch_embed2d_with_pad(self):
        img = torch.rand(2, 4, 7, 7)
        pe_2d = PatchEmbed2D((7, 7), (2, 2), 4, 8)
        out = pe_2d(img)
        self.assertEqual(out.shape, (2, 8, 4, 4))

    def test_patch_embed2d_with_norm(self):
        img = torch.rand(2, 4, 7, 7)
        pe_2d = PatchEmbed2D((7, 7), (2, 2), 4, 8, nn.LayerNorm)
        out = pe_2d(img)
        self.assertEqual(out.shape, (2, 8, 4, 4))

    def test_patch_embed2d_without_pad(self):
        img = torch.rand(2, 4, 8, 8)
        pe_2d = PatchEmbed2D((8, 8), (2, 2), 4, 8)
        out = pe_2d(img)
        self.assertEqual(out.shape, (2, 8, 4, 4))

    def test_patch_embed2d_single(self):
        img_1 = torch.rand(2, 4, 7, 8)
        pe_2d_1 = PatchEmbed2D((7, 8), (2, 2), 4, 8)
        out_1 = pe_2d_1(img_1)
        self.assertEqual(out_1.shape, (2, 8, 4, 4))

        img_2 = torch.rand(2, 4, 7, 8)
        pe_2d_2 = PatchEmbed2D((7, 8), (3, 2), 4, 8)
        out_2 = pe_2d_2(img_2)
        self.assertEqual(out_2.shape, (2, 8, 3, 4))

    def test_patch_embed3d_with_pad(self):
        img = torch.rand(2, 4, 3, 7, 7)
        pe_3d = PatchEmbed3D((3, 7, 7), (2, 2, 2), 4, 8)
        out = pe_3d(img)
        self.assertEqual(out.shape, (2, 8, 2, 4, 4))

    def test_patch_embed3d_with_norm(self):
        img = torch.rand(2, 4, 3, 7, 7)
        pe_3d = PatchEmbed3D((3, 7, 7), (2, 2, 2), 4, 8, nn.LayerNorm)
        out = pe_3d(img)
        self.assertEqual(out.shape, (2, 8, 2, 4, 4))

    def test_patch_embed3d_without_pad(self):
        img = torch.rand(2, 4, 2, 8, 8)
        pe_3d = PatchEmbed3D((2, 8, 8), (1, 2, 2), 4, 8)
        out = pe_3d(img)
        self.assertEqual(out.shape, (2, 8, 2, 4, 4))

    def test_patch_embed3d_single(self):
        img_1 = torch.rand(2, 4, 3, 7, 8)
        pe_3d_1 = PatchEmbed3D((3, 7, 8), (2, 2, 2), 4, 8)
        out_1 = pe_3d_1(img_1)
        self.assertEqual(out_1.shape, (2, 8, 2, 4, 4))

        img_2 = torch.rand(2, 4, 3, 7, 8)
        pe_3d_1 = PatchEmbed3D((3, 7, 8), (2, 3, 2), 4, 8)
        out_2 = pe_3d_1(img_2)
        self.assertEqual(out_2.shape, (2, 8, 2, 3, 4))

        img_3 = torch.rand(2, 4, 3, 7, 8)
        pe_3d_3 = PatchEmbed3D((3, 7, 8), (2, 3, 3), 4, 8)
        out_3 = pe_3d_3(img_3)
        self.assertEqual(out_3.shape, (2, 8, 2, 3, 3))
