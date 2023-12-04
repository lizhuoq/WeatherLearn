import torch
from torch import nn
from torch.nn import functional as F

from typing import Sequence

from ..swin_transformer.swin_transformer_v2 import BasicLayer
from ..pangu.utils.pad import get_pad2d


class CubeEmbedding(nn.Module):
    """
    Args:
        img_size: T, Lat, Lon
        patch_size: T, Lat, Lon
    """
    def __init__(self, img_size, patch_size, in_chans, embed_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2]]

        self.img_size = img_size
        self.patches_resolution = patches_resolution
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor):
        B, C, T, Lat, Lon = x.shape
        assert T == self.img_size[0] and Lat == self.img_size[1] and Lon == self.img_size[2], \
            f"Input image size ({T}*{Lat}*{Lon}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})."
        x = self.proj(x).reshape(B, self.embed_dim, -1).transpose(1, 2)  # B T*Lat*Lon C
        if self.norm is not None:
            x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, self.embed_dim, *self.patches_resolution)
        return x


class DownBlock(nn.Module):
    def __init__(self, in_chans, out_chans, num_groups, num_residuals=2):
        super().__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=(3, 3), stride=2, padding=1)

        blk = []
        for i in range(num_residuals):
            blk.append(nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1))
            blk.append(nn.GroupNorm(num_groups, out_chans))
            blk.append(nn.SiLU())

        self.b = nn.Sequential(*blk)

    def forward(self, x):
        x = self.conv(x)

        shortcut = x

        x = self.b(x)

        return x + shortcut


class UpBlock(nn.Module):
    def __init__(self, in_chans, out_chans, num_groups, num_residuals=2):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2)

        blk = []
        for i in range(num_residuals):
            blk.append(nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1))
            blk.append(nn.GroupNorm(num_groups, out_chans))
            blk.append(nn.SiLU())

        self.b = nn.Sequential(*blk)

    def forward(self, x):
        x = self.conv(x)

        shortcut = x

        x = self.b(x)

        return x + shortcut


class UTransformer(nn.Module):
    def __init__(self, embed_dim, num_groups: list[int] | int, input_resolution, num_heads,
                 window_size: int, depth=48):
        super().__init__()
        num_groups = num_groups if isinstance(num_groups, list) else [num_groups] * 2
        padding = get_pad2d(input_resolution, [window_size] * 2)
        padding_left, padding_right, padding_top, padding_bottom = padding
        self.padding = padding
        self.pad = nn.ZeroPad2d(padding)
        input_resolution = list(input_resolution)
        input_resolution[0] = input_resolution[0] + padding_top + padding_bottom
        input_resolution[1] = input_resolution[1] + padding_left + padding_right
        self.down = DownBlock(embed_dim, embed_dim, num_groups[0])
        self.layer = BasicLayer(
            embed_dim, input_resolution, depth, num_heads, window_size
        )
        self.up = UpBlock(embed_dim * 2, embed_dim, num_groups[1])

    def forward(self, x):
        B, C, Lat, Lon = x.shape
        padding_left, padding_right, padding_top, padding_bottom = self.padding
        x = self.down(x)

        shortcut = x

        # pad
        x = self.pad(x)
        _, _, pad_lat, pad_lon = x.shape

        x = x.reshape(B, C, -1).transpose(1, 2)  # B Lat*Lon C
        x = self.layer(x)
        x = x.transpose(1, 2).reshape(B, C, pad_lat, pad_lon)

        # crop
        x = x[:, :, padding_top: pad_lat - padding_bottom, padding_left: pad_lon - padding_right]

        # concat
        x = torch.cat([shortcut, x], dim=1)  # B 2*C Lat Lon

        x = self.up(x)
        return x


class Fuxi(nn.Module):
    """
    Args:
        img_size (Sequence[int]): T, Lat, Lon.
        patch_size (Sequence[int]): T, Lat, Lon.
        in_chans (int): number of input channels.
        out_chans (int): number of output channels.
        embed_dim (int): number of embed channels.
        num_groups (Sequence[int] | int): number of groups to separate the channels into.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
    """
    def __init__(self, img_size=(2, 721, 1440), patch_size=(2, 4, 4), in_chans=70, out_chans=70,
                 embed_dim=1536, num_groups=32, num_heads=8, window_size=7):
        super().__init__()
        input_resolution = int(img_size[1] / patch_size[1] / 2), int(img_size[2] / patch_size[2] / 2)
        self.cube_embedding = CubeEmbedding(img_size, patch_size, in_chans, embed_dim)
        self.u_transformer = UTransformer(embed_dim, num_groups, input_resolution, num_heads, window_size)
        self.fc = nn.Linear(embed_dim, out_chans * patch_size[1] * patch_size[2])

        self.patch_size = patch_size
        self.input_resolution = input_resolution
        self.out_chans = out_chans
        self.img_size = img_size

    def forward(self, x: torch.Tensor):
        B, _, _, _, _ = x.shape
        _, patch_lat, patch_lon = self.patch_size
        Lat, Lon = self.input_resolution
        Lat, Lon = Lat * 2, Lon * 2
        x = self.cube_embedding(x).squeeze(2)  # B C Lat Lon
        x = self.u_transformer(x)
        x = self.fc(x.permute(0, 2, 3, 1))  # B Lat Lon C
        x = x.reshape(B, Lat, Lon, patch_lat, patch_lon, self.out_chans).permute(0, 1, 3, 2, 4, 5)
        # B, lat, patch_lat, lon, patch_lon, C

        x = x.reshape(B, Lat * patch_lat, Lon * patch_lon, self.out_chans)
        x = x.permute(0, 3, 1, 2)  # B C Lat Lon

        # bilinear
        x = F.interpolate(x, size=self.img_size[1:], mode="bilinear")

        return x
