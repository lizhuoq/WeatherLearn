"""FengWu multi-modal encode-fuse-decode weather forecast backbone.

Faithful PyTorch skeleton of Chen et al., arXiv:2304.02948.
Clean reimplementation in WeatherLearn style (timm Swin V2). Does not vendor
third-party training source; channel/modality layout follows the paper and the
practical 13-level OpenEarthLab / public training configs.

Default ``n_levels=13`` → 69 channels. Pass ``n_levels=37`` for the paper's
189-channel setting (paper humidity symbol is ``r``; released stacks use ``q``).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
from timm.layers.helpers import to_2tuple
from timm.models.swin_transformer_v2 import SwinTransformerV2Stage

from ..pangu.utils.pad import get_pad2d

MODALITY_NAMES = ("s", "z", "q", "u", "v", "t")
SURFACE_VARS = ("u10", "v10", "t2m", "msl")
PRESSURE_VARS = ("z", "q", "u", "v", "t")
DEFAULT_LEVELS_13 = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)


def channel_layout(n_levels: int = 13) -> dict:
    """Channel index map for stacked tensor ``[B, C, Lat, Lon]``."""
    layout = {
        "n_levels": n_levels,
        "n_channels": 4 + 5 * n_levels,
        "surface": {name: i for i, name in enumerate(SURFACE_VARS)},
        "pressure": {},
        "order": "u10,v10,t2m,msl | z×L | q×L | u×L | v×L | t×L",
    }
    offset = 4
    for var in PRESSURE_VARS:
        layout["pressure"][var] = (offset, offset + n_levels)
        offset += n_levels
    return layout


def split_modalities(x: torch.Tensor, n_levels: int) -> List[torch.Tensor]:
    expected = 4 + 5 * n_levels
    if x.shape[1] != expected:
        raise ValueError(f"Expected C={expected} (n_levels={n_levels}), got C={x.shape[1]}")
    parts = [x[:, :4]]
    offset = 4
    for _ in range(5):
        parts.append(x[:, offset : offset + n_levels])
        offset += n_levels
    return parts


def concat_modalities(parts: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat(list(parts), dim=1)


class PatchEmbed2d(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.img_size = tuple(img_size)
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        if (H, W) != self.img_size:
            raise ValueError(f"Input spatial {(H, W)} != model img_size {self.img_size}")
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim: int, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        if H % 2 or W % 2:
            raise ValueError(f"PatchMerging needs even H,W got {(H, W)}")
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        return self.reduction(self.norm(x))


class PatchExpand(nn.Module):
    def __init__(self, dim: int, norm_layer=nn.LayerNorm):
        super().__init__()
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = norm_layer(dim // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        x = self.expand(x).view(B, H, W, 2, 2, C // 2)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H * 2, W * 2, C // 2)
        return self.norm(x)


class SwinStage(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size):
        super().__init__()
        window_size = to_2tuple(window_size)
        padding = get_pad2d(input_resolution, window_size)
        pad_l, pad_r, pad_t, pad_b = padding
        self.padding = padding
        self.pad = nn.ZeroPad2d(padding)
        padded_res = (
            input_resolution[0] + pad_t + pad_b,
            input_resolution[1] + pad_l + pad_r,
        )
        self.stage = SwinTransformerV2Stage(dim, dim, padded_res, depth, num_heads, window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_l, pad_r, pad_t, pad_b = self.padding
        x = self.pad(x.permute(0, 3, 1, 2))
        _, _, pH, pW = x.shape
        x = self.stage(x.permute(0, 2, 3, 1))
        h1 = pH - pad_b if pad_b else pH
        w1 = pW - pad_r if pad_r else pW
        return x[:, pad_t:h1, pad_l:w1, :]


class ModalEncoder(nn.Module):
    """Patch embed + alternating Swin / PatchMerging (U-Net down path)."""

    def __init__(self, img_size, patch_size, in_chans, enc_dim, depths, num_heads, window_size):
        super().__init__()
        assert len(depths) == len(num_heads) and len(depths) >= 1
        self.patch_embed = PatchEmbed2d(img_size, patch_size, in_chans, enc_dim)
        self.stages = nn.ModuleList()
        self.merges = nn.ModuleList()
        dim = enc_dim
        res = self.patch_embed.grid_size
        for i, (d, h) in enumerate(zip(depths, num_heads)):
            self.stages.append(SwinStage(dim, res, d, h, window_size))
            if i < len(depths) - 1:
                self.merges.append(PatchMerging(dim))
                dim *= 2
                res = (res[0] // 2, res[1] // 2)
        self.out_dim = dim
        self.out_resolution = res
        self.num_stages = len(depths)

    def forward(self, x: torch.Tensor):
        skips = []
        x = self.patch_embed(x)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < self.num_stages - 1:
                skips.append(x)
                x = self.merges[i](x)
        return x, skips


class ModalDecoder(nn.Module):
    """U-Net up path with skip concat, then linear patch expand to pixels."""

    def __init__(
        self,
        patch_size,
        out_chans,
        enc_dim,
        depths,
        num_heads,
        window_size,
        bottleneck_resolution,
        predict_uncertainty=False,
    ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.out_chans = out_chans
        self.predict_uncertainty = predict_uncertainty
        n = len(depths)
        dims = [enc_dim * (2**i) for i in range(n)]

        self.expands = nn.ModuleList()
        self.skip_projs = nn.ModuleList()
        self.stages = nn.ModuleList()

        res = bottleneck_resolution
        # From deepest-1 down to finest: expand, fuse skip, Swin
        for i in range(n - 1, 0, -1):
            self.expands.append(PatchExpand(dims[i]))
            self.skip_projs.append(nn.Linear(2 * dims[i - 1], dims[i - 1]))
            res = (res[0] * 2, res[1] * 2)
            self.stages.append(
                SwinStage(dims[i - 1], res, depths[i - 1], num_heads[i - 1], window_size)
            )

        # Single-stage encoder (n==1): refine bottleneck tokens before head
        self.bottleneck_stage = (
            SwinStage(enc_dim, bottleneck_resolution, depths[0], num_heads[0], window_size)
            if n == 1
            else nn.Identity()
        )

        out_factor = 2 if predict_uncertainty else 1
        ph, pw = self.patch_size
        self.head = nn.Linear(enc_dim, out_chans * out_factor * ph * pw)

    def forward(self, x: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        # skips: [after_stage0, after_stage1, ...] excluding bottleneck (len = n-1)
        skips_rev = list(reversed(skips))
        if isinstance(self.bottleneck_stage, nn.Identity):
            for i, (expand, proj, stage) in enumerate(
                zip(self.expands, self.skip_projs, self.stages)
            ):
                x = expand(x)
                x = proj(torch.cat([x, skips_rev[i]], dim=-1))
                x = stage(x)
        else:
            x = self.bottleneck_stage(x)

        B, H, W, _ = x.shape
        ph, pw = self.patch_size
        out_factor = 2 if self.predict_uncertainty else 1
        out_c = self.out_chans * out_factor
        x = self.head(x).view(B, H, W, ph, pw, out_c)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, out_c, H * ph, W * pw)
        return x


class CrossModalFuser(nn.Module):
    def __init__(
        self, in_dim, n_modalities, fuse_dim, input_resolution, depth, num_heads, window_size
    ):
        super().__init__()
        self.n_modalities = n_modalities
        self.in_proj = nn.Linear(in_dim * n_modalities, fuse_dim)
        self.blocks = SwinStage(fuse_dim, input_resolution, depth, num_heads, window_size)
        self.out_proj = nn.Linear(fuse_dim, in_dim * n_modalities)

    def forward(self, zs: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        z = self.in_proj(torch.cat(list(zs), dim=-1))
        z = self.blocks(z)
        z = self.out_proj(z)
        return list(torch.chunk(z, self.n_modalities, dim=-1))


class FengWu(nn.Module):
    """FengWu encode–fuse–decode backbone.

    Args:
        img_size: (Lat, Lon). Full ERA5: (721, 1440). Must be divisible by
            ``patch_size * 2**(len(enc_depths)-1)``.
        n_levels: default **13** (69 ch). Paper: **37** (189 ch).
        predict_uncertainty: if True, ``forward`` returns ``(mean, log_var)``.
    """

    def __init__(
        self,
        img_size: Tuple[int, int] = (721, 1440),
        patch_size: Union[int, Tuple[int, int]] = (4, 4),
        n_levels: int = 13,
        enc_dim: int = 96,
        embed_dim: int = 768,
        enc_depths: Sequence[int] = (2, 2),
        enc_heads: Sequence[int] = (3, 6),
        fuser_depth: int = 4,
        fuser_heads: int = 6,
        window_size: Union[int, Tuple[int, int]] = (6, 12),
        predict_uncertainty: bool = False,
    ):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        img_size = tuple(img_size)
        # Pad spatial dims so patch embed + (len(depths)-1) merges divide evenly
        # (paper ERA5 lat=721 is odd; pad then crop in forward).
        factor = patch_size[0] * (2 ** max(len(enc_depths) - 1, 0))
        factor_w = patch_size[1] * (2 ** max(len(enc_depths) - 1, 0))
        pad_h = (factor - img_size[0] % factor) % factor
        pad_w = (factor_w - img_size[1] % factor_w) % factor_w
        work_size = (img_size[0] + pad_h, img_size[1] + pad_w)

        self.img_size = img_size
        self.work_size = work_size
        self._spatial_pad = (0, pad_w, 0, pad_h)  # left,right,top,bottom for F.pad
        self.patch_size = patch_size
        self.n_levels = n_levels
        self.predict_uncertainty = predict_uncertainty
        self.in_chans = self.out_chans = 4 + 5 * n_levels
        self.layout = channel_layout(n_levels)

        in_chans_list = [4] + [n_levels] * 5
        self.encoders = nn.ModuleList(
            [
                ModalEncoder(work_size, patch_size, c, enc_dim, enc_depths, enc_heads, window_size)
                for c in in_chans_list
            ]
        )
        b_dim = self.encoders[0].out_dim
        b_res = self.encoders[0].out_resolution
        self.fuser = CrossModalFuser(
            b_dim, 6, embed_dim, b_res, fuser_depth, fuser_heads, window_size
        )
        self.decoders = nn.ModuleList(
            [
                ModalDecoder(
                    patch_size,
                    c,
                    enc_dim,
                    enc_depths,
                    enc_heads,
                    window_size,
                    b_res,
                    predict_uncertainty=predict_uncertainty,
                )
                for c in in_chans_list
            ]
        )

    def forward(self, x: torch.Tensor):
        if x.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")
        _, _, H, W = x.shape
        if (H, W) != self.img_size:
            raise ValueError(f"Spatial {(H, W)} != img_size {self.img_size}")
        if self._spatial_pad != (0, 0, 0, 0):
            x = F.pad(x, self._spatial_pad)

        modalities = split_modalities(x, self.n_levels)
        bottlenecks, skips_list = [], []
        for enc, m in zip(self.encoders, modalities):
            z, skips = enc(m)
            bottlenecks.append(z)
            skips_list.append(skips)
        fused = self.fuser(bottlenecks)

        if self.predict_uncertainty:
            means, log_vars = [], []
            for dec, z, skips in zip(self.decoders, fused, skips_list):
                out = self._match_spatial(dec(z, skips))
                mean_m, log_var_m = torch.chunk(out, 2, dim=1)
                means.append(mean_m)
                log_vars.append(log_var_m)
            return concat_modalities(means), concat_modalities(log_vars)

        outs = [
            self._match_spatial(dec(z, skips))
            for dec, z, skips in zip(self.decoders, fused, skips_list)
        ]
        return concat_modalities(outs)

    def _match_spatial(self, y: torch.Tensor) -> torch.Tensor:
        _, _, h, w = y.shape
        th, tw = self.img_size
        if (h, w) == (th, tw):
            return y
        if h >= th and w >= tw:
            return y[:, :, :th, :tw]
        return F.interpolate(y, size=self.img_size, mode="bilinear", align_corners=False)


def FengWu_lite(
    img_size: Tuple[int, int] = (64, 128),
    patch_size: Union[int, Tuple[int, int]] = (4, 4),
    n_levels: int = 13,
    enc_dim: int = 32,
    embed_dim: int = 128,
    enc_depths: Sequence[int] = (1, 1),
    enc_heads: Sequence[int] = (2, 4),
    fuser_depth: int = 2,
    fuser_heads: int = 4,
    window_size: Union[int, Tuple[int, int]] = (4, 4),
    predict_uncertainty: bool = False,
    **kwargs,
) -> FengWu:
    """Small FengWu for CPU / ZeroGPU smoke (default Lat×Lon = 64×128)."""
    return FengWu(
        img_size=img_size,
        patch_size=patch_size,
        n_levels=n_levels,
        enc_dim=enc_dim,
        embed_dim=embed_dim,
        enc_depths=enc_depths,
        enc_heads=enc_heads,
        fuser_depth=fuser_depth,
        fuser_heads=fuser_heads,
        window_size=window_size,
        predict_uncertainty=predict_uncertainty,
        **kwargs,
    )


def uncertainty_loss(mean, log_var, target, eps: float = 1e-6) -> torch.Tensor:
    """Gaussian NLL with softplus variance (paper multi-task uncertainty loss)."""
    var = F.softplus(log_var) + eps
    return (0.5 * (torch.log(var) + (target - mean) ** 2 / var)).mean()
