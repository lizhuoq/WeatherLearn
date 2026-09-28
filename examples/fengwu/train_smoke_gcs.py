#!/usr/bin/env python3
"""FengWu_lite training smoke: pull 2 consecutive ERA5 frames from public GCS.

URI (default):
  gs://gcp-public-data-arco-era5/ar/1959-2022-6h-128x64_equiangular_conservative.zarr

Channel map (69-ch / 13-level FengWu layout):
  surface: 10m_u, 10m_v, 2m_temperature, mean_sea_level_pressure
  pressure (level dim = 13): geopotential, specific_humidity,
                             u_component_of_wind, v_component_of_wind, temperature

ARCO spatial layout is (longitude=128, latitude=64). This script transposes to
WeatherLearn (Lat, Lon) = (64, 128) to match FengWu_lite defaults.

Usage:
  python examples/fengwu/train_smoke_gcs.py
  python examples/fengwu/train_smoke_gcs.py --steps 2 --device cpu
  python examples/fengwu/train_smoke_gcs.py --synthetic   # no network
"""

from __future__ import annotations

import argparse
import sys
import traceback
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn

DEFAULT_URI = (
    "gs://gcp-public-data-arco-era5/ar/"
    "1959-2022-6h-128x64_equiangular_conservative.zarr"
)
FULL_URI = (
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
)

SURFACE = [
    ("u10", "10m_u_component_of_wind"),
    ("v10", "10m_v_component_of_wind"),
    ("t2m", "2m_temperature"),
    ("msl", "mean_sea_level_pressure"),
]
PRESSURE = [
    ("z", "geopotential"),
    ("q", "specific_humidity"),
    ("u", "u_component_of_wind"),
    ("v", "v_component_of_wind"),
    ("t", "temperature"),
]


def _open_zarr(uri: str):
    import gcsfs
    import zarr

    path = uri.replace("gs://", "")
    fs = gcsfs.GCSFileSystem(token="anon")
    return zarr.open_group(fs.get_mapper(path), mode="r")


def load_arco_pair(
    uri: str = DEFAULT_URI,
    time_index: int = 1000,
    n_levels: int = 13,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Load frames t and t+1 as ``[1, C, Lat, Lon]`` float32 tensors."""
    root = _open_zarr(uri)
    levels = np.asarray(root["level"][:])
    if len(levels) < n_levels:
        raise RuntimeError(f"zarr has {len(levels)} levels, need {n_levels}")
    meta = {
        "uri": uri,
        "time_index": time_index,
        "levels_hpa": levels[:n_levels].tolist(),
        "lon_lat_native": (int(root["longitude"].shape[0]), int(root["latitude"].shape[0])),
        "channel_order": [a for a, _ in SURFACE]
        + [f"{a}@{lv}" for a, _ in PRESSURE for lv in levels[:n_levels]],
    }

    def _surf(name, t):
        # native (lon, lat) → (lat, lon)
        arr = np.asarray(root[name][t], dtype=np.float32)
        return np.transpose(arr, (1, 0))

    def _plev(name, t):
        # native (level, lon, lat) → (level, lat, lon)
        arr = np.asarray(root[name][t, :n_levels], dtype=np.float32)
        return np.transpose(arr, (0, 2, 1))

    def frame(t: int) -> np.ndarray:
        parts = [_surf(src, t)[None, ...] for _, src in SURFACE]
        for _, src in PRESSURE:
            parts.append(_plev(src, t))
        return np.concatenate(parts, axis=0)

    x = torch.from_numpy(frame(time_index)).unsqueeze(0)
    y = torch.from_numpy(frame(time_index + 1)).unsqueeze(0)
    meta["tensor_lat_lon"] = (x.shape[-2], x.shape[-1])
    meta["channels"] = x.shape[1]
    return x, y, meta


def synthetic_pair(n_levels: int = 13, lat: int = 64, lon: int = 128):
    c = 4 + 5 * n_levels
    x = torch.randn(1, c, lat, lon)
    y = x + 0.01 * torch.randn_like(x)
    meta = {"uri": "synthetic", "channels": c, "tensor_lat_lon": (lat, lon)}
    return x, y, meta


def standardize_pair(x: torch.Tensor, y: torch.Tensor):
    """Per-channel standardize using input frame stats (smoke only)."""
    # x: B C H W
    mean = x.mean(dim=(0, 2, 3), keepdim=True)
    std = x.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-4)
    return (x - mean) / std, (y - mean) / std


def save_pair_cache(path: str, x: torch.Tensor, y: torch.Tensor, meta: dict) -> None:
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"x": x.cpu(), "y": y.cpu(), "meta": meta}, path)
    print(f"cached={path} bytes={os.path.getsize(path)}")


def load_pair_cache(path: str):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return blob["x"], blob["y"], blob["meta"]


def run_train(
    steps: int = 2,
    lr: float = 1e-3,
    device: Optional[str] = None,
    uri: str = DEFAULT_URI,
    time_index: int = 1000,
    synthetic: bool = False,
    n_levels: int = 13,
    cache_path: Optional[str] = None,
    load_cache_only: bool = False,
    prefetch_only: bool = False,
) -> int:
    from weatherlearn.models import FengWu_lite

    device_t = torch.device(
        device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"torch={torch.__version__}")
    print(f"device={device_t}")
    print(f"cuda_available={torch.cuda.is_available()}")

    try:
        if load_cache_only:
            if not cache_path:
                raise ValueError("load_cache_only requires cache_path")
            print(f"loading cache={cache_path}")
            x, y, meta = load_pair_cache(cache_path)
        elif synthetic:
            x, y, meta = synthetic_pair(n_levels=n_levels)
        else:
            print(f"loading GCS uri={uri} time_index={time_index}")
            x, y, meta = load_arco_pair(uri=uri, time_index=time_index, n_levels=n_levels)
            if cache_path:
                save_pair_cache(cache_path, x, y, meta)
        print(f"meta={meta}")
    except Exception as exc:
        print("STATUS=FAIL")
        print(f"data_load_error={exc!r}")
        traceback.print_exc()
        return 1

    if prefetch_only:
        print("STATUS=OK")
        print("prefetch_only=True")
        return 0

    x, y = standardize_pair(x, y)
    lat, lon = x.shape[-2], x.shape[-1]
    model = FengWu_lite(
        img_size=(lat, lon),
        n_levels=n_levels,
        enc_dim=32,
        embed_dim=128,
        enc_depths=(1, 1),
        enc_heads=(2, 4),
        fuser_depth=2,
        fuser_heads=4,
        window_size=(4, 4),
    )
    nparams = sum(p.numel() for p in model.parameters())
    print(f"params={nparams:,}")
    model = model.to(device_t)
    x = x.to(device_t)
    y = y.to(device_t)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        pred = model(x)
        loss = loss_fn(pred, y)
        if not torch.isfinite(loss):
            print("STATUS=FAIL")
            print(f"non_finite_loss_at_step={step} loss={loss}")
            return 1
        loss.backward()
        opt.step()
        print(f"step={step} loss={loss.item():.6f} pred_finite={bool(torch.isfinite(pred).all())}")

    print("STATUS=OK")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="FengWu_lite GCS ERA5 train smoke")
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default=None, help="cpu|cuda (default: auto)")
    p.add_argument("--uri", default=DEFAULT_URI)
    p.add_argument("--time-index", type=int, default=1000)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--n-levels", type=int, default=13)
    p.add_argument("--print-full-uri", action="store_true",
                   help="Print documented full-res URI and exit")
    args = p.parse_args()
    if args.print_full_uri:
        print(FULL_URI)
        return 0
    # argparse with choices including None is awkward; accept string device
    device = args.device
    return run_train(
        steps=args.steps,
        lr=args.lr,
        device=device,
        uri=args.uri,
        time_index=args.time_index,
        synthetic=args.synthetic,
        n_levels=args.n_levels,
    )


if __name__ == "__main__":
    # Allow running from repo root without install
    import os

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if root not in sys.path:
        sys.path.insert(0, root)
    raise SystemExit(main())
