# FengWu integration status

**Branch:** `feat/fengwu` (fork `GISWLH/WeatherLearn`)  
**PR:** **blocked until user explicitly asks** — do not open upstream or fork→upstream PR yet.

## What matches the paper

| Item | Status |
|------|--------|
| Multi-modal split `{s,z,q,u,v,t}` | yes |
| Encode → concat fuse → decode | yes |
| Optional mean+var / uncertainty loss | yes (`predict_uncertainty`, `uncertainty_loss`) |
| 37-level / 189-ch via `n_levels=37` | supported as kwarg |
| Default 13-level / 69-ch | yes (OpenEarthLab practical) |
| Replay buffer | **not in v1** |
| Exact Swin/ViT hybrid fuser + paper width 1152 / 3-stage U-Net | simplified (configurable depths/dims) |
| Two-frame `inp_length=2` | **not in v1** (single-frame) |
| Pretrained weights | **none** (no invented checkpoints) |

## Channel layout (default)

`C=69`: `u10,v10,t2m,msl` + `z/q/u/v/t × 13`  
See `papers/FENGWU_SPECS.md` and `weatherlearn.models.fengwu.channel_layout`.

Humidity: paper text uses relative humidity `r`; released ONNX/training stacks and ARCO smoke use **specific humidity `q`**.

## Param counts (approximate; run locally to confirm)

| Model | Typical call | ~params |
|-------|--------------|---------|
| `FengWu_lite()` | 64×128, L=13, enc_dim=32, embed=128 | ~few M (print in smoke) |
| `FengWu` tiny | 32×64, L=13, enc_dim=16… | unit-test scale |
| Paper-scale | 721×1440, L=37, wide fuser | not smoked (memory) |

## How to run

### Unit tests (CPU)

```bash
cd WeatherLearn
python -m unittest discover -s tests/models/fengwu -v
```

### Train smoke (GCS public ARCO)

```bash
pip install xarray zarr fsspec gcsfs
cd WeatherLearn
python examples/fengwu/train_smoke_gcs.py --steps 2
# offline:
python examples/fengwu/train_smoke_gcs.py --synthetic --steps 2
```

GCS URI used:  
`gs://gcp-public-data-arco-era5/ar/1959-2022-6h-128x64_equiangular_conservative.zarr`

### HF Space

Files under `/workspace/weatherlearn/hf_space_fengwu/` — parent uploads to Hugging Face.  
Buttons: unit tests, lite train smoke (ZeroGPU), tiny forward.

## License awareness

- WeatherLearn package metadata: **BY-NC-SA 4.0** (see `pyproject.toml`).
- Do not redistribute proprietary FengWu ONNX weights.
- Reference GitHub training repo has unclear license — we reimplemented, did not copy.

## Acceptance checklist

- [x] Unit tests for shapes
- [x] Train smoke script (GCS + synthetic fallback)
- [x] HF Space files prepared
- [ ] User confirms HF GPU smoke OK
- [ ] User says OK to open PR
