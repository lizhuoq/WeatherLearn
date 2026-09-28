# FengWu architecture specs (arXiv:2304.02948)

Sources:
- Paper PDF/HTML: `fengwu-2304.02948.pdf`, `fengwu.html`
- Practical channel stack: OpenEarthLab ONNX (13 levels) + public training config
  (`yuchendoudou/FengWu` `config/fengwu.yaml`, license not clearly MIT — **not copied**)

---

## 1. Problem / I/O

| Item | Paper | WeatherLearn default |
|------|-------|----------------------|
| Grid | 0.25°, **721 × 1440** (lat × lon) | Configurable `img_size` |
| Pressure levels | **37** | **`n_levels=13`** (OpenEarthLab / ARCO 6h-128x64) |
| Surface vars `s` | **4**: u10, v10, t2m, msl | same |
| Upper-air modalities | **5**: z, r, u, v, t × levels | **z, q, u, v, t** (released stacks use specific humidity `q`) |
| Channels | **189** = 4 + 5×37 | **69** = 4 + 5×13; `n_levels=37` → 189 |
| Mapping | \(X^i \mapsto X^{i+1}\) (6 h) | same (single-frame input) |
| Output | mean **and** variance (Gaussian) | optional `predict_uncertainty`; smoke uses MSE on mean |

### Channel layout (stacked `[B, C, Lat, Lon]`)

```
[0:4]     u10, v10, t2m, msl
[4:4+L]   z × L
[4+L:4+2L] q × L
[4+2L:4+3L] u × L
[4+3L:4+4L] v × L
[4+4L:4+5L] t × L
```

13-level list (hPa): 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000.

---

## 2. Network: encode–fuse–decode

| Block | Paper | Skeleton in `weatherlearn/models/fengwu` |
|-------|-------|------------------------------------------|
| Modal encoders | 6× transformer encoders (Swin / U-Net) for `{s,z,q,u,v,t}` | `ModalEncoder`: PatchEmbed2d + Swin V2 stages + PatchMerging |
| Cross-modal fuser | Concat on feature dim → Transformer (ViT global + Swin local in later writeups) | `CrossModalFuser`: Linear → Swin V2 stage(s) → Linear split |
| Modal decoders | 6× decoders (mirror of encoders); predict μ and σ | `ModalDecoder`: PatchExpand + skip concat + Swin + linear patch head |
| Skips | U-Net-like across encoder/decoder | yes |

### Configurable hyperparameters (defaults)

| Name | `FengWu` (practical full-grid skeleton) | `FengWu_lite` |
|------|-------------------------------------------|---------------|
| `img_size` | (721, 1440)* | (64, 128) |
| `patch_size` | (4, 4) | (4, 4) |
| `n_levels` | 13 | 13 |
| `enc_dim` | 96 | 32 |
| `embed_dim` (fuser) | 768 | 128 |
| `enc_depths` | (2, 2) | (1, 1) |
| `enc_heads` | (3, 6) | (2, 4) |
| `fuser_depth` | 4 | 2 |
| `window_size` | (6, 12) | (4, 4) |

\*721 is not divisible by 4; construct with a padded/croppable size or use lite/`img_size` that divides cleanly. Reference training used patch/stride `(3,2)/(2,2)` on 721×1440 — not required for this skeleton.

Reference YAML (guidance only): `enc_dim=96`, `embed_dim=1152`, `enc_depths=[2,2,2]`, `lg_depths=[4,4,4]`, `window_size=[6,12]`, `inp_length=2` (two frames concatenated). v1 skeleton uses **single-frame** input; two-frame concat can be added later.

---

## 3. Uncertainty loss

\[
\mathcal{L} = \tfrac{1}{2}\Big(\log\sigma^2 + (y-\mu)^2 / \sigma^2\Big)
\]

Implemented as `uncertainty_loss(mean, log_var, target)` with `softplus(log_var)`.

---

## 4. Replay buffer (out of v1 smoke scope)

Paper §3.3: CPU-side queue of autoregressive predictions mixed into training to improve long-lead skill without huge GPU graph unrolls. **Not implemented** in v1 (documented gap).

---

## 5. Public GCS ERA5 for smoke

| Dataset | URI | Notes |
|---------|-----|-------|
| **Smoke (prefer)** | `gs://gcp-public-data-arco-era5/ar/1959-2022-6h-128x64_equiangular_conservative.zarr` | 6 h, **13 levels**, native `(lon=128, lat=64)` |
| Full-res (too heavy) | `gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3` | 37 levels, 0.25° |

Access: `gcsfs.GCSFileSystem(token="anon")` + `zarr` / `xarray` (no credentials).
