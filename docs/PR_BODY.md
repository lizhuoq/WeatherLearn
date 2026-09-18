## Summary

Surgical fixes after packaging hygiene + paper alignment against **Pangu-Weather** (arXiv:2211.02556) and **FuXi** (arXiv:2306.12873). No OOP rewrite.

### Packaging / DX
- Add missing package `__init__.py` so `pip install -e .` and `python -m unittest discover` work.
- Relax `pyproject.toml` pins to installable `>=` ranges (exact pins broke on Python 3.12/3.13).
- Prefer `timm.layers` with fallback for `timm.models.layers` deprecation.
- Make FuXi `depth` configurable (default **48**, backward compatible).
- Skip full `Pangu()` unless `WEATHERLEARN_RUN_HEAVY=1`.
- Document Issue #8: last layer is **Linear patch FC + bilinear interpolate** (matches FuXi paper). See `docs/fuxi_last_layer.md`.

### Paper alignment bugs
1. **Pangu default `window_size` lat/lon swapped (real bug)**  
   Paper: each window up to **Wpl×Wlat×Wlon = 2×12×6**.  
   Official pseudocode uses `window_size=(2,6,12)` on tensors shaped `(pl, lon, lat)` (e.g. `(8,360,181)`).  
   This repo stores **`(pl, lat, lon)`** and documents window as `[pl, lat, lon]`, but previously defaulted to `(2,6,12)`, i.e. **Wlat=6, Wlon=12** — physically swapped vs the paper. Earth-specific bias also treats the **last** window dim as cyclic longitude, so the swap is semantically wrong.  
   **Fix:** default `(2,12,6)`; `shift_size = half window`.

2. **`PatchEmbed3D` height pad used wrong modulus (real logic bug; latent on defaults)**  
   Previously: `h_remainder = height % l_patch_size` while pad amount used `h_patch_size`.  
   For stock Pangu `patch=(2,4,4)` @ height 721, `721%2` and `721%4` coincide, so **default shapes were unaffected**. Wrong when `l≠h` or other heights (can yield non-divisible padded height).  
   **Fix:** `h_remainder = height % h_patch_size` (+ regression test). Please treat as a **latent correctness** fix, not a default-training breaker.

### Not bugs / out of scope
- FuXi Issue #8 last layer: **not** a bug (FC + bilinear 720→721 matches paper).
- FuXi Short/Medium/Long cascade; Pangu 1h/3h/6h/24h hierarchical inference: system-level, undocumented here as intentional gaps.
- Patch-embed GeLU: paper text vs official pseudocode disagree → left following official (no GeLU).

Details: `docs/SPECS.md`, `docs/PAPER_DIFF.md`.

## Test plan
- [x] `pip install -e .` (relaxed deps) + `from weatherlearn.models import Fuxi, Pangu, Pangu_lite`
- [x] `python -m unittest discover -s tests -v` → **45 OK, 1 skipped** (heavy `test_pangu`)
- [x] New checks: default window `(2,12,6)`; PatchEmbed3D height pad regression
- [x] HF Space ZeroGPU smoke (`LonghaoWang/weatherlearn-fuxi-smoke`): FuXi+Pangu unit tests, lite + tiny721 forward OK
- [ ] Optional: `WEATHERLEARN_RUN_HEAVY=1` full `Pangu()` on high-RAM GPU

## Notes
- License remains **BY-NC-SA 4.0**. No invented pretrained weight URLs.
- Related: Issue #8.

## 中文说明
- **Pangu window**：相对论文是实打实的轴序 bug（本仓库 `(pl,lat,lon)` 却沿用官方伪代码在 `(pl,lon,lat)` 下的 `(2,6,12)`）。
- **PatchEmbed3D**：取模公式确实错，但对默认 `721×(2,4,4)` 潜伏；PR 按 latent correctness 表述，不夸大默认训练影响。
- FuXi 末层与论文一致，不重开 Issue #8。
