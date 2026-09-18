# FuXi last layer: FC vs bilinear interpolate (Issue #8)

This note clarifies the discussion in [Issue #8](https://github.com/lizhuoq/WeatherLearn/issues/8) relative to the FuXi paper and this implementation. Tone: the repository already matches the paper; the confusion was about naming two consecutive steps.

## What the paper describes

FuXi’s final stage has **two** operations:

1. A **fully connected (FC) / Linear** layer that expands each spatial token from `embed_dim` to `out_chans * patch_h * patch_w`, then reshapes to a regular grid — with default patch `(2, 4, 4)` this yields **70 × 720 × 1440**.
2. **Bilinear interpolation** that restores latitude from **720 → 721**, producing **70 × 721 × 1440**.

The paper (arXiv:2306.12873 / npj Climate and Atmospheric Science) states that the output is reshaped to 70×720×1440 and then restored to 70×721×1440 by bilinear interpolation. The named “FC” is the patch un-embedding projection, not a dense map from 720×1440 → 721×1440.

## What this repo does

In `weatherlearn/models/fuxi/fuxi.py`, `Fuxi.forward`:

- `self.fc = nn.Linear(embed_dim, out_chans * patch_size[1] * patch_size[2])` — paper FC / patch expand.
- Reshape/permute to `(B, C, 720, 1440)` with default settings (`floor(721/4)*4 = 720`).
- `F.interpolate(..., mode="bilinear")` to `(B, C, 721, 1440)` — same restore as the paper.

So interpolate is **not** a compute shortcut that replaces the paper’s FC; both steps are present.

## Why a literal “FC to 721×1440” is not what people usually mean

A dense `Linear(720*1440, 721*1440)` would be on the order of **~1T parameters** and is not described by the paper. A small learned lat-only upsample (`Linear(720, 721)`) would be a different design choice (tiny params); the paper and this repo use bilinear for that +1 latitude step.

Pangu in this zoo recovers 721 via **ConvTranspose + center crop** (`PatchRecovery2D/3D`), which is a different recovery style — also valid, just not FuXi’s.

## Practical note

Default FuXi uses `depth=48` and `embed_dim=1536` (~1.5B-scale). For tests and smoke, pass a smaller `depth` / `embed_dim` / spatial size. Set `WEATHERLEARN_RUN_HEAVY=1` to run full-resolution heavy unit tests (e.g. full `Pangu()`).
