# Architecture specs extracted from papers

Sources:
- Pangu-Weather: arXiv:2211.02556 / Nature 2023 (`pangu-2211.02556.pdf`, `pangu.html`)
- FuXi: arXiv:2306.12873 (`fuxi-2306.12873.pdf`, `fuxi.html`)
- Official Pangu pseudocode: https://github.com/198808xc/Pangu-Weather/blob/main/pseudocode.py

Axis-order note: the Pangu paper / official pseudocode store tensors as **(pl, lon, lat)** (e.g. `8 × 360 × 181`). WeatherLearn stores **(pl, lat, lon)** (e.g. `(8, 181, 360)`). Specs below give both where order matters.

---

## 1. Pangu-Weather (3DEST)

### 1.1 Inputs / outputs / grid

| Item | Paper | Notes |
|------|-------|-------|
| Grid | 0.25°, **721 × 1440** (lat × lon) | lon=1440, lat=721 (incl. both poles) |
| Pressure levels | **13** (50–1000 hPa subset) | |
| Upper-air vars | **5**: Z, Q, T, U, V | cube `13 × 1440 × 721 × 5` (paper order pl×lon×lat×C) |
| Surface vars | **4**: T2M, U10, V10, MSLP | cube `1440 × 721 × 4` |
| Static masks | **3**: topography, land-sea, soil type | concatenated to surface before embed → 7 ch |
| Single-model lead times | separate models for **1h / 3h / 6h / 24h** | hierarchical temporal aggregation at inference |
| Loss | MAE; surface weight 0.25 | training detail |

### 1.2 Patch embedding / recovery

| Item | Paper | Official pseudocode |
|------|-------|---------------------|
| Upper patch | **2 × 4 × 4** (pl × horiz × horiz) | Conv3d k=s=(2,4,4), in=5 |
| Surface patch | **4 × 4** | Conv2d k=s=(4,4), in=**7** (4+3 masks) |
| Activation | text: “linear + **GeLU**” | **no GeLU** after conv |
| Embedded upper | `7 × 360 × 181 × C` (pl×lon×lat) | |
| Embedded surface | `360 × 181 × C` | |
| Concat | along height → **`8 × 360 × 181 × C`** | surface as extra “level” |
| Recovery | separate ConvTranspose 3D / 2D; **no** param share with embed; crop pads | same |
| Default **C** | **192** | `PatchEmbedding(..., 192)` |

### 1.3 Encoder / decoder

| Stage | Depth | Resolution (paper pl×lon×lat) | Channels | Heads (official) |
|-------|-------|-------------------------------|----------|------------------|
| Encoder layer1 | **2** | 8 × 360 × 181 | C | 6 |
| Downsample | — | → 8 × 180 × 91 | C → 2C | spatial only (pl unchanged) |
| Encoder layer2 | **6** | 8 × 180 × 91 | 2C | 12 |
| Decoder layer3 | **6** | 8 × 180 × 91 | 2C | 12 |
| Upsample | — | → 8 × 360 × 181 | 2C → C | |
| Decoder layer4 | **2** | 8 × 360 × 181 | C | 6 |
| Skip | concat **2nd encoder** output with **7th decoder** (channel) | → 2C into recovery | |

Total Earth-specific blocks: **8 encoder + 8 decoder** (2+6 / 6+2).

Downsample: merge 2×2 spatial tokens (4C) → Linear → 2C; LayerNorm; **pressure dim unchanged**.  
Upsample: reverse; crop back to odd lat (181 / 91).

### 1.4 Window attention & Earth-specific positional bias (ESPB)

| Item | Paper |
|------|-------|
| Window | **Wpl × Wlat × Wlon = 2 × 12 × 6** |
| On paper tensor `(pl, lon, lat)` | equivalent window tuple **(2, 6, 12)** |
| On WeatherLearn `(pl, lat, lon)` | must be **(2, 12, 6)** |
| Shifted window | every other block; half-window shift; lon wrap merge |
| Bias | **Earth-specific absolute** bias (not plain relative): `Mpl×Mlat` sub-matrices (lon shared), each `Wpl² × Wlat² × (2 Wlon − 1)` params × heads |
| Drop path | linear 0 → 0.2 over 8 blocks (official) |

### 1.5 Out of single-backbone scope

- Hierarchical **1h/3h/6h/24h** multi-model inference schedule
- Perlin-noise perturbation ensemble
- Pretrained ONNX weights

---

## 2. FuXi

### 2.1 Inputs / outputs / grid

| Item | Paper |
|------|-------|
| Grid | 0.25°, **721 × 1440** |
| Channels | **70** = 5 upper-air × 13 levels + 5 surface (incl. TP) |
| Temporal input | **2** frames (t−1, t) → predict t+1 (6 h step) |
| Input cube | paper layout `2 × 70 × 721 × 1440` (T×C×H×W) |
| Output | `70 × 721 × 1440` |

### 2.2 Cube embedding

| Item | Paper |
|------|-------|
| Op | Conv3d kernel/stride **2 × 4 × 4**, out channels **C = 1536** |
| Norm | LayerNorm after embed |
| Latent | **C × 180 × 360** (T collapsed; lat floor 721/4 → 180) |

### 2.3 U-Transformer

| Item | Paper |
|------|-------|
| Backbone | **Swin Transformer V2**, scaled cosine attention, relative bias B, learnable τ |
| Depth | **48** repeated Swin V2 blocks |
| Down Block | stride-2 **3×3 Conv2d** → residual of two **3×3** + **GN** + **SiLU**; size → `C × 90 × 180` |
| Up Block | **ConvTranspose2d** k=2,s=2 + same residual; → `C × 180 × 360` |
| Skip | concat Down output with transformer output (2C) into Up |
| Window / heads | **not numerically specified** in paper (impl-defined) |

### 2.4 Output head

| Item | Paper |
|------|-------|
| FC | fully connected patch expand |
| Reshape | **70 × 720 × 1440** |
| Restore | **bilinear** → **70 × 721 × 1440** |

### 2.5 Cascade (system-level)

| Model | Lead window | Role |
|-------|-------------|------|
| FuXi-Short | 0–5 days | base; 20 steps × 6 h |
| FuXi-Medium | 5–10 days | finetuned from Short; fed step-20 |
| FuXi-Long | 10–15 days | finetuned from Medium; fed step-40 |

Same backbone architecture; cascade is **training/inference scheduling**, not a different tensor graph.

---

## 3. Quick cross-check constants

| Constant | Pangu | FuXi |
|----------|-------|------|
| Spatial grid | 721×1440 | 721×1440 |
| Embed dim C | 192 | 1536 |
| Transformer depth | 16 blocks (8+8) | 48 Swin V2 blocks |
| Multi-frame input | 1 frame | 2 frames |
| Output recovery | ConvTranspose + crop | FC + bilinear |
| Earth-specific bias | yes (ESPB) | no (Swin V2 relative + τ) |
