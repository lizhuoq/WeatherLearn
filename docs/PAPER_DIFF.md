# WeatherLearn vs papers — mismatch report

对照：`docs/SPECS.md (also /workspace/weatherlearn/papers/SPECS.md)`（Pangu arXiv:2211.02556 / FuXi arXiv:2306.12873）与
`WeatherLearn/` @ branch `fix/paper-alignment`（基于 `fix/packaging-and-tests`）。

分类：
- **bug** — 相对论文/官方伪代码错误，且像是无意的
- **intentional** — 有意简化 / 文档化的 lite / 超出单骨干范围
- **ambiguous** — 论文未写死、或论文表述与官方伪代码冲突

---

## Summary

| Severity | Count | Action |
|----------|-------|--------|
| **bug** (fixed this branch) | 2 | window size 轴序；`PatchEmbed3D` height pad |
| **intentional** scope gaps | 若干 | 仅文档，不改图 |
| **ambiguous** | 若干 | 记录，不改 |
| FuXi last layer (Issue #8) | 0 bug | **保持**：FC + bilinear，与论文一致 |

**Verdict:** 骨干结构大体对齐；本分支修了 **Pangu window 在 `(pl,lat,lon)` 下 lat/lon 对调**，以及 **3D patch pad 用错取模除数**。无 cascade / 多 lead-time 聚合属系统级 scope，不算结构 bug。

---

## Pangu-Weather

### Matches（对齐）

| Spec | Code evidence |
|------|----------------|
| Grid 721×1440；upper 5×13；surface 4+3 masks | `pangu.py:368–379` `PatchEmbed2D/3D`；`in_chans=4+3` / `5` |
| Patch 2×4×4 / 4×4 | `pangu.py:370–378` |
| C=192；heads (6,12,12,6) | `pangu.py:366` |
| Depths 2+6 encoder / 6+2 decoder | `pangu.py:384–414` `depth=2/6/6/2` |
| Downsample **仅水平**；pl 不变 8 | `DownSample` assert `in_pl == out_pl`；`(8,181,360)→(8,91,180)` |
| Skip concat encoder↔decoder → 2C recovery | `pangu.py:444` + `PatchRecovery2D/3D(..., 2*embed_dim, ...)` |
| Earth-specific bias（非 plain relative） | `EarthAttention3D` `earth_position_bias_table` + `get_earth_position_index` |
| Separate surface/upper recovery | `patchrecovery2d` / `patchrecovery3d` |
| `Pangu_lite` 更大 patch = 有意降算力 | `pangu.py:487–497` patch `(8,8)` / `(2,8,8)` → **intentional** |

### Bugs（本分支已修）

#### P1. Window size lat/lon 对调 — **bug → fixed**

| | |
|--|--|
| Paper | `Wpl × Wlat × Wlon = 2 × 12 × 6` |
| Official pseudocode | tensor `(pl, lon, lat)` + `window_size=(2, 6, 12)` → 物理上 lat=12, lon=6 |
| WeatherLearn（修前） | tensor `(pl, lat, lon)` + `window_size=(2, 6, 12)` → **lat=6, lon=12（反了）** |
| WeatherLearn（修后） | `window_size=(2, 12, 6)`，`shift_size = tuple(w//2 for w in window_size)` → `(1, 6, 3)` |

Evidence: `pangu.py:224–227`, `366`, `485`；注释说明轴序。

#### P2. `PatchEmbed3D` height pad 用 `l_patch_size` 取模 — **bug → fixed**

| | |
|--|--|
| 修前 | `h_remainder = height % l_patch_size` |
| 修后 | `h_remainder = height % h_patch_size` |
| 影响 | **潜伏**：默认 `(2,4,4)` @ 721 碰巧结果相同；`h≠l` 或其它高度时可能 pad 后无法被 `h_patch_size` 整除 |

Evidence: `utils/patch_embed.py:75`；回归测试 `test_patch_embed3d_height_pad_uses_h_patch`。

### Intentional / out of scope

| Item | Classification | Notes |
|------|----------------|-------|
| 无 1h/3h/6h/24h 多模型推理调度 | **intentional** | 单 `Pangu` 骨干；系统级 hierarchical temporal aggregation 未实现 |
| 无 Perlin ensemble | **intentional** | 论文系统能力，非 3DEST 结构 |
| 无预训练权重 | **intentional** | 任务禁止发明权重 |
| `Pangu_lite` 非论文配置 | **intentional** | 明确 lite |

### Ambiguous

| Item | Notes |
|------|-------|
| Patch embed 后 GeLU | 论文写 “linear + GeLU”；官方伪代码 **无** GeLU。本仓库跟随伪代码（仅 Conv）→ **ambiguous**，不改 |
| Pre-norm vs post-norm | 官方伪代码写法混乱；本仓库用 Swin 式 pre-norm → **ambiguous** |
| Skip “2nd encoder / 7th decoder” | 论文字面 vs 整段 layer4 后 concat；官方伪代码也是整段后 concat；本仓库同 → **ambiguous** / 跟官方 |
| `w_path_size` 拼写 | `PatchEmbed2D` 局部变量 typo，行为正确 → 非功能 bug |

---

## FuXi

### Matches（对齐）

| Spec | Code evidence |
|------|----------------|
| in/out 70 ch；2-frame；721×1440 | `fuxi.py:157–158` defaults |
| Cube embed Conv3d `2×4×4`，C=1536，LayerNorm | `CubeEmbedding` `fuxi.py:12–38`, `161` |
| U-Transformer：Down → SwinV2×48 → concat skip → Up | `UTransformer` `fuxi.py:94–141` |
| Down: stride-2 3×3 Conv + residual Conv-GN-SiLU | `DownBlock` `42–68` |
| Up: ConvTranspose k=2 + residual | `UpBlock` `71–91` |
| FC patch expand → 70×720×1440 → **bilinear** → 721 | `fuxi.py:163,177–186`；见 `docs/fuxi_last_layer.md`（Issue #8） |
| `depth` 可配，默认 48 | `fuxi.py:155–158`（packaging 分支已做） |

### Intentional / out of scope

| Item | Classification | Notes |
|------|----------------|-------|
| **无 Short/Medium/Long cascade** | **intentional** | 论文系统 = 三个同构骨干 + 分时段 finetune/接力；仓库只有单 `Fuxi` 类。**缺 cascade 调度，不是 U-Transformer 结构 bug** |
| 无预训练/finetune 脚本对接 cascade | **intentional** | 同上 |
| API 布局 `(B,C,T,H,W)` vs 论文 `T×C×H×W` | **intentional** | Conv3d 语义等价 |

### Ambiguous

| Item | Notes |
|------|-------|
| `window_size=7`, `num_heads=8` | 论文未给出数值；合理默认 → **ambiguous** |
| Down residual 是 Conv-GN-SiLU×2 vs Conv×2 后 GN+SiLU | 论文措辞含糊；实现为常见 Conv-GN-SiLU×2 → **ambiguous** |
| Down 对奇数边 `:-1` crop | 工程需要；论文未提 → **ambiguous** / 实用 |

### Non-bug（Issue #8）

**不要重开：** 最后一层 = Linear patch FC + bilinear 720→721，与论文一致。讨论里曾把 “FC” 误读成整幅 720×1440→721×1440 的 dense 层。

---

## Fixes on `fix/paper-alignment`

1. `weatherlearn/models/pangu/pangu.py` — default `window_size=(2, 12, 6)`；`shift_size` 由 window 半窗推导。
2. `weatherlearn/models/pangu/utils/patch_embed.py` — `PatchEmbed3D` height 用 `h_patch_size` 取模。
3. Tests — 回归 + 默认 window 断言。

未改：FuXi cascade、Pangu 多 lead-time、GeLU-after-embed、整体 OOP。

---

## Remaining gaps（不阻塞）

1. 系统级 Pangu hierarchical temporal aggregation  
2. 系统级 FuXi Short/Medium/Long cascade  
3. 预训练权重与加载器  
4. 论文未指定的 FuXi window/heads 超参正式出处  
