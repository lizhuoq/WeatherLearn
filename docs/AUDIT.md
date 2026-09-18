# WeatherLearn full-library audit

**Repo:** https://github.com/lizhuoq/WeatherLearn  
**Local clone:** `/workspace/weatherlearn/WeatherLearn` @ `41333e1` + branch `fix/packaging-and-tests`  
**Audit date:** 2026-09-18 00:04 HKT  
**Scope:** `weatherlearn/`, `tests/`, `examples/`, `finetune/`, `inference/`  
**Related:** `findings.md` (Issue #8); this audit drives PR-ready packaging/test fixes.

Severity: **blocker** | **high** | **med** | **low**

---

## Applied on branch `fix/packaging-and-tests`

| ID | Severity | Location | Issue | Fix applied |
|----|----------|----------|-------|-------------|
| A1 | **blocker** | `pyproject.toml` L9–13 | Exact pins `torch==2.1.0`, `timm==0.9.10`, `numpy==1.23.5` fail on Python 3.12/3.13 (no wheels / broken builds). `pip install -e .` with deps unusable. | Relaxed to `>=` ranges; documented in CHANGELOG-PR. Version bump `0.1.0` → `0.1.1`. Added `packages = [{include = "weatherlearn"}]`. |
| A2 | **blocker** | Missing `weatherlearn/__init__.py`, `.../fuxi/__init__.py`, `.../pangu/__init__.py`, `.../utils/__init__.py`, `data_utils/__init__.py`, `tests/**/__init__.py` | Package not a proper regular package; `unittest discover` finds 0 tests / import errors without hacks. | Added package inits; root re-exports `Fuxi, Pangu, Pangu_lite`. |
| A3 | **high** | `tests/models/pangu/test_main.py:105` `test_pangu` | Full `Pangu()` @ 721×1440 OOMs (~exit 137) on modest RAM; kills whole discover runs. | `@unittest.skipUnless` unless `WEATHERLEARN_RUN_HEAVY=1`. |
| A4 | **high** | `weatherlearn/models/pangu/pangu.py:4` | `from timm.models.layers import ...` emits FutureWarning; will break when removed. | `try: from timm.layers` with ImportError fallback. |
| A5 | **med** | `weatherlearn/models/fuxi/fuxi.py:161` | `depth=48` hardcoded; lite smoke cannot reduce transformer depth (only embed/spatial). | `depth: int = 48` kwarg; backward compatible. |
| A6 | **med** | Docs / Issue #8 | Discussion conflated FC patch head vs bilinear 720→721; no in-repo clarification. | `docs/fuxi_last_layer.md` + README note + forward comment. |
| A7 | **low** | `tests/models/fuxi/test_fuxi.py` | No coverage that `depth` is wired. | Added `test_depth_configurable`. |

---

## Catalogued but **not** fixed (defer / out of surgical scope)

| ID | Severity | Location | Issue | Proposed fix (for later) |
|----|----------|----------|-------|--------------------------|
| B1 | med | `finetune/finetune_cpu.py` | ONNX-centric; shadows builtin `input`; hard-coded local `pangu_weather_1.onnx` + npy dirs; not using `weatherlearn.models.Pangu`. | Separate script README; rename `input`→`input_upper`; optional path args; document onnx/onnx2torch extras. |
| B2 | med | `inference/pangu.py` | Assumes CUDA ORT provider + `sys.path` hacks + local ERA5 data under `examples/pangu_lite/data`. | Make provider list configurable; use package install instead of `sys.path.append`. |
| B3 | med | `examples/pangu_lite/*.py` | `sys.path.append("../../")` instead of installed package; extra deps (`pandas`, etc.) not in pyproject. | Document example extras; remove path hacks once packaging lands. |
| B4 | low | `weatherlearn/data_utils/coroutine_download.py:49` | `os.path.exists(out_path)` if `out_path is None` → TypeError. | Default `out_path` or early validate. |
| B5 | low | `weatherlearn/models/pangu/pangu.py` `Mlp` | Class defined after first use (works in Python; readability smell). Author plans OOP refactor — leave. | Move `Mlp` above `EarthSpecificBlock` in refactor. |
| B6 | low | `Fuxi.forward` interpolate | No `align_corners=` / `antialias=` — torch defaults; paper silent. | Optional explicit `align_corners=False` for determinism across versions. |
| B7 | low | `CubeEmbedding` assert | Strict size match; no helpful resize path. | Document only; or optional interpolate input (API change — avoid now). |
| B8 | low | README / TODO | FengWu still TODO; no FuXi training recipe; no pretrained FuXi URLs (correct — don't invent). | Keep TODO; add training sketch when ready. |
| B9 | low | `cdsapi` required dep | Core models don't need CDS; install fails if cdsapi unused. | Move to optional `[cds]` extra. |
| B10 | info | License BY-NC-SA 4.0 | Restricts commercial HF Spaces hosting / redistribution. | Keep; disclose on Spaces README (done in hf_space). |
| B11 | info | No FuXi pretrained weights | Expected; Pangu uses external ONNX. | Do not invent URLs. |
| B12 | low | `Pangu` / `Pangu_lite` hardcoded 721×1440 | Cannot construct smaller spatial for unit tests of full model (only lite embed). | Future: img_size kwargs (larger API change). |
| B13 | low | `torch.meshgrid` warning in earth_position_index | Missing `indexing=` arg (PyTorch future). | Add `indexing="ij"` when touching that file. |
| B14 | med | HF Space / examples GPU | Full default FuXi inappropriate for small Spaces. | Space already warns; smoke uses `depth=2` for lite/tiny721. |

---

## Module inventory (audit coverage)

| Path | Role | Notes |
|------|------|-------|
| `weatherlearn/__init__.py` | **added** | Re-export public API |
| `weatherlearn/models/__init__.py` | Public API | Unchanged exports |
| `weatherlearn/models/fuxi/fuxi.py` | FuXi | depth configurable; FC+interpolate clarified |
| `weatherlearn/models/pangu/pangu.py` | Pangu / lite | timm import fixed |
| `weatherlearn/models/pangu/utils/*` | Pad/crop/embed/recovery/mask | OK; meshgrid warning B13 |
| `weatherlearn/data_utils/coroutine_download.py` | CDS async download | B4 |
| `tests/models/fuxi/*` | FuXi unit tests | All pass |
| `tests/models/pangu/*` | Pangu unit tests | `test_pangu` skipped by default |
| `examples/pangu_lite/*` | Train/test/data | B3 |
| `finetune/*` | ONNX finetune | B1 |
| `inference/pangu.py` | ORT inference | B2 |

---

## Test results (recorded)

**Env:** `/workspace/weatherlearn/.venv` — Python 3.13.5, torch 2.14.0+cu130 (CUDA runtime N/A on box), timm 1.0.29, numpy 2.5.3  
**Command:** `cd WeatherLearn && pip install -e . --no-deps && python -m unittest discover -s tests -v`  
**Result:** **Ran 43 tests in ~10s — OK (skipped=1)**  
- Skipped: `test_pangu` (heavy; needs `WEATHERLEARN_RUN_HEAVY=1`)  
- All FuXi tests (9 including new `test_depth_configurable`): **PASS**  
- Public API: `from weatherlearn.models import Fuxi, Pangu, Pangu_lite` and `from weatherlearn import Fuxi` — **OK**

HF Space copy: FuXi discover under `hf_space/WeatherLearn/tests/models/fuxi` — **9 tests OK**.

---

## Breaking pin changes (deps)

| Package | Was | Now | Rationale |
|---------|-----|-----|-----------|
| torch | `==2.1.0` | `>=2.1.0` | Current CUDA/CPU wheels; 2.1.0 unavailable for 3.13 |
| timm | `==0.9.10` | `>=0.9.10` | `timm.layers` layout; Swin V2 Stage still present |
| numpy | `==1.23.5` | `>=1.23.5` | 1.23.5 cannot build on 3.13 |
| cdsapi | `==0.6.1` | `>=0.6.1` | Consistency; still required until optional extra (B9) |
| python | `^3.11` | `^3.11` (unchanged) | — |

Users who need the old lock should pin in their own environment; library consumers get installable ranges.

---

## Explicit non-goals (this PR)

- No full OOP rewrite (author plans separately).  
- No GitHub PR opened from this agent (draft files only).  
- No invented pretrained weight URLs.  
- License BY-NC-SA preserved.
