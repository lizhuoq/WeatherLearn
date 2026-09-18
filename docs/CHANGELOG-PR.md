# Changelog — branch `fix/packaging-and-tests`

## 中文摘要

面向「完整复刻」审计后的**最小侵入**修复，便于上游 PR：

- **打包**：补齐 `weatherlearn` / `tests` 的 `__init__.py`，`pip install -e .` 与 `python -m unittest discover` 可直接使用。
- **依赖**：`pyproject.toml` 将 `torch` / `timm` / `numpy` / `cdsapi` 从精确钉死改为可安装的 `>=` 下限（原钉死在 Python 3.13 上无法安装）；版本号 `0.1.0` → `0.1.1`。
- **timm**：Pangu 改为优先 `timm.layers`，旧路径作 fallback，消除弃用警告。
- **FuXi**：`depth` 可配置，默认仍为 `48`（向后兼容）；补充 Issue #8 说明文档（FC patch + 双线性插值，与论文一致）。
- **测试**：完整 `Pangu()` 默认 skip，需 `WEATHERLEARN_RUN_HEAVY=1`；新增 FuXi `depth` 单测。本地：43 tests，1 skipped，全部通过。

**未改**：整体 OOP 重构、预训练权重 URL、finetune/inference 脚本大改（见 AUDIT.md 延期项）。

## English summary

Surgical packaging/test fixes after a full-library audit (PR-ready, no OOP rewrite):

- **Packaging:** add missing `__init__.py` so editable install and `unittest discover` work.
- **Deps:** relax exact pins to installable `>=` ranges for modern Python/torch/timm; bump `0.1.0` → `0.1.1`.
- **timm:** prefer `timm.layers` with fallback for older timm.
- **FuXi:** configurable `depth` (default **48**); docs clarifying Issue #8 (Linear patch FC + bilinear interpolate matches the paper).
- **Tests:** skip full `Pangu()` unless `WEATHERLEARN_RUN_HEAVY=1`; add FuXi depth unit test. **43 tests, 1 skipped, OK.**

Deferred items (ONNX finetune path hacks, cdsapi optional extra, meshgrid indexing, etc.) are listed in `AUDIT.md`.

## Dependency pin changes

| Package | Before | After |
|---------|--------|-------|
| torch | ==2.1.0 | >=2.1.0 |
| timm | ==0.9.10 | >=0.9.10 |
| numpy | ==1.23.5 | >=1.23.5 |
| cdsapi | ==0.6.1 | >=0.6.1 |

## Public API

Unchanged:

```python
from weatherlearn.models import Fuxi, Pangu, Pangu_lite
# also:
from weatherlearn import Fuxi, Pangu, Pangu_lite
```

New optional kwarg: `Fuxi(..., depth=48)`.
