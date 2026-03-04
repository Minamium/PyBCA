---
title: PyBCA Guide (v0.1)
nav_order: 20
---

# PyBCA Guide (v0.1)

PyBCA は Brownian Cellular Automaton のシミュレーション実装です。
現行の推奨実行経路は `Engine` API です。

- Engine API 詳細: [Engine API (PyBCA)](engine_api.md)
- GUI ツール: [GUI Tools Guide](guitools_guide.md)

## 1. 実行モデル

実行レイヤは次の構成です。

- `PyBCA.api`:
  - `Config`（設定）
  - `Engine`（実行）
  - `Result`（結果）
- `PyBCA.core`:
  - 実シミュレータ実装（`BCA_Simulator`）
  - state/scheme registry
- `PyBCA._legacy`:
  - 旧実装（互換確認用）

## 2. 最小実行例（推奨）

```python
from PyBCA.api import Config, Engine

cfg = Config(
    cellspace_path="Sample/Cellspace/C-Join.yaml",
    rule_paths=("Sample/rule/base-rule.yaml",),
    device="cpu",
    trials=3,
    steps=20,
    global_prob=0.5,
    seed=7,
)

result = Engine(cfg).run()
print(result.current_step)
```

## 3. 省略形 run

```python
from PyBCA.api import run

result = run({
    "cellspace_path": "Sample/Cellspace/test.yaml",
    "rule_paths": ["Sample/rule/base-rule.yaml"],
    "device": "cpu",
    "trials": 2,
    "steps": 8,
})
```

## 4. 特殊イベント付き実行

```python
from PyBCA.api import Config, Engine

cfg = Config(
    cellspace_path="Sample/Cellspace/test.yaml",
    rule_paths=("Sample/rule/base-rule.yaml",),
    spatial_event_file_path="Sample/Specialevent/test_event.py",
    trials=2,
    steps=110,
    event_history_path="/tmp/test_event.jsonl",
    event_history_format="jsonl_trials",
    use_tqdm="false",
)

result = Engine(cfg).run()
```

## 5. 旧来 API（互換）

`BCA_Simulator` 直接実行はまだ利用できます。
新規コードでは `Engine` を推奨します。

```python
from PyBCA.cli_simClass import BCA_Simulator

sim = BCA_Simulator(
    cellspace_path="Sample/Cellspace/C-Join.yaml",
    rule_paths=["Sample/rule/base-rule.yaml"],
    device="cpu",
)
sim.Allocate_torch_Tensors_on_Device()
sim.set_ParallelTrial(3)
sim.run_steps(steps=20, global_prob=0.5)
```

## 6. 入力ファイル

- CellSpace YAML: `Sample/Cellspace/*.yaml`
- Rule YAML: `Sample/rule/*.yaml`
- Special Event `.py`: `Sample/Specialevent/*.py`
- BCL ソース: `Sample/bclfile/*.bcl`（`bcl` で YAML へ変換して利用）

## 7. BCL から PyBCA へ

1. `bcl` で `.bcl` をセル空間 YAML に変換
2. 変換した YAML を `Config.cellspace_path` に指定
3. `Engine.run()` で実行

BCL仕様は [BCL Guide](../bcl/0.1/guide.md) を参照。
