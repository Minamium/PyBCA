---
title: PyBCA Guide
parent: Japanese
nav_order: 11
---

# PyBCA Guide

このガイドは、現行実装に沿って PyBCA を実行するための入口です。
新規コードでは `PyBCA.api.Engine` を使い、legacy API は互換用途として扱います。

関連ページ:

- [Engine API](engine_api.md)
- [GUI Tools Guide](guitools_guide.md)
- [Implementation Parity Audit](parity_audit.md)
- [BCL Guide](../bcl/0.1/guide.md)

## 1. パッケージ構成

- `src/PyBCA/api`
  `Config`, `Engine`, `Result` を提供する公開 API 層です。
- `src/PyBCA/core`
  実際のシミュレータ実装、state builder、scheme builder を含みます。
- `src/PyBCA/_legacy`
  旧実装の退避先です。互換確認や GUI の現行実装で参照されています。
- `src/BCL`
  BCL compiler、editor、rule editor を含みます。

## 2. インストールと依存

基本依存は `pyproject.toml` に定義されています。

```bash
pip install -e .
```

含まれる主要依存:

- `numpy`
- `torch>=2.0`
- `scipy`
- `pyyaml`

GUI を使う場合は追加で `PySide6` が必要です。

```bash
pip install PySide6
```

開発中にインストールせず直接実行する場合は、`PYTHONPATH=src` を付けてください。

## 3. 推奨ワークフロー

1. CellSpace YAML を用意する
2. Rule YAML を 1 つ以上指定する
3. 必要なら Special Event `.py` を用意する
4. `Config` を作って `Engine.run()` を呼ぶ
5. 必要なら `event_history_path` を指定してイベント履歴を書き出す

BCL を使う場合は、先に `.bcl` を YAML に変換してから `Config.cellspace_path` に渡します。

## 4. 最小実行例

```python
from pathlib import Path

from PyBCA.api import Config, Engine

root = Path.cwd()

cfg = Config(
    cellspace_path=str(root / "Sample" / "Cellspace" / "C-Join.yaml"),
    rule_paths=(str(root / "Sample" / "rule" / "base-rule.yaml"),),
    device="cpu",
    trials=3,
    steps=20,
    global_prob=0.5,
    seed=7,
    use_tqdm="false",
)

result = Engine(cfg).run()
print(result.current_step)
```

ポイント:

- `Config` はパスを自動補正しません
- `device` の既定値は `"cuda"` です
- `steps=0` なら初期化だけ行い、step 実行はしません

## 5. `run(dict)` の簡略形

```python
from PyBCA.api import run

result = run(
    {
        "cellspace_path": "Sample/Cellspace/test.yaml",
        "rule_paths": ["Sample/rule/base-rule.yaml"],
        "device": "cpu",
        "trials": 2,
        "steps": 8,
        "global_prob": 0.5,
        "seed": 13,
        "use_tqdm": "false",
    }
)
```

辞書入力でも最終的には `Config(**dict)` が使われます。

## 6. Special Event と event history

Special Event を使うときは `spatial_event_file_path` を指定します。
イベント履歴をファイル保存したい場合は `event_history_path` と `event_history_format` を指定します。

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
    event_history_deduplicate=True,
    event_history_return_df=True,
    use_tqdm="false",
)

result = Engine(cfg).run()
print(type(result.event_history))
```

厳密な挙動:

- `set_ParallelTrial()` 時点で `simulator.event_history` が trial ごとに初期化されます
- `Engine.run()` は `event_history_path` が指定された場合だけ `save_event_histry_for_dataframe(...)` を呼びます
- そのため `Result.event_history` は、出力ヘルパの戻り値を反映します
- 生の履歴辞書は `result.simulator.event_history` から確認できます

## 7. Trial ごとの確率 sweep

Rule YAML 側で `probability: *alias` を使っている場合、`trial_constant_sweep` で trial ごとに異なる確率を与えられます。

```python
cfg = Config(
    cellspace_path="Sample/Cellspace/BNN.yaml",
    rule_paths=("Sample/rule/base-rule.yaml", "Sample/rule/Join_fork.yaml"),
    trials=4,
    steps=15,
    trial_constant_sweep={
        "join_err_0_input": {"base": 0.0, "delta": 0.001},
        "join_err_1_input": {"base": 0.0, "delta": 0.0005},
        "fork_err_0_input": {"base": 0.0, "delta": 0.001},
    },
)
```

`alias` 名は、ロード済み rule 内の `constants` と一致していなければなりません。
一致しない場合は `ValueError` になります。

## 8. Legacy 互換 API

`src/PyBCA/cli_simClass.py` では、現行 `BCA_Simulator` と legacy simulator の両方にアクセスできます。

```python
from PyBCA.cli_simClass import BCA_Simulator, LegacyBCA_Simulator

sim = BCA_Simulator(
    cellspace_path="Sample/Cellspace/C-Join.yaml",
    rule_paths=["Sample/rule/base-rule.yaml"],
    device="cpu",
)
sim.Allocate_torch_Tensors_on_Device()
sim.set_ParallelTrial(3)
sim.run_steps(steps=20, global_prob=0.5)
```

使い分け:

- `BCA_Simulator`: `src/PyBCA/core/simulator.py`
- `LegacyBCA_Simulator`: `src/PyBCA/_legacy/cli_simClass.py`
- `PyBCA.lib`: まだ legacy utility 実装を再公開
- `PyBCA.guitools`: まだ legacy GUI 実装を再公開

## 9. Sample ファイルの場所

- CellSpace YAML: `Sample/Cellspace/*.yaml`
- Rule YAML: `Sample/rule/*.yaml`
- Special Event: `Sample/Specialevent/*.py`
- BCL source: `Sample/bclfile/*.bcl`

検証に使う主なスクリプト:

- parity: `PYTHONPATH=src python tests/simulator_parity.py`
- sample runner:
  `tests/BNN.py`,
  `tests/BCA-IP.py`,
  `tests/Join_acc.py`,
  `tests/Fork_acc.py`

## 10. どの資料を見ればよいか

- `Engine` の引数仕様を見たい: [Engine API](engine_api.md)
- BCL 文法を見たい: [BCL Guide](../bcl/0.1/guide.md)
- GUI 操作を見たい: [GUI Tools Guide](guitools_guide.md)
- legacy/new の整合性を見たい: [Implementation Parity Audit](parity_audit.md)
