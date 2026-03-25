---
title: PyBCA Guide
parent: Japanese
nav_order: 11
---

# PyBCA Guide

本ガイドは、PyBCA を用いて数値実験を構成・実行するための導入資料である。
まず標準的な利用手順を示し、その後に計算ロジックおよび API 詳細へ接続する。

関連ページ

- [Simulation Logic](simulation_logic.md)
- [Engine API](engine_api.md)
- [GUI Tools Guide](guitools_guide.md)
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

基本依存は `pyproject.toml` に定義されている。

```bash
pip install -e .
```

含まれる主要依存:

- `numpy`
- `torch>=2.0`
- `scipy`
- `pyyaml`

GUI を用いる場合は追加で `PySide6` が必要である。

```bash
pip install PySide6
```

インストールせずに直接実行する場合は、`PYTHONPATH=src` を付与する。

## 3. 標準的な実行手順

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

留意点:

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

辞書入力であっても、最終的には `Config(**dict)` が用いられる。

## 6. 入力と出力

- CellSpace 入力:
  `Config.cellspace_path`
- Rule 入力:
  `Config.rule_paths`
- Special Event 入力:
  `Config.spatial_event_file_path`
- event history 出力:
  `Config.event_history_path`, `Config.event_history_format`

主な出力形式:

- `jsonl_trials`
- `jsonl_trials_dict`
- `jsonl`
- `csv`
- `yaml`
- `parquet`

## 7. Special Event と event history

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

## 8. trial ごとの sweep

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

`alias` 名は、ロード済み rule 内の `constants` と一致している必要がある。
一致しない場合は `ValueError` が送出される。

この仕組みは、エラー率や受理率を sweep する数値実験に用いられる。
ただし、シミュレータ自身が内部変数として温度を持つわけではなく、確率と有効温度との対応づけは解析側で与える必要がある。

## 9. 計算ロジック

セル更新順序、競合解決、state gate、special event の適用順は [Simulation Logic](simulation_logic.md) にまとめています。

## 10. Sample ファイルの場所

- CellSpace YAML: `Sample/Cellspace/*.yaml`
- Rule YAML: `Sample/rule/*.yaml`
- Special Event: `Sample/Specialevent/*.py`
- BCL source: `Sample/bclfile/*.bcl`

付属の実行スクリプト:

- sample runner:
  `tests/BNN.py`,
  `tests/BCA-IP.py`,
  `tests/Join_acc.py`,
  `tests/Fork_acc.py`

## 11. 互換 API

既存コードとの互換性のため `PyBCA.cli_simClass.BCA_Simulator` も利用可能である。
ただし、新規コードに対する参照点としては `PyBCA.api.Engine` を採用する方が明快である。

## 12. どの資料を見ればよいか

- 実行例から入りたい: [PyBCA Guide](PyBCA_guide.md)
- 更新則や計算順を知りたい: [Simulation Logic](simulation_logic.md)
- 引数仕様を見たい: [Engine API](engine_api.md)
- BCL 文法を見たい: [BCL Guide](../bcl/0.1/guide.md)
- GUI 操作を見たい: [GUI Tools Guide](guitools_guide.md)
