---
title: Engine API (PyBCA)
nav_order: 25
---

# Engine API (PyBCA)

`Engine` は PyBCA の実行エントリです。
`Config` で条件を定義し、`Engine.run()` が `Result` を返します。

実装基準:

- `src/PyBCA/api/config.py`
- `src/PyBCA/api/engine.py`
- `src/PyBCA/api/result.py`

## Quick Start

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
print(result.current_step, result.elapsed_sec)
```

## Config フィールド

| Field | Default | Notes |
| --- | --- | --- |
| `model` | `bca` | Enum: `bca` |
| `scheme` | `default` | Enum: `default` |
| `backend` | `torch` | Enum: `torch` |
| `cellspace_path` | required | セル空間 YAML |
| `rule_paths` | required | 1件以上必須 |
| `device` | `cuda` | 例: `cpu`, `cuda` |
| `trials` | `1` | `>=1` |
| `steps` | `1` | `>=0` |
| `global_prob` | `1.0` | `0.0-1.0` |
| `seed` | `0` | int |
| `spatial_event_file_path` | `None` | 特殊イベント `.py` |
| `gui_mode` | `False` | 通常は `False` |
| `use_tqdm` | `true` | `true` / `false` |
| `trial_constant_sweep` | `None` | ルール確率の trial sweep |
| `state_gate_enable` | `False` | 大域状態ゲート |
| `state_gate_interval` | `500` | `>=1` |
| `debug` | `False` | デバッグ出力 |
| `debug_per_trial` | `False` | trial別デバッグ |
| `log_level` | `info` | `debug/info/warning/error` |
| `event_history_path` | `None` | 出力先 |
| `event_history_format` | `jsonl_trials` | `jsonl_trials` など |
| `event_history_deduplicate` | `True` | step重複除去 |
| `event_history_return_df` | `False` | DataFrame返却 |

### Enum の別名

`Config.__post_init__` で以下の別名が許可されます。

- `model`: `default -> bca`
- `scheme`: `bca -> default`
- `backend`: `pytorch -> torch`
- `use_tqdm`: `1 -> true`, `0 -> false`
- `log_level`: `warn -> warning`, `err -> error`

## バリデーション

`Config` 生成時の制約:

- `cellspace_path` は必須
- `rule_paths` は空不可
- `trials >= 1`
- `steps >= 0`
- `global_prob in [0, 1]`
- `state_gate_interval >= 1`

## Engine.run の挙動

`Engine.run()` の処理順序:

1. `build_state(config)` で `BCA_Simulator` 構築・テンソル確保・trial設定
2. `steps > 0` の場合、stepper (`core/schemes/bca_default.py`) を `steps` 回実行
3. `event_history_path` 指定時は `save_event_histry_for_dataframe(...)` を実行
4. `Result` を返す

## Result

`Result`（`src/PyBCA/api/result.py`）のフィールド:

- `simulator`
- `current_step`
- `elapsed_sec`
- `event_history`
- `meta`

## 特殊イベント

`spatial_event_file_path` で `.py` を指定し、`events` リストを定義します。

```python
events = [
    ("name", (x, y), ref_state, (wx, wy), write_state),
]
```

拡張形式（確率・有効ステップ範囲あり）:

```python
events = [
    ("name", (x, y), ref_state, (wx, wy), write_state, prob, start_step, end_step),
]
```

## trial_constant_sweep

YAML側で `probability: *alias` を使うルールに対して trial 別確率を入れられます。

```python
cfg = Config(
    cellspace_path="Sample/Cellspace/BNN.yaml",
    rule_paths=("Sample/rule/base-rule.yaml", "Sample/rule/Join_fork.yaml"),
    trials=4,
    steps=15,
    trial_constant_sweep={
        "join_err_0_input": {"base": 0.0, "delta": 0.001},
        "join_err_1_input": {"base": 0.0, "delta": 0.0005},
    },
)
```

## event_history 出力

`event_history_path` 指定時に保存されます。

- `jsonl_trials`
- `jsonl_trials_dict`
- `jsonl`
- `csv`
- `yaml`
- `parquet`

```python
cfg = Config(
    cellspace_path="Sample/Cellspace/test.yaml",
    rule_paths=("Sample/rule/base-rule.yaml",),
    spatial_event_file_path="Sample/Specialevent/test_event.py",
    steps=110,
    trials=2,
    event_history_path="/tmp/test_event.jsonl",
    event_history_format="jsonl_trials",
)

result = Engine(cfg).run()
```

## ショートカット API

```python
from PyBCA.api import run

result = run({
    "cellspace_path": "Sample/Cellspace/C-Join.yaml",
    "rule_paths": ["Sample/rule/base-rule.yaml"],
    "steps": 10,
    "trials": 1,
    "device": "cpu",
})
```

## パス運用の注意

`Config` はパスを自動補正しません。実運用では絶対パス化を推奨します。

```python
from pathlib import Path
from PyBCA.api import Config

root = Path(__file__).resolve().parents[1]
cfg = Config(
    cellspace_path=str(root / "Sample" / "Cellspace" / "C-Join.yaml"),
    rule_paths=(str(root / "Sample" / "rule" / "base-rule.yaml"),),
)
```
