---
title: Engine API (PyBCA)
nav_order: 25
---

# Engine API (PyBCA)

`Engine` は PyBCA のシミュレーション実行を統一する高水準 API です。
`Config` に実行条件をまとめ、`Engine.run()` が `Result` を返します。

このページは `Engine` と `Config` の使い方、引数の意味、出力の扱いをまとめます。

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

`cellspace_path` と `rule_paths` は必須です。相対パスは実行カレントディレクトリ基準なので、必要に応じて絶対パス化してください。

## Config フィールド

`Config` は dataclass です。主なフィールドと既定値は以下の通りです。

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| model | str or Model | `bca` | 通常は `bca` 固定。|
| scheme | str or Scheme | `default` | 通常は `default` 固定。|
| backend | str or Backend | `torch` | 既定は PyTorch 実装。|
| cellspace_path | str | required | セル空間 YAML。|
| rule_paths | tuple or list | required | ルール YAML の配列。|
| device | str | `cuda` | `cpu` も可。|
| trials | int | `1` | 並列試行数。|
| steps | int | `1` | 実行ステップ数。|
| global_prob | float | `1.0` | 0.0-1.0。|
| seed | int | `0` | 乱数シード。|
| spatial_event_file_path | str or None | `None` | 特殊イベント定義 `.py`。|
| gui_mode | bool | `False` | GUI向けフラグ。|
| use_tqdm | str or UseTqdm | `true` | `true` または `false`。|
| trial_constant_sweep | dict or None | `None` | ルール確率の trial sweep。|
| state_gate_enable | bool | `False` | 大域状態ゲートの有効化。|
| state_gate_interval | int | `500` | ゲート適用間隔。|
| debug | bool | `False` | 追加デバッグ。|
| debug_per_trial | bool | `False` | trial別デバッグ。|
| log_level | str or LogLevel | `info` | `debug`/`info`/`warning`/`error`。|
| event_history_path | str or None | `None` | `event_history` 出力先。|
| event_history_format | str | `jsonl_trials` | `jsonl_trials` など。|
| event_history_deduplicate | bool | `True` | ステップ重複除去。|
| event_history_return_df | bool | `False` | DataFrame を返すか。|

### Path の取り扱い

`cellspace_path` や `rule_paths` は `Config` が自動的に絶対パス化しません。
CLI 以外の実行では `Path.resolve()` を使った絶対パス化が安全です。

```python
from pathlib import Path
from PyBCA.api import Config

root = Path(__file__).resolve().parents[1]

cfg = Config(
    cellspace_path=str(root / "Sample" / "Cellspace" / "C-Join.yaml"),
    rule_paths=(str(root / "Sample" / "rule" / "base-rule.yaml"),),
)
```

## Engine の実行

`Engine.run()` は以下の結果を返します。

- `result.simulator`: 実際に走った `BCA_Simulator` インスタンス
- `result.current_step`: 最終ステップ
- `result.elapsed_sec`: 実行時間(秒)
- `result.event_history`: 出力を要求した場合の履歴
- `result.meta`: `Config` を含むメタ情報

```python
from PyBCA.api import Engine, Config

cfg = Config(
    cellspace_path="Sample/Cellspace/test.yaml",
    rule_paths=("Sample/rule/base-rule.yaml",),
    steps=8,
    trials=2,
    device="cpu",
)

result = Engine(cfg).run()
print(result.simulator.TCHW.shape)
```

## 特殊イベント

`spatial_event_file_path` に `.py` を指定します。
イベントファイルは `events` というリストを定義します。

旧形式:

```python
events = [
    ("name", (x, y), ref_state, (x2, y2), write_state),
]
```

拡張形式:

```python
events = [
    ("name", (x, y), ref_state, (x2, y2), write_state, prob, start_step, end_step),
]
```

## trial_constant_sweep

`trial_constant_sweep` はルール確率を trial ごとに変化させます。
ルール YAML 内で `probability: *alias` を使っている場合に有効です。

```python
cfg = Config(
    cellspace_path="Sample/Cellspace/BNN.yaml",
    rule_paths=("Sample/rule/base-rule.yaml", "Sample/rule/Join_fork.yaml"),
    steps=15,
    trials=4,
    trial_constant_sweep={
        "join_err_0_input": {"base": 0.0, "delta": 0.001},
        "join_err_1_input": {"base": 0.0, "delta": 0.0005},
    },
)
```

`base + trial_index * delta` が確率となり、0-1 へクランプされます。

## event_history 出力

`event_history_path` を指定すると出力されます。
`event_history_format` は以下に対応します。

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

## 省略形 run

`PyBCA.api.run` は `Config` または辞書を受け取って実行できます。

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
