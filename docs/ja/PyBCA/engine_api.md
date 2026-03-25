---
title: Engine API
parent: Japanese
nav_order: 12
---

# Engine API

`Engine` は現行 PyBCA のトップレベル実行 API です。
設定は `Config` に集約され、`Engine.run()` は `Result` を返します。

実装参照:

- `src/PyBCA/api/config.py`
- `src/PyBCA/api/engine.py`
- `src/PyBCA/api/result.py`
- `src/PyBCA/core/states/state_bca.py`
- `src/PyBCA/core/schemes/bca_default.py`

## 1. Quick Start

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
    use_tqdm="false",
)

result = Engine(cfg).run()
print(result.current_step, result.elapsed_sec)
```

## 2. `Config` フィールド一覧

| Field | Default | 型/意味 |
| --- | --- | --- |
| `model` | `bca` | `Model` または文字列 |
| `scheme` | `default` | `Scheme` または文字列 |
| `backend` | `torch` | `Backend` または文字列 |
| `cellspace_path` | required | CellSpace YAML パス |
| `rule_paths` | required | Rule YAML パス列 |
| `device` | `cuda` | 文字列。`torch` 利用可能デバイス名 |
| `trials` | `1` | 並列試行数 |
| `steps` | `1` | 実行 step 数 |
| `global_prob` | `1.0` | 大域確率ゲート |
| `seed` | `0` | 乱数 seed |
| `spatial_event_file_path` | `None` | Special Event `.py` |
| `gui_mode` | `False` | GUI 向け戻り値分岐フラグ |
| `use_tqdm` | `true` | 文字列 enum |
| `trial_constant_sweep` | `None` | alias ごとの確率 sweep |
| `state_gate_enable` | `False` | 大域状態ゲート有効化 |
| `state_gate_interval` | `500` | 状態ゲート適用間隔 |
| `debug` | `False` | `simulator.step()` の debug 引数 |
| `debug_per_trial` | `False` | trial 別 debug 引数 |
| `log_level` | `info` | `debug/info/warning/error` |
| `event_history_path` | `None` | イベント履歴保存先 |
| `event_history_format` | `jsonl_trials` | 出力形式 |
| `event_history_deduplicate` | `True` | step 重複除去 |
| `event_history_return_df` | `False` | 保存ヘルパの戻り値として DataFrame を返すか |

## 3. 受理される alias

`Config.__post_init__()` では以下の別名が許可されます。

- `model`: `default -> bca`
- `scheme`: `bca -> default`
- `backend`: `pytorch -> torch`
- `use_tqdm`: `1 -> true`, `0 -> false`
- `log_level`: `warn -> warning`, `err -> error`

`use_tqdm` は bool ではなく enum/文字列として処理されます。

## 4. バリデーション

`Config` 生成時に次を検査します。

- `cellspace_path` は空不可
- `rule_paths` は 1 件以上必須
- `trials >= 1`
- `steps >= 0`
- `global_prob` は `[0, 1]`
- `state_gate_interval >= 1`

補足:

- `device` の存在確認までは行いません
- パスの存在確認は `Config` ではなく、実際のロード時に失敗します

## 5. `Engine.__init__`

```python
engine = Engine(config, apply_logging_flag=True)
```

挙動:

1. 必要なら `apply_logging(config)` を適用
2. `build_state(config)` で `BCA_Simulator` を構築
3. `build_stepper(config, state)` で step 関数を構築

`apply_logging_flag=False` にすると logger 設定を自前で制御できます。

## 6. `Engine.run()` の正確な処理順序

`Engine.run()` は次の順番で進みます。

1. state に保持された `simulator` を取得
2. `steps > 0` なら `range(steps)` を反復
3. `use_tqdm == true` なら `tqdm` でラップ
4. 各 step で `self.stepper(step_idx)` を呼ぶ
5. `event_history_path` が指定されていれば `save_event_histry_for_dataframe(...)` を呼ぶ
6. `Result` を返す

重要点:

- `step_idx` 自体は現在の default stepper では使っていません
- step 実行本体は `simulator.step(...)` です
- `Result.current_step` は `simulator._current_step` を参照します

## 7. `Result`

`Result` のフィールドは次の通りです。

- `simulator`
- `current_step`
- `elapsed_sec`
- `event_history`
- `meta`

`meta` には `{"config": config.as_dict}` が入ります。

`event_history` については注意が必要です。

- `event_history_path is None` の場合、`Result.event_history` は `None`
- ただし `result.simulator.event_history` には trial ごとの生履歴が残ります
- 出力ヘルパの戻り値をメモリ上で取得したい場合は `event_history_return_df=True` を使います

## 8. Special Event ファイル

`spatial_event_file_path` で参照する `.py` には `events` を定義します。

基本形:

```python
events = [
    ("name", (x, y), ref_state, (wx, wy), write_state),
]
```

拡張形:

```python
events = [
    ("name", (x, y), ref_state, (wx, wy), write_state, prob, start_step, end_step),
]
```

特殊イベントは、通常の CA 更新の後に `apply_spatial_events()` で適用されます。

## 9. `trial_constant_sweep`

`trial_constant_sweep` は、読み込んだ rule の `probability: *alias` に対して trial ごとの確率列を作るための設定です。

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

実際には `set_ParallelTrial()` 内で `[T, N]` の `rule_probs_tensor` が構築されます。

## 10. `event_history` 出力形式

`event_history_format` は以下を受理します。

- `jsonl_trials`
- `jsonl_trials_dict`
- `jsonl`
- `csv`
- `yaml`
- `parquet`

`Engine` 側の既定値は `jsonl_trials` です。
一方で、基底 helper `save_event_histry_for_dataframe()` の関数既定値は `parquet` です。
`Engine` を通さず simulator を直接使う場合は、この差を意識してください。

## 11. ログ設定

`apply_logging(config)` は `PyBCA` logger に `StreamHandler` を 1 本だけ追加し、`log_level` に合わせて level を設定します。

既定 formatter:

```text
%(levelname)s : %(message)s
```

## 12. よくある利用パターン

### パスを絶対化する

```python
from pathlib import Path
from PyBCA.api import Config

root = Path(__file__).resolve().parents[1]
cfg = Config(
    cellspace_path=str(root / "Sample" / "Cellspace" / "C-Join.yaml"),
    rule_paths=(str(root / "Sample" / "rule" / "base-rule.yaml"),),
)
```

### メモリ上でイベント履歴 DataFrame を得る

```python
result = Engine(cfg).run()
df = result.simulator.save_event_histry_for_dataframe(path=None, return_df=True)
```

### 進捗バーを消す

```python
cfg = Config(
    cellspace_path="Sample/Cellspace/test.yaml",
    rule_paths=("Sample/rule/base-rule.yaml",),
    use_tqdm="false",
)
```
