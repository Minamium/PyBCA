---
title: Implementation Parity Audit
parent: Japanese
nav_order: 14
---

# Implementation Parity Audit

このページは、legacy 実装と refactor 後の実装の整合性を 2026-03-26 時点で整理した監査メモです。

## 1. 監査対象

- `src/PyBCA/_legacy/cli_simClass.py`
- `src/PyBCA/core/simulator.py`
- `src/PyBCA/_legacy/lib.py`
- `src/PyBCA/core/io.py`
- `src/PyBCA/api/*`
- `src/PyBCA/cli_simClass.py`
- `src/PyBCA/lib.py`
- `src/PyBCA/guitools.py`
- `src/BCL/compiler.py`
- `tests/simulator_parity.py`

## 2. 直接比較の結論

### `core/simulator.py` と legacy simulator

- 移植先 `src/PyBCA/core/simulator.py` は、更新ロジックを保ったまま import と docstring を整理した形です
- `BCASimulator = BCA_Simulator` の別名が追加されています
- 試験面では step 単位、`run_steps`、`Engine.run()` の全てで parity を確認しました

### `core/io.py` と legacy `lib.py`

- 監査時点で実質同一です
- CellSpace/Rule/Event のロード系は新旧で差分を観測しませんでした

### 公開ラッパ

- `src/PyBCA/cli_simClass.py` は新 core simulator を公開しつつ `LegacyBCA_Simulator` も再公開します
- `src/PyBCA/lib.py` はまだ legacy utility 実装のラッパです
- `src/PyBCA/guitools.py` はまだ legacy GUI 実装のラッパです

これは不具合ではなく、現行では互換維持のための構成です。
ただし「すべてが新 core 実装へ移行済み」とは言えません。

## 3. 実行検証

CPU で以下を実行し、成功を確認しました。

```bash
PYTHONPATH=src python tests/simulator_parity.py
```

確認内容:

- 20 ケース
- `step()` の逐次比較
- `run_steps()` の比較
- `Engine.run()` の比較
- `event_history` の比較
- `jsonl_trials`, `jsonl_trials_dict`, `jsonl` の export 比較
- `BNN`, `Join/Fork`, `C-Join err`, `BCA-IP`, special event を含む Sample 群

結果:

- 現行 parity suite は全件成功

## 4. BCL 側の整合性

以下を確認済みです。

- `Sample/bclfile/*.bcl` はすべて parse 可能
- `Sample/bclfile/BNN.bcl` の出力は `Sample/Cellspace/BNN.yaml` と一致
- `Sample/bclfile/C-Join_from_JF.bcl` の出力は `Sample/Cellspace/2-way_C-Join.yaml` と一致

したがって、現行 sample 範囲では BCL compiler と CellSpace YAML 資産の整合性も取れています。

## 5. 現時点での境界

- parity 検証は CPU 実行で確認済みです
- GPU parity はこの監査ページの範囲では再実行していません
- GUI Viewer は legacy 実装に依存します
- utility 関数群も legacy 実装を再公開しています

## 6. 実務上の解釈

新規のシミュレーション実行コードは `PyBCA.api.Engine` を使って問題ありません。
一方で、GUI や utility import まで含めて完全に新実装へ置き換わったわけではないため、次の認識が正確です。

- CA 更新理論と Sample 上の実行結果は parity あり
- 公開 API の一部はまだ互換ラッパ
- GUI は未移植領域を含む
