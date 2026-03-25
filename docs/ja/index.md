---
title: Japanese
nav_order: 10
has_children: true
permalink: /ja/
---

# 日本語ドキュメント

更新基準日: 2026-03-26

PyBCA は Brownian Cellular Automaton のシミュレータと、そのセル空間記述用の BCL を含むリポジトリです。
現行の推奨実行経路は `PyBCA.api.Engine` です。

## 主要ページ

- [PyBCA Guide](PyBCA/PyBCA_guide.md)
- [Engine API](PyBCA/engine_api.md)
- [GUI Tools Guide](PyBCA/guitools_guide.md)
- [Implementation Parity Audit](PyBCA/parity_audit.md)
- [BCL Guide](bcl/0.1/guide.md)
- [PyBCA Dev Memo](dev-memo/PyBCAdevMemo.md)

## 実装監査の要点

- `src/PyBCA/core/simulator.py` は legacy のセルオートマトン更新理論を保持した移植先です。
- `src/PyBCA/api` は `Config -> Engine -> Result` の薄い実行ラッパで、更新則そのものは `core` を呼びます。
- `src/PyBCA/lib.py` と `src/PyBCA/guitools.py` は、現時点では legacy 実装を再公開する互換レイヤです。
- CPU 上では `tests/simulator_parity.py` により legacy/core/Engine の parity を検証済みです。

## 利用の起点

- 新規コード: `PyBCA.api.Engine`
- 旧コード互換: `PyBCA.cli_simClass.BCA_Simulator`
- BCL 変換: `bcl INPUT.bcl -o OUTPUT.yaml`
- GUI 編集: `bcl-editor`, `python -m BCL.rule_editor`, `python -m PyBCA._legacy.guitools`
