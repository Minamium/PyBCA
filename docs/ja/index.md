---
title: Japanese
nav_order: 10
has_children: true
permalink: /ja/
---

# 日本語ドキュメント

更新日: 2026-03-26

PyBCA は Brownian Cellular Automaton のシミュレータと、その CellSpace 記述用言語 BCL を含む OSS です。
新規コードでは `PyBCA.api.Engine` を使うのが推奨です。

## 主要ページ

- [PyBCA Guide](PyBCA/PyBCA_guide.md)
- [Simulation Logic](PyBCA/simulation_logic.md)
- [States and Rules](PyBCA/states_and_rules.md)
- [Engine API](PyBCA/engine_api.md)
- [GUI Tools Guide](PyBCA/guitools_guide.md)
- [BCL Guide](bcl/0.1/guide.md)

## この docs の方針

- 使い方を先に説明する
- 計算ロジックは独立ページで整理する
- BCL と GUI は入力作成・検証の手段として説明する
- 移植履歴や parity 監査は補助資料に留める

## 利用の起点

- 新規コード: `PyBCA.api.Engine`
- 旧コード互換: `PyBCA.cli_simClass.BCA_Simulator`
- BCL 変換: `bcl INPUT.bcl -o OUTPUT.yaml`
- GUI 編集: `bcl-editor`, `python -m BCL.rule_editor`, `python -m PyBCA._legacy.guitools`

## 補助資料

- [Implementation Parity Audit](PyBCA/parity_audit.md)
- [PyBCA Dev Memo](dev-memo/PyBCAdevMemo.md)
