---
title: States and Rules
parent: Japanese
nav_order: 14
---

# 状態値と規則ファミリ

本ページでは、PyBCA における状態値の扱い、追加状態を用いた遷移規則の設計、および `Join/Fork` 系サンプルの読み方を、現行実装に即して整理する。
あわせて、BCL Editor、Rule Editor、CellSpace Viewer がどの範囲まで設計と検証を支援できるかを明記する。

関連ページ

- [PyBCA Guide](PyBCA_guide.md)
- [Simulation Logic](simulation_logic.md)
- [GUI Tools Guide](guitools_guide.md)

## 1. 実装上の状態モデル

PyBCA の CellSpace と Rule は、実装上はいずれも整数状態の格子として扱われる。
したがって、`Join` や `Fork` のような素子はカーネルに内蔵された特別な演算子ではなく、追加状態を含む local rule の集合として構成される。

実装上、特に注意すべき点は次の通りである。

- `0` は vacant を表す基本状態である
- `-1` は CellSpace や `prev` 側では通常の整数状態として照合される
- ただし `next` 側の `-1` は「状態 `-1` を書く」意味ではなく、書き戻し対象から除外する番兵値である
- 内部テンソルは `int8` を用いるため、実用上の状態値範囲は `-128..127` である

従って、`0` と `-1` を除けば、任意の追加状態は設計者が意味を与えるユーザー定義状態とみなしてよい。

## 2. 基本状態と sample 上の慣用

公開 docs でまず説明すべき基本状態は次の通りである。

| 状態値 | 通常の意味 |
| --- | --- |
| `0` | vacant |
| `1` | wire |
| `2` | token |
| `-1` | recycle-bin connection point |

これに対し、付属 sample は追加状態を局所素子の内部状態として用いている。
とくに `Sample/rule/Join_fork.yaml` では、以下の慣用が採用されている。

| 状態値 | `Join_fork.yaml` における慣用的役割 |
| --- | --- |
| `3` | Join 系素子の安定中心状態 |
| `4` | Join 遷移直後の一時状態 |
| `5` | Fork 系素子の安定中心状態 |
| `6` | Fork 遷移直後の一時状態 |

ここで重要なのは、これらの意味が実装で予約されているわけではないことである。
あくまで sample がそう定義しているにすぎず、別のモデルでは別の整数に別の意味を与えてよい。

## 3. Rule YAML は単一 rule ではなく rule family を表す

Brownian circuit の素子は、実際には 1 本の rule ではなく、複数の向き、誤り則、状態変換を含む rule family として記述されることが多い。

Rule YAML の主要セクションは次の 3 つである。

- `constants`
  確率パラメータの定数定義
- `conv`
  state gate により適用される大域状態変換
- `rules`
  local `prev -> next` 規則の列

この分離により、設計者は次の三層を独立に扱える。

- CellSpace 上の配置と初期状態
- local rule による近傍遷移
- `conv` による一時状態から安定状態への緩和

## 4. `Join_fork.yaml` の読み方

`Sample/rule/Join_fork.yaml` は、追加状態と rule family を用いた設計例である。

### 4.1 確率定数

先頭の `constants` では、誤り則に対応する確率 alias が定義されている。

- `join_err_0_input`
- `join_err_1_input`
- `fork_err_0_input`

これらは Rule YAML 内で `probability: *alias` として参照され、さらに `Config.trial_constant_sweep` により trial ごとに sweep できる。

### 4.2 一時状態の緩和

`conv` では次の変換が定義されている。

- `4 -> 3`
- `6 -> 5`

これは、`Join` および `Fork` の遷移直後に導入された一時状態を、state gate により安定状態へ戻すための設計である。
したがって、`Join_fork.yaml` を意図どおりに動かすには、通常は `state_gate_enable=True` を併用する。

### 4.3 規則群の構成

同ファイルの rule 群は、方向ごとに次の family に分かれている。

- `200-203`: 右向き Join
- `204-205`: 右向き Fork
- `206-209`: 上向き Join
- `210-211`: 上向き Fork
- `212-215`: 左向き Join
- `216-217`: 左向き Fork
- `218-221`: 下向き Join
- `222-223`: 下向き Fork

したがって、`Join` や `Fork` は「状態 3 や 5 を見たら自動的に動く」わけではない。
実際には、中心状態、入出力方向、入力 token 配置、誤り配置、および確率定数を含む複数 rule の組として振る舞いが定義される。

### 4.4 誤り則の位置づけ

`join_err_*` や `fork_err_*` を参照する規則は、成功則の周辺に配置された確率的な誤り則である。
これらは「入力条件の摂動を含む遷移も、ある確率で許す」という形で記述されている。

ここで重要なのは、誤り率の物理的解釈は rule ファイル名から自明には決まらない点である。
PyBCA が提供するのは確率付き局所遷移であり、それを温度や熱雑音へどう対応づけるかは、解析モデル側の理論設定に属する。

## 5. 任意の追加状態と規則を設計する手順

任意の追加状態をもつ素子を設計する場合、実装上の最小手順は次の通りである。

1. 使用する整数状態を決める
2. CellSpace 側にその状態を配置する
3. Rule YAML の `rules` に `prev` と `next` を定義する
4. 一時状態を安定状態へ戻したい場合は `conv` を定義し、state gate を有効化する
5. 確率 sweep を行いたい場合は `constants` と alias を使う

このとき、次の注意が必要である。

- 左右反転や上下反転は自動生成されないため、必要なら明示的に mirror した rule を追加する
- `0` は vacant として省略記法に使われやすいため、内部状態として多用しない方が可読性が高い
- `next=-1` は書き戻し除外として扱われるため、通常の状態値として使うべきではない

## 6. エディタの自由度と限界

### 6.1 BCL Editor

BCL Editor は、CellSpace と element 配置を作る GUI である。

できること:

- 点、矩形、線による CellSpace 作成
- `element` 定義と `place.Element(...)` の配置
- カスタム状態値の配置
- キャンバスの動的拡張

実装上の具体的制約:

- プリセットブラシは `0, 1, 2, -1, 3, 4` である
- それ以外の状態値も `Custom` により `-128..127` の範囲で入力できる
- BCL source pane は読み取り専用であり、完全なテキストエディタではない

従って、BCL Editor は CellSpace の構成には強いが、複雑な rule family 自体を記述する道具ではない。

### 6.2 Rule Editor

Rule Editor は local rule の視覚編集に特化している。

できること:

- `prev` / `next` パターンの編集
- rule の追加、削除、複製的編集
- pattern size の変更
- 任意整数状態を用いた規則記述
- `probability` の直接値編集
- 既存 `constants` alias の選択と保持

実装上の具体的制約:

- pattern size は `3x3`, `5x5`, `7x7`, `9x9`
- カスタムブラシは `-128..127`
- `constants` と `conv` は読み込み・保存時に保持される
- ただし `constants` や `conv` を GUI 上で体系的に新規編集する専用 UI はない

従って、Rule Editor は local pattern の設計には十分柔軟であるが、rule ファイル全体の高次構造を一から編集する作業では YAML 手編集を併用する方が明快である。

### 6.3 CellSpace Viewer

CellSpace Viewer は、CellSpace、Rule、Special Event を読み込んで step 実行を確認するための GUI である。

できること:

- CellSpace / Rule / Event の読み込み
- `global_prob`, `seed`, `device`, `state_gate` の設定
- 1 step 実行と連続実行

限界:

- 現行 Viewer は legacy backend に依存する
- GUI モードでは parallel trial は `1` 固定である
- `trial_constant_sweep` や `torchrun` 分散は GUI の主対象ではない

従って、Viewer は設計結果の可視確認には有効であるが、大規模 sweep や HPC 実験を GUI 上で完結させる用途には向かない。

## 7. どのツールを使うべきか

- CellSpace の構図と素子配置を設計したい: `BCL Editor`
- local rule family を設計・修正したい: `Rule Editor`
- 単一条件の挙動を対話的に確認したい: `CellSpace Viewer`
- 大規模 trial、sweep、分散実行を行いたい: `Engine` とスクリプト実行

研究用途では、GUI は設計・可視化のための前段と位置づけ、最終的な実験実行は `PyBCA.api.Engine` を用いたスクリプトへ移すのが最も見通しがよい。
