# PyBCA
Brownian Cellular Automaton for Python and torch library

## Installation
```bash
git clone https://github.com/Minamium/PyBCA.git
pip install -e ./PyBCA
```

# 開発方針メモ

2次元のブラウン回路の動作をシミュレートするセルオートマトン. 

### cudaBCA
既存のセル空間と遷移規則を保存したyamlよりセル空間と遷移規則を読み込み, cuda上でシミュレーションを行うクラス.

特殊イベントとして, 任意セルの状態により任意セルを任意の状態に変更するイベントを定義できる機能も持つ.

また重要な機能として, 任意に新しい遷移規則を追加してC-JoinやCrossのエラーレートを表現できるようしなければならない.

### CellSpaceViewer
セル空間を表示するGUIアプリケーション.
またセル空間の更新の様子を確認するため, 任意ステップ数の更新後のセル空間の表示や, 1ステップ毎の更新を見れるよう, 連続更新も可能.


## セル空間の更新操作の検討

### セル空間 1 ステップ更新の実装仕様（CUDA/並列安全）

目的：**GPU（CUDA）上でセル並列**のまま、**ランダム性の公平性**と**トークン（状態=2）の増殖禁止**を同時に満たす 1 ステップ更新を定義する。  
状態は **4値**：`0 = Vacant`, `1 = Wire(信号線)`, `2 = Token`, `-1 = ReCycleBinとの接続点`。

ReCycleBinとは、理想的にトークンが十分に存在する領域であり、その接続点はトークンが出てきたり、また入っていく(回路上からは消える)動作を示す.
---

## 更新アルゴリズム

### 事前準備

乱数生成器の準備

遷移規則配列のシャッフル

更新に使うテンソル, 変数の定義

TCHW: [Trial, 1, H, W] dtype=int8, Trial別セル空間配列(引数, 戻り値)

rule_arrays: [N,2,3,3] dtype=int8, N種類の遷移規則を記録した配列(引数)

rule_mask: [[0, 1, 0], [1, 1, 1], [0, 1, 0]] dtype=bool, 四近傍遷移規則マッチング用のマスク配列

rule_probs: [N] dtype=float32, N種類の遷移規則の確率配列(引数)

global_prob: float32, グローバル確率(引数)

Pickup_rule: [2, 3, 3] dtype=int8, ループ内でシャッフル遷移規則から取り出す遷移規則

TNHW_boolMask: [Trial, N, H, W] dtype=bool, 遷移規則をマッチして適用できたセルの中心座標を1とするboolマスク. ルール数次元に並列に遷移規則マッチングできる.

tmp_mask: [Trial, 1, H, W] dtype=int8, 取り出した遷移規則により書き換えられる差分セルだけを検査するk_writeカーネルを元に書き換え予定のセルに1を足していくためのテンソル

TCHW_applied: [Trial, 1, H, W] dtype=bool, 今までの遷移規則の適用により書き換えが行われたセルに1を立てておくboolマスク

### ルールマッチング

TCHW, ruleテンソルによりTNHW_boolMaskにマッチした3*3領域の中心座標を1へ(GPUでセル, trial並列)

### グローバル確率ゲート

TNHW_boolMaskの1のセルに対して独立に乱数スコアを与え、グローバル確率ゲートにより受容か棄却を判定、棄却ならTNHW_boolMaskの該当要素を0に(GPUでセル, trial並列)

### シャッフルされた遷移規則配列の要素順に以下のループを実行

### loop begin

### 遷移規則確率ゲート

遷移規則確率ゲートを遷移規則別確率よりかけて受容か棄却を判定、棄却ならTNHW_boolMaskの該当要素を0に(GPUでセル, trial並列)

### 規則内競合解決

TCHW_boolMaskより中心座標からk_writeカーネルにより加算したtmp_mask(int8)を作成して規則内で書き換え領域がかぶっている(値が1より大きい)書き換えセルをもつTCHW_boolMaskの1の要素については両方とも0にする(GPUでセル, trial並列)

### 他規則競合解決
TCHW_boolMaskより中心座標からk_writeカーネルにより加算したtmp_mask(int8)を上書き
これまでの遷移規則により書き換えられたセルについて1を立てるフラグ情報配列[Trial, 1, H, W]であるTCHW_applied(bool)とtmp_maskがかぶるTCHW_boolMaskの1の要素を0にする(GPUでセル, trial並列)

### 書き換え実行

TCHW_boolMaskとruleのnextを使ってTCHWを書き換える。同時にTCHW_appliedの書き換えた差分セルの要素に対応する座標を0から1へ(GPUでセル, trial並列)

### loop end

### TCHWが更新後のセル空間になる

---