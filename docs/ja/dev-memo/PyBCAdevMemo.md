---
title: PyBCA Dev Memo
parent: Japanese
nav_order: 16
---

# PyBCA Dev Memo

このメモは、PyBCA のシミュレーション理論と更新アルゴリズムの意図を整理した開発者向け資料です。

## 1. 対象モデル

PyBCA は、2 次元 Brownian circuit の動作をセルオートマトンとして近似する実装です。

主要状態:

- `0`: vacant
- `1`: wire
- `2`: token
- `-1`: recycle-bin 接続点

recycle-bin 接続点は、理想的に十分なトークン源または吸収先として扱います。

## 2. 実装上の中心クラス

- `BCA_Simulator`
  YAML から CellSpace と rule を読み込み、`torch` 上で更新します。
- CellSpace Viewer
  セル空間の可視化と手動検査に使います。

## 3. 1 step 更新の狙い

GPU 上で trial 並列、セル並列を維持しつつ、次を満たすことが目的です。

- ランダム性の公平性
- 競合更新の抑制
- token の不正増殖防止

## 4. 主なテンソル

- `TCHW`
  `[Trial, 1, H, W]` のセル空間
- `rule_arrays`
  `[N, 2, 3, 3]` の遷移規則
- `rule_mask`
  四近傍マッチ用マスク
- `rule_probs`
  規則別確率
- `TNHW_boolMask`
  各 rule が適用可能な中心座標マスク
- `tmp_mask`
  書き換え競合検査用
- `TCHW_applied`
  既に書き換えられたセルのフラグ

## 5. 更新アルゴリズム

1. 乱数生成器を現在 step と seed から初期化
2. 全 rule に対してマッチ候補を `TNHW_boolMask` に展開
3. `global_prob` により候補を間引く
4. rule のシャッフル順に各 rule を処理
5. rule 内競合を除去
6. 既適用セルとの競合を除去
7. `next` パターンを `TCHW` に反映
8. 必要なら state gate を適用
9. 必要なら spatial event を適用

## 6. 競合解決の意図

同じ step 内で複数 rule が同じセルを書き換えると、セルオートマトンの解釈が不定になります。
そこで PyBCA では、各 rule の書き換え領域を `tmp_mask` に投影し、重なりを持つ候補を両方落とします。

この方針により、更新順依存の偏りを抑えつつ GPU 並列化を維持します。

## 7. Special Event と state gate

- state gate
  一定 interval ごとに大域状態変換をかける仕組みです。
- spatial event
  任意セルの状態条件に応じて任意セルを書き換える仕組みです。

通常更新とは別レイヤですが、実験系のモデル化には重要です。

## 8. 現状の移植状態

- シミュレーション本体は `src/PyBCA/core/simulator.py` に移植済み
- GUI Viewer は legacy 実装を継続利用
- utility 群も legacy ラッパが残る

実行理論自体は parity suite で legacy と一致確認済みです。
