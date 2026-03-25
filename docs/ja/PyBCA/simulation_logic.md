---
title: Simulation Logic
parent: Japanese
nav_order: 13
---

# Simulation Logic

本ページでは、`src/PyBCA/core/simulator.py` に実装された更新機構に即して、PyBCA の 1 step 計算を記述する。
主眼は公開 API ではなく、セル状態表現、規則照合、確率ゲート、競合解決、状態変換、および特殊イベントの適用順序にある。

## 1. 状態表現と入力資産

PyBCA の CellSpace は整数状態を持つ 2 次元格子として表現される。
基本状態は次の通りである。

- `0`: vacant
- `1`: wire
- `2`: token
- `-1`: recycle-bin connection point

入力資産は主として以下の 3 種から成る。

- CellSpace YAML
- Rule YAML
- Special Event `.py`

CellSpace YAML は最小外接矩形に切り出され、内部ではオフセット情報とともに保持される。
したがって、外部ファイル上の座標系と内部テンソル上の添字系は必ずしも一致しない。

## 2. 内部テンソル表現

`set_ParallelTrial(T)` を呼ぶと、単一 CellSpace が `T` 個の trial に複製される。
主要テンソルの概念的な形状は次の通りである。

- `TCHW`: `[Trial, 1, Height, Width]`
- `rule_arrays_tensor`: `[N, 2, 3, 3]`
- `rule_probs_tensor`: `[N]` または `[T, N]`
- `TNHW_boolMask`: `[Trial, N, Height, Width]`
- `TCHW_applied`: `[Trial, 1, Height, Width]`

ここで `N` は読み込まれた rule 数である。
`trial_constant_sweep` を用いる場合、`rule_probs_tensor` は `[T, N]` に拡張され、trial ごとに異なる rule 適用確率が与えられる。

## 3. 1 step の計算順序

1. `self.rng.manual_seed(self._current_step + 65536 + seed)` により乱数生成器を初期化する
2. `TNHW_boolMask`, `tmp_mask`, `TCHW_applied` を初期化する
3. `_match_centers_all_rules()` により全 rule の適用可能中心を求める
4. `_global_prob_gate()` により大域確率ゲートを適用する
5. `_rule_prob_gate()` により rule 個別確率ゲートを適用する
6. `torch.randperm(N)` により、その step における rule の処理順を 1 回だけ生成する
7. 各 rule について、規則内競合解決、規則間競合解決、書き戻しを順に行う
8. 必要なら `apply_state_gates()` を適用する
9. 必要なら `apply_spatial_events()` を適用する
10. `current_step` を 1 増やす

重要なのは、rule のシャッフル順が trial ごとに独立ではなく、その step では全 trial で共有されることである。

## 4. rule の照合

各 rule は `prev` と `next` から成る 3x3 パターンとして保持される。
照合は `_match_centers_all_rules()` において一括して行われる。

実装上の特徴は次の通りである。

- 境界外は `0` でパディングした上で照合する
- 十字型マスク `[[0,1,0],[1,1,1],[0,1,0]]` の位置は常に比較対象となる
- 四隅は、`prev` の対応セルが非ゼロの場合に限って比較対象となる
- 比較対象に含まれた位置では、値は厳密一致で判定される
- `-1` はワイルドカードではなく、比較対象に含まれた場合には通常の状態値として扱われる

従って、四隅に `0` を記述したからといって「四隅が 0 であること」を必ずしも要求するわけではない。
現実装では、四隅の `0` は実質的に未指定として機能する。

## 5. 大域確率ゲート

`global_prob` は、rule 個別確率とは別に適用される大域的受理率である。
`_global_prob_gate()` は次の入力形状を受理する。

- スカラー
- `[N]`
- `[T, N]`
- `[T, N, H, W]`
- `None`

`None` の場合はゲートを適用しない。
それ以外では `[T, N, H, W]` へ正規化した上で、候補ごとに独立な Bernoulli 判定を行う。

## 6. rule 個別確率と sweep

rule 確率は Rule YAML の `probability` から読み込まれる。
`trial_constant_sweep` を指定すると、YAML 内の `probability: *alias` をもつ rule 群に対し、trial ごとに

`base + trial_index * delta`

で定義される確率列が与えられる。
実装上は `[0, 1]` に clamp された後、alias に対応する全 rule index に同じ列が適用される。

この機構が表現するのは確率パラメータであり、温度そのものではない。
有効温度との対応づけは、解析モデル側で与える解釈である。

## 7. 競合解決

PyBCA では、1 step 内における更新競合を二段階で除去する。

### 7.1 規則内競合

`_rule_inner_conflict_resolution()` は、同一 rule の複数の中心候補が同一ターゲットへ書き込む場合、それらに対応する中心候補を除外する。

### 7.2 規則間競合

`_rule_outer_conflict_resolution()` は、既に `TCHW_applied` に記録された書き込み済みセルと衝突する候補を除外する。

この二段階を経た後に `_write_back()` が呼ばれるため、最終的な書き戻しではターゲット競合が解消済みであることが前提となる。

## 8. 書き戻し規則

書き戻し対象は `prev` と `next` の差分セルのみである。
具体的には、

- `post != pre`
- `post != -1`

を満たすセルのみが書き込み対象となる。
従って、`next` 側の `-1` は「状態 `-1` を書く」ことを意味せず、むしろ「書き戻し対象から除外する」ための番兵値として扱われる。

## 9. state gate

`apply_state_gates()` は、Rule YAML の `conv` セクションから読み込まれた状態変換を LUT に変換し、`TCHW` 全体へ同時適用する。

実装上の注意点:

- 変換は逐次ではなく同時置換である
- 同じ `prev` が複数回定義された場合、後勝ちとなる
- `state_gate_enable=True` であれば、`_current_step % state_gate_interval == 0` を満たす step で適用される
- `current_step` のインクリメント前に評価されるため、step `0` でも条件を満たせば適用される

## 10. spatial event

`apply_spatial_events()` は、通常の local rule 更新の後に適用される。
各 event 行は

- 旧形式: 6 列
- 拡張形式: 9 列

を取り得る。

拡張形式では以下が指定できる。

- event probability
- `start_step`
- `end_step`

実装上の要点は次の通りである。

- 座標は外部ファイル上では大域座標で与えられる
- 適用時に `offset_x`, `offset_y` を引いて内部添字系へ変換する
- 範囲外に出る event は破棄される
- `start_step`, `end_step` は両端を含む
- 確率ゲートは trial ごとに独立である
- event による書き込み後、その座標は `TCHW_applied` にも反映される

## 11. event history とメタデータ

special event 名が定義されている場合、`set_ParallelTrial()` 時に

- `List[Dict[str, List[int]]]`

の形式で `event_history` が初期化される。

`save_event_histry_for_dataframe()` は、履歴を DataFrame および各種ファイル形式へ変換し、あわせて以下のメタデータを保存できる。

- `parallel_trial`
- `current_step`
- `device`
- `offset_x`, `offset_y`
- `rule_ids`
- `rule_probs_base`
- `probability_sweep`

したがって、実験後の統計解析では、event の発火時刻だけでなく、各 trial に対応する sweep 条件も追跡できる。

## 12. BNN 実験の解釈

`tests/BNN.py` では、`join_err_*` および `fork_err_*` を trial ごとに sweep しつつ、

- `output`
- `add_weight`
- `subs_weight_pre`
- `subs_weight_post`

といった event を観測する。

PyBCA が直接提供するのは、確率 sweep と event 履歴である。
そこから有効温度に対する性能曲線を構成するかどうかは、別途与える理論的対応関係に依存する。

## 13. 性能と設定

実行性能に大きく影響する主要パラメータは次の通りである。

- `device`
- `trials`
- `steps`
- `use_tqdm`

大規模 grid や多数 trial の場合、`device="cuda"` の有無が支配的である。
一方、希少 event の観測では `steps` を十分長く取る必要がある。

## 14. 参照先

- 実行方法: [PyBCA Guide](PyBCA_guide.md)
- 引数仕様: [Engine API](engine_api.md)
- BCL 文法: [BCL Guide](../bcl/0.1/guide.md)
