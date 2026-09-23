# 再現可能な長時間実験と CUDA 計算コア

新しい実験用コアは `execution_mode="cuda", rng_mode="independent"` で選択します。
既存コードのデフォルトは `reference / legacy` のままです。GUI の旧実装も変更していません。
V100 で使用できます。CUDA がない環境では `torch_sparse` を使います。

## 構成

```text
src/PyBCA/
  api/config.py         実験条件、試行 ID、保存間隔
  api/engine.py         ステップ実行、定期保存、再開
  api/distributed.py    torchrun による独立試行の分割
  api/streaming.py      有限長の履歴バッファ、原子的な保存、チェックポイント
  core/simulator.py    既存 API と reference 実装
  core/optimized.py    int8 照合、候補ベースの競合判定、Torch 実装
  core/cuda_backend.py CUDA バッファ管理、コンパイルキャッシュ、起動
  core/kernels.cu      V100 対応の整数演算カーネル
  core/random.py       試行 ID に基づく Philox4x32-10
  _legacy/             変更しない比較対象と旧 GUI 用実装
scripts/
  run_bca_ip.py        BCA-IP 実験・再開 CLI
  benchmark_core.py    同期を含めた速度・GPU メモリ測定
  compare_runs.py      最終状態と全イベント・規則履歴の照合
  rokko/              PBS の環境確認、検証、測定、8 GPU パイロット
tests/
  test_core_runtime.py 境界、競合、乱数、保存・再開、破損検知
  validate_core.py     Sample 20 条件と BCA-IP の各ステップ比較
```

## 計算の意味を維持する条件

1. ステップ開始時のセル空間ですべての規則を照合します。先行規則の書込み後に照合し直しません。
2. 十字の 5 セルは 0 を含めて厳密一致です。隅は pre が 0 のとき不問、非 0 のとき厳密一致です。pre の -1 は通常の値です。
3. 境界外は 0 と照合し、境界外への書込みは捨てます。
4. 書込み対象は `pre != post` かつ `post != -1` の位置です。
5. 同じ規則の候補が書込み先を共有した場合、関係する候補をすべて棄却します。勝者を選びません。
6. 先行規則の書込み先と一つでも衝突する候補は棄却します。
7. 状態変換、特殊イベントの順で適用します。イベント条件はすべてのイベント書込みより前に評価します。ステップ範囲は両端を含みます。履歴は 0 始まりです。

CUDA 版は `unfold` と `[T,N,9,H,W]` の比較テンソルを作りません。
`[T,N,H,W]` の bool マスク、int8 状態、規則ごとの固定長候補バッファを使います。
候補バッファの既定値は各試行・各規則 4096 個です。超過時は当該規則・試行を全セルで処理するため、候補は失われません。
CPU/GPU 間への候補数転送や、規則ごとの `nonzero()` 同期を CUDA 版では行いません。

規則、イベント、状態変換の配置は初期化時にコンパイルします。これらを直接書き換える場合は `set_ParallelTrial()` でプランを作り直してください。
実行中に変えてよいパラメータは scalar の `global_prob` と同じ形状の規則確率です。

## 乱数モード

| モード | 用途と再現性 |
| --- | --- |
| `legacy` | 現行のステップ単位再シード、共有規則順序、Torch の乱数消費を維持。等しい入力・Torch・GPU 条件で reference と逐次一致を検証するためのモード。 |
| `independent` | 64 bit の実験 seed と global trial ID からキーを作り、セル・規則・ステップ・用途を Philox のカウンタに割り当てる。試行ごとに Fisher–Yates の規則順序を生成。候補順、GPU rank、試行の分割に依存しない。 |

`independent` の乱数バージョンは `philox4x32-10-v1`。seed と試行 ID は符号なし 64 bit 整数です。
確率ゲートは 24 bit の `[0,1)` 浮動小数点値を用います。非常に小さな確率ではこの離散化を考慮してください。
同じ seed と ID を再利用すれば同じ試行です。別の試行には別の ID を割り当てます。
CPU と CUDA で整数アルゴリズムとゲートの比較を揃えています。
Philox のゼロ入力は [Random123 の公式既知解テスト](https://github.com/DEShawResearch/random123/blob/main/tests/kat_vectors) の先頭ワードと照合しています。

独立モードの確率行列は常に `[trial, rule]` です。旧実装では trial 数と rule 数が等しい場合に行列が転置されるため、legacy モードは比較用としてその挙動も維持します。
`trial_constant_sweep` は global な試行順の添字から計算し、分割時の浮動小数点丸めの違いを避けます。任意の ID を使う場合も、sweep の添字は `trial_offset` で別に指定します。
同じ書込み先の特殊イベントが複数発火した場合、独立モードは定義順の後勝ちです。BCA-IP の既存イベントにはこの重複がありません。

## 保存と再開

```python
from PyBCA import Config, Engine

config = Config(
    cellspace_path="Sample/Cellspace/BCA-IP.yaml",
    rule_paths=["Sample/rule/base-rule.yaml"],
    spatial_event_file_path="Sample/Specialevent/BCA-IP_event.py",
    device="cuda", execution_mode="cuda", rng_mode="independent",
    seed=20260923, trials=8, trial_ids=list(range(8)), steps=2_000_000,
    stream_dir="results/my-run", flush_interval=1000, checkpoint_interval=10000,
    quiet=True, use_tqdm="false",
)
result = Engine(config).run()
```

```text
results/my-run/
  manifest.json                         参照すべき履歴チャンク、条件、入力ハッシュ
  history/000000000000-000000001000.jsonl
  history/000000001000-000000002000.jsonl
  checkpoint.pt                         最新の再開地点（原子的に置換）
```

履歴は `{kind, trial, step, name, count}` の JSONL です。規則の発火回数を同じステップの重複リストに展開しません。
`manifest.json` が列挙するファイルだけを読みます。履歴の保存間隔を超えて GPU/CPU メモリに保持しません。規則履歴は既定で無効です。
stream 保存と旧形式の最終一括出力は同時に指定できません。stream モードの `simulator.event_history` / `rule_history` は `None` です。

チェックポイントには int8 の全セル、形状と原点、次のステップ、規則確率、乱数状態、試行 ID、入力・実装ハッシュ、履歴の確定範囲が含まれます。
一時ファイルを fsync してから rename します。チェックポイントより先の履歴があっても、再開時にはその参照を戻して再計算します。
実行開始時にもステップ 0 のチェックポイントを作るため、最初の定期保存より前の中断からも再開できます。
参照されなくなった古いチャンクは安全のため削除しません。`history/*.jsonl` を直接全結合しないでください。
履歴の破損、異なる入力・seed・試行 ID・シミュレーション条件・実装による再開は拒否します。
同一ディレクトリへの複数 writer はファイルロックで防ぎます。
保存チャンクのメタデータには GPU の現在・ピーク割当量とプロセスの最大 RSS も記録します。
チェックポイントは `core/` と `api/` の実装ハッシュを照合するため、実装を更新する前に実験用スナップショットを保持してください。

```bash
PYTHONPATH=src python scripts/run_bca_ip.py \
  --output-dir results/my-run --trials 8 --steps 2000000 --seed 20260923
# 中断後は同じ引数に --resume を加える。steps は追加数ではなく最終到達ステップ。
PYTHONPATH=src python scripts/run_bca_ip.py \
  --output-dir results/my-run --trials 8 --steps 2000000 --seed 20260923 --resume
```

```python
from PyBCA.api.streaming import iter_history
for record in iter_history("results/my-run", verify=True):
    if record["kind"] == "event":
        print(record)
```

torchrun では `rank_0000/` 等に分かれます。`iter_history` は親ディレクトリも受け付けます。
新規実験は任意の GPU 分割で同じ試行を生成できます。既存チェックポイントからの再開は元の world size・試行配置で行います。チェックポイントの再分割機能は含みません。
独立モードでは、同じ試行配置で CPU の `torch_sparse` と GPU の `cuda` を切り替えて再開できます。legacy モードは同じデバイス種別・Torch バージョンが必要です。
SIGTERM/SIGINT は現在のステップを完了して保存します。強制 kill の場合は直前の定期チェックポイントから再開します。

## rokko

```bash
cd /home/IM25D029/PyBCA_workspcace/streaming-gpu-core
qsub scripts/rokko/validate.pbs
qsub scripts/rokko/benchmark.pbs
qsub scripts/rokko/pilot.pbs
# 本番の投入例。試行 ID 0..63、200 万ステップ。既存ディレクトリは上書きしない。
qsub -v PYBCA_RUN_DIR=results/bca-reference-seed20260923 scripts/rokko/run.pbs
# 同じ GPU 数で再開する場合:
qsub -v PYBCA_RUN_DIR=results/bca-reference-seed20260923,PYBCA_RESUME=1 scripts/rokko/run.pbs
```

既存の `G` キュー、`ngpus` 指定、venv を使います。
CUDA カーネルのコンパイルに `module load cuda/11.8.0 gcc/11.4.0` が必要です。
PyTorch C++ の ABI に依存せず、現在の Torch CUDA stream へ起動します。Torch や CUDA Toolkit の置換は必要ありません。
コンパイル済みライブラリはソース・CUDA コンパイラ・GPU 世代をキーにしたキャッシュへ置きます。

8 GPU では `python -m torch.distributed.run --standalone --nproc-per-node=8 scripts/run_bca_ip.py ...` を使います。
rokko の既存 venv は移設前のパスを指す `torchrun` 実行ファイルを含むため、Python モジュール形式で起動します。
空き GPU が少ない場合は `qsub -l select=1:ngpus=4:ncpus=16:mem=64gb scripts/rokko/pilot.pbs` のように資源数を変更できます。プロセス数は割当て GPU 数から自動取得します。
1 台の GPU を1つの試行群に割り当てます。セル空間を GPU 間に分割する実装ではありません。
PBS の割当てた `CUDA_VISIBLE_DEVICES` を引き継ぎます。

## 論文の実験入力について

付属の BCA-IP セル空間は全体回路 1 ファイルです。既存サンプルを `legacy-reference` として識別し、論文の Instance 1 / 2 と同一であるとは認定しません。
CLI の summary に `instance_certified=false` を記録します。実験条件に対応するセル空間と解のデコーダーを確定するまで、出力イベントの初出を最適解到達時刻とみなさないでください。
回路、規則、イベントのファイルハッシュを記録するため、確定後の入力差し替えは新しい実験ディレクトリで区別できます。
