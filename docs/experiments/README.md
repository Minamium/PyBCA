# BCA-IP 実験結果

従来セル空間を512独立試行、global_prob=0.5で実行した。600万ステップまでのFSM持続出力による最適解到達観測は457/512（89.26%）、終了時の確認は446/512（87.11%）。判定条件・感度・各試行の解は以下のレポートとJSONに保存している。

2026-10-07 15:11 JST時点、新しい条件1・N=1の本番は約154万/300万ステップまで進行中。全512試行を150万ステップで揃えた途中集計では、最適解 `111100` の安定出力は314/512（61.33%、最小持続期間1万）、10万期間では306/512（59.77%）。全履歴内の支持区間を含めると315試行で確認した。これは途中結果であり、残りを最終的な失敗とは扱わない。条件1・N=2と条件2・N=2の本番は順番待ち。全3条件のGPU事前検証は成功済み。

| 実験・解析 | レポート |
| --- | --- |
| 条件1・N=1の150万ステップ途中結果 | [集計・活動・検証](2026-10-07-variants/interim-1500000/report.json)・[全試行のFSM読取](2026-10-07-variants/interim-1500000/readout/trials.jsonl)・[短期判定](2026-10-07-variants/interim-1500000/stability/summary.json) |
| 3条件×512試行・300万ステップのrokko実験 | [実行条件](2026-10-07-variants/protocol.json)・[投入記録](2026-10-07-variants/submission.json) |
| 条件1のN=1/N=2、条件2のN=2セル空間 | [ファイル・対応イベント・reset検証](../../Sample/Cellspace/BCA-IP-variants/README.md) |
| 最適解の安定出力率の時系列とN=1/2/5の比較計画 | [時系列図とNの役割](2026-10-05-retention/report.md) |
| 600万ステップの到達率と300万からの変化 | [結果と全試行データ](2026-10-01-production-p05-6m/report.md) |
| 600万でも未到達の55試行と次の実験 | [停滞・リセット・段間流量の調査](2026-10-01-production-p05-6m/nonhit-diagnosis.md) |
| 300万ステップの完走・履歴監査 | [実行結果](2026-09-28-production-p05/report.md) |
| 300万時点の短い安定判定と未到達例 | [判定条件と出力変化](2026-09-28-production-p05/short-stability-and-nonhits.md) |
| 300万から600万への継続設定 | [ジョブ・チェックポイント](2026-09-28-continuation-6m.md) |
| 京大A100の速度・容量測定 | [ベンチマーク](2026-09-25-a100.md) |
| global_probの診断 | [1.0と0.5の比較](2026-09-25-probability-check/report.md) |
| 論文向け追加実験の整理 | [残作業](2026-09-30-experiment-status.md) |

## 元データ

Gitには解析コード、各試行の集計、検証記録、図を保存する。全イベント履歴と最終チェックポイントのアーカイブはGitHub Release `bca-ip-results-2026-10-01` の添付ファイルで配布する。

- [データRelease](https://github.com/Minamium/PyBCA/releases/tag/bca-ip-results-2026-10-01)
- `bca-ip-p05-512-3m.tar.gz`: 300万まで、81,845,628 bytes、SHA-256 `a258873e93452c926dfd3323faf2a7a59bc85e66aec2809a08021e5511aebd74`。
- `bca-ip-p05-512-6m.tar.gz`: 600万まで、217,297,964 bytes、SHA-256 `24fa669b840d5c3462b7596f42dce5cb0a42dd9f134e543a046b781d0230c8cb`。先頭300万は同じ試行の履歴。

rawファイルを展開する `results/` はGit管理外。アーカイブには再開用の全512最終セル状態と、イベント履歴・実行条件が含まれる。各レポートの `download.json` と `SHA256SUMS` も照合できる。

公開時にGitHub側のサイズ・SHA-256とローカルの原本を照合した。[Releaseの検証記録](2026-10-01-production-p05-6m/published-release.json)。リポジトリのルートから次のように取得・展開できる（GitHub CLIを使用）。

```sh
gh release download bca-ip-results-2026-10-01 --repo Minamium/PyBCA \
  --dir results/github-release-20261001
(cd results/github-release-20261001 && shasum -a 256 -c SHA256SUMS)
mkdir -p results/production-p05-20260928 results/production-p05-6m-20261001
tar -xzf results/github-release-20261001/bca-ip-p05-512-3m.tar.gz \
  -C results/production-p05-20260928
tar -xzf results/github-release-20261001/bca-ip-p05-512-6m.tar.gz \
  -C results/production-p05-6m-20261001
```
