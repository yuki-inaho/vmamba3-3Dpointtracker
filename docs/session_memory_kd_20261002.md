# 2026-10-02: KD公開・終了時の引き継ぎ

専用branch: `distill/refiner-kd-onnx-20261002`。本記録はcommit前のsnapshotで、push後のSHAはGit履歴で確認する。mainへの統合・force pushは今回行わない。

R2追加学習はblock100＋finetune200で完了。best180をexperimental軽量モデルとして既存Releaseへ追加し、9assetsを再downloadしてSHA一致確認済み。教師best80推奨と既存19assetsは保全。全体KDの最終精度受入は未達。詳細は [スコア](kd_metrics_release_20261002.md)、[次回の課題](diary/2026-10-02_refiner_kd_remaining_issues.md)、[作業書snapshot](workdocs/README.md)。

## 保存場所と再現

- 学習: `result/refiner_kd_improved_20261002/pilot_R2_fullcache_s42/`。resumeはoptimizer入り `last.pt`、公開bestは `student_step180.pt`。
- 公開モデル: [Release](https://github.com/yuki-inaho/vmamba3-3Dpointtracker/releases/tag/mamba3-preview-20261001-step200) の `refiner_kd_v2_best180_20261002` prefix。PT/ONNX/元best/source ZIP/card/metrics/CPU/CUDA reports/SHA一覧の9assets。
- 4列動画: Desktopの `tracking_comparison_kd_20261002/comparison_all.mp4`、PStudio150F/DriveTrack31F/ADT300F。動画本体はGitに追加しない。
- 学習終了・15件評価・export・CUDA・動画の小型JSON/card: `docs/evidence/refiner_kd_20261002/`。
- 既存cached inputsがある場合の再開/評価: `bash scripts/run_kd_additional_r2.sh`。終了済み学習や既存評価は上書きしない。source/config/input hashが違うとresumeを拒否する。
- 初期A0–A4 pilot: `scripts/run_kd_pilots.py`。旧実験の固定条件の再現用であり、現在の追加学習用ではない。
- 補助監査: `scripts/audit_kd_inventory.py`、`scripts/check_kd_native_geometry.py`、`scripts/verify_staged_smoke.py`。それぞれデータ重複、teacher geometry、固定2clip GPU smoke/resumeを確認する。
- 公開パッケージの再現: `scripts/package_kd_release.py`。固定best180の証跡とホワイトリストを使い、既存公開packageの上書きを拒否する。自動uploadは行わない。

## 電源断前後の残り処理

22:47 JST時点で学習プロセスはなく、minival取得、minival DA3 depth生成、strict window32 train cache生成が動作中。これらは最終学習の完了を意味しない。生データ・cache・進行stateはGit対象外のローカル領域に保全。

取得state: `/workspace/vmamba3_eval/download_status.json`。depth state: `/workspace/vmamba3_eval/depth/preparation_status.json`。window32 manifest: `result/refiner_kd_improved_20261002/inputs_w32_train/input_manifest.json` と `inputs_w32_validation/input_manifest.json`。

成果ファイルはpartialからrenameする方式。完了したファイルは残るが、電源断中の処理は完了を保証せず再実行が必要。再開前にstatusとmanifestを確認し、同じジョブを二重起動しない。OSの通常のシャットダウンを使用し、未完了部分を全件完了扱いしない。

repo rootで、前回と同じ契約を維持して再開するコマンド:

```bash
uv run --no-sync python scripts/download_kd_minival.py --split minival --data-root /workspace/vmamba3_eval/raw --train-manifest /workspace/vmamba3_eval/source_manifest.json --state-json /workspace/vmamba3_eval/download_status.json --raw-budget-gb 30 --total-data-budget-gb 60 --reserve-gb 30
```

depth生成はピン留めDA3 revision/SHAを確認する。以下は別terminalで実行し、必要な利用権とローカルweightsを前提とする。

```bash
source scripts/cudnn_env.sh
uv run --no-sync python scripts/prepare_kd_minival_depth.py --data-root /workspace/vmamba3_eval/raw --out-root /workspace/vmamba3_eval/depth --minival --watch --skip-existing --auto-batch --chunk-frames 4 --max-batch-frames 8 --gpu-reserve-gb 12 --process-res 504 --total-data-budget-gb 60 --stats-json /workspace/vmamba3_eval/depth_batch_stats.jsonl
```

window32の再開は、現在のsource/configを保ったままtrain→validationを順に実行する。起動中ジョブとの並行再実行はしない。

```bash
source scripts/cudnn_env.sh
export HF_HUB_OFFLINE=1
for partition in train validation; do
  uv run --no-sync python scripts/prepare_refiner_kd.py --config configs/distill_refiner_local_global_w32.yaml --mode cache --partition "$partition" --window 32 --reanchor-window --compute-frontend --frontend-precision fp32 --new-input-root result/refiner_kd_improved_20261002/inputs --new-input-budget-gb 150 --out-dir "result/refiner_kd_improved_20261002/inputs_w32_${partition}" || break
done
```

private認証ファイル・tokens・利用制限付きbackbone weights・生データ・巨大cache・動画/PT/ONNX本体をGitに含めない。公開コード/設定/既存tests/docs/必要再現補助と小型証跡をcommit対象とする。追加の論文調査・改善学習はユーザーの撤回指示に従い開始しない。
