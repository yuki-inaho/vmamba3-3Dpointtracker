# 2026-10-02: 必要な追加学習とearly stopping

本日の残り約30分という希望を受け、まず改善モデルR2の追加学習のみを開始する。最終学習完了や正式なR0/R1/R2比較完了を意味しない。

追記（2026-10-02 22:47 JST）: 追加学習はblock100＋finetune200で完了。early stoppingは有効だが上限到達で終了。最良step180のGT lossは0.2924778565、教師の同じmonitor lossは0.1962026500。[最新スコア](kd_metrics_release_20261002.md)と[次回の課題](diary/2026-10-02_refiner_kd_remaining_issues.md)を参照。以下の開始時点の計画と最終受入を区別する。

## 実行内容

- モデル: `vssd_local_global_128x2_v2_train`。教師固定、DAなし、strict GT reanchor、FP32 frontend。
- 入力: window16、train1027件、validation15件。effective batch32、microbatch上限32。
- 教師へのblock整合100ステップ、その後finetune最大200ステップ、seed42。
- Early stopping有効: finetuneの検証を10ステップごとに行い、`min_delta=0.001`以上の改善が5回連続でなければ停止する。block整合には適用しない。
- `last.pt`は再開用。検証で観測した最小lossの`student_step*.pt`を別途保存し、終了後、その最良checkpointを全フレームmonitor15件で自動評価する。
- 全フレームmonitor評価は既知の検証動画を用いるため、公式blind minivalの精度ではない。教師とstudentの実行精度も異なるのでreportの契約に従う。

`--early-stop`を追加し、従来fullモードだけだったpatience判定を今回のpilotにも明示的に適用した。有効化状態はcheckpoint identityに保存され、異なる設定での無言再開を拒否する。従来のfullモードは引き続き自動で有効。

## 保存先・確認

リポジトリのルートで実行する。

```bash
tail -f result/refiner_kd_improved_20261002/additional_R2_background.log
jq '{status, phase, block_step, ft_step, bad, best_monitor_gt_loss, best_checkpoint, exit_reason}' result/refiner_kd_improved_20261002/pilot_R2_fullcache_s42/training_status.json
```

学習成果: `result/refiner_kd_improved_20261002/pilot_R2_fullcache_s42/`。
全フレーム評価: `result/refiner_kd_improved_20261002/monitor_full_pilot_R2_s42/accuracy_report.json`。

中断後、ソース・設定・入力を変更せず、次のコマンドで再開する。二重起動はflockで拒否する。正常終了済みなら再学習せず評価に進む。既存評価reportは上書きしない。

```bash
bash scripts/run_kd_additional_r2.sh
```

## 今回開始しないもの

R0/R1の200ステップ比較、最終800ステップの複数seed、本選択後のwindow32適応、学習済みモデルの最終ONNX受入・GPU専有benchmark。window32キャッシュ生成と公式minivalダウンロード／前処理は既存ジョブを継続する。これらの未完了項目を作業書の完了として扱わない。

追記: best180のCPU/CUDA数値検証とexperimental Release追加は後続作業で完了。ただし最終選択後の全件タスク精度・専有速度gateとは別。再開にはローカルcacheと未変更sourceが必要で、Release weightsだけでは訓練再開できない。
