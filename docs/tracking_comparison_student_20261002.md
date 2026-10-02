# 軽量モデルを追加する比較動画モード

`scripts/demo_tracking_comparison.py --mode with-student` で、旧best200、新best80 native、新best80 ONNX CPU、軽量refiner CUDA FP32の4列を出力する。従来defaultの3列モードは変更しない。

## 短い実動画での確認済み成果

以下は初回DriveTrack動作確認の履歴。ユーザー指摘後、PStudio150F/DriveTrack31F/ADT300Fの**全3シーン版（計481F）を生成済み**。現在の `comparison_all.mp4` は3本の連結で、2560×1080/H264/yuv420p、4動画ファイルすべての全デコード監査が成功。PStudio/ADTのsnapshotも4列表示を確認した。初回31Fだけのファイルは同じ出力先で更新した。

- 保存先: `/home/kasm-user/Desktop/tracking_comparison_kd_20261002/`
- 初回動画: `comparison_drivetrack.mp4`（31フレーム、256点、2560×1080、H264/yuv420p）。初回確認時の `comparison_all.mp4` はこの1本だけだったが、上記3シーン版へ更新済み。
- モデル: `pilot_R2_fullcache_s42/student_step180.pt`、617,374 train-form parameters。
- モデルSHA256: `16ec743e3a46c00a7bffd77db83862a8037a09fbc8ecc52e586d3c1d60761ada`。
- metric-AJ: 旧0.111063、新native0.154284、新ONNX0.154359、軽量0.136199。この選択済み1動画の数値であり、独立test/公式minivalの精度ではない。
- DINO抽出1回、4方法で前処理・全フレーム・全点・flow可視性を共有。studentの参照depthは全点で計算し、点方向chunk32でも固定。
- 2動画ファイル（単体と連結）の全デコード監査成功。snapshotの4列表示、共有ID/色/3D cubeを目視確認。
- 元の3列動画4ファイルは既存監査のSHAと一致し、上書きされていない。

これはstudentのPyTorch推論比較で、student ONNX精度一致や最終蒸留受入の証明ではない。別途作成したfull-frame入力cacheは旧デモとDINO hashが異なったため、このデモでは再利用せずlive共有抽出を使用した。

## 再実行

repo rootでbashから実行する。元の準備済みrelease comparison bundleが必要。

```bash
source scripts/cudnn_env.sh
export HF_HUB_OFFLINE=1
uv run --no-sync python scripts/demo_tracking_comparison.py \
  --mode with-student \
  --student-checkpoint result/refiner_kd_improved_20261002/pilot_R2_fullcache_s42/student_step180.pt \
  --stage infer --subset drivetrack
uv run --no-sync python scripts/demo_tracking_comparison.py --mode with-student --stage render --subset drivetrack
uv run --no-sync python scripts/demo_tracking_comparison.py --mode with-student --stage audit --subset drivetrack
```

3本すべての手順はrepo README 4.6を参照。別checkpointは新しい `--out-dir` へ出力する。最初に `--student-sha256` を指定して期待SHAを照合することもできる。`last.pt`ではなく公開 `student_step*.pt` を指定する。

ユーザー指示に従い、新規テストは作らず、既存11テストとruff/format/ty、実動画infer/render/auditで確認した。
