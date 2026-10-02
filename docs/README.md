# 実装と実験の記録

次回の作業は [2026年10月2日の継続メモ](session_memory_20261002.md) から確認してください。
100 GBデータ取得、DA無しキャッシュと訓練、best80の公開、比較動画、GPU版ONNX Runtimeの実検証までの判断と保存先をまとめています。
モデルと動画の本体はGitに含めません。

最新の軽量モデルの継続作業は [次回に残す課題](diary/2026-10-02_refiner_kd_remaining_issues.md) と [KD公開時点の引き継ぎ](session_memory_kd_20261002.md) を確認してください。

- [約62万パラメータへのrefiner蒸留計画](knowledge_distillation_plan_20261002.md): 初期設計と短期KD訓練の記録。短window訓練・全F monitor・ONNX CPU/CUDAは実施済みですが、精度維持の最終受入は未達です。
- [KD精度改善の詳細研究と診断](kd_accuracy_research_20261002.md): 回帰KD、Vision/軽量Mamba、圧縮・再パラメータ化の一次研究、query整合性・未使用重み・長系列の問題、段階的蒸留と対照実験。
- [必要な追加学習とearly stopping](kd_additional_training_20261002.md): 本日のR2追加学習、停止条件、最良モデル評価、進捗確認と再開手順。最終受入とは区別します。
- [軽量モデルを加えた4列比較動画](tracking_comparison_student_20261002.md): with-studentモード、同一前処理、PStudio/DriveTrack/ADTの3シーン481F出力・監査、再実行手順。
- [軽量モデルのスコアと論文指標の対応](kd_metrics_release_20261002.md): 同一15動画のAJ/APD/OA、絶対精度の差、PointOdyssey指標との区別、配布モデルの範囲。
- [ONNX化の方式とCPUおよびGPU検証](onnx_export_design_20261002.md): TritonのMamba-3を標準演算へ展開した方法、保持した計算、制約、再実行コマンド。
- [best80のONNX Runtime GPU実動画検証](onnx_gpu_video_20261002.md): 同じ3本の全481フレーム・全点でCUDA実行、CPU数値差とAJ、再実行手順。
- [best200のONNXとC++ CPU実装記録](onnx_best200_cpu_20261001.md): 従来のCPU配布経路と固定9動画の受入条件。新best80の学習成績とは区別します。
- [best200の元作業書](workdoc_Oct01-2026_onnx_best200.md): 当初の要求と未達項目を含む履歴。

## 小型の証跡

| 内容 | 証跡 |
| :--- | :--- |
| データ台帳とキャッシュの集計 | [data_cache_summary.json](evidence/daoff_best80_20261002/data_cache_summary.json) |
| DA無し学習の設定とloss履歴 | [training_summary_20261002.json](evidence/daoff_best80_20261002/training_summary_20261002.json)、[run_config_20261002.json](evidence/daoff_best80_20261002/run_config_20261002.json) |
| best80の安全な公開とONNX変換 | [public_export_report_20261002.json](evidence/daoff_best80_20261002/public_export_report_20261002.json)、[export_report.json](evidence/daoff_best80_20261002/export_report.json) |
| Release再ダウンロード検証 | [public_download_audit.json](evidence/daoff_best80_20261002/public_download_audit.json) |
| 同一3動画の旧モデルと新モデルとONNX CPU比較 | [summary.json](evidence/tracking_comparison_20261002/summary.json) |
| ONNX Runtime GPU実行 | [best200_gpu_report.json](evidence/onnx_gpu_20261002/best200_gpu_report.json)、[best80_gpu_report.json](evidence/onnx_gpu_20261002/best80_gpu_report.json) |
| best80のGPU実動画3本とAJ | [gpu_report.json](evidence/onnx_gpu_video_20261002/gpu_report.json)、[summary.json](evidence/onnx_gpu_video_20261002/summary.json)、[fixture_manifest.json](evidence/onnx_gpu_video_20261002/fixture_manifest.json)、[quality.json](evidence/onnx_gpu_video_20261002/quality.json) |
| 統合時の単体テストと品質確認 | [quality.json](evidence/integration_20261002/quality.json) |

各JSONの `success` / `pass` はその証跡の検証範囲だけを示します。
人工入力の一致、選択済み3動画の比較、固定9動画の受入条件、公式minival全150本の評価は別です。
