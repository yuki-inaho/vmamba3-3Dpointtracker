# KD作業書の公開時点snapshot

2026-10-02のローカルtemp作業書とレビューを保全したもの。チェック済みはその時点の証跡の範囲だけを意味する。KD全体の最終精度受入は未達であり、動画/Release公開完了とは別。

- [初期KD作業書](workdoc_Oct02-2026_refiner_kd_onnx.md) / [レビュー](review_Oct02-2026_refiner_kd_onnx.md)
- [精度改善作業書](workdoc_Oct02-2026_kd_accuracy_improvement.md) / [レビュー](review_Oct02-2026_kd_accuracy_improvement.md)
- [3シーン動画・Release・指標作業書](workdoc_Oct02-2026_student_comparison_demo.md) / [レビュー](review_Oct02-2026_student_comparison_demo.md)

最新の課題は [diary](../diary/2026-10-02_refiner_kd_remaining_issues.md)、現在の実行方法は [引き継ぎ](../session_memory_kd_20261002.md)。履歴中のtemp補助スクリプト8本はscriptsへ移植済み。元tempファイルは無視対象として残し、動作中ジョブのsourceは変更していない。旧pilot/geometry/smokeスクリプトは当時の固定条件を再現する補助で、新しいモデル選択や自動再学習を指示するものではない。
