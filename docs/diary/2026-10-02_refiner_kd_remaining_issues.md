# 2026-10-02: 軽量refinerの現状と次回の課題

記録時点: 2026-10-02 22:47 JST。ユーザーの最終指示に従い、追加の論文調査・改善実装・追加学習は行わず、確認済みの課題と未検証事項だけを整理する。次回の確認項目は未実施であり、改善効果を主張しない。

## 確認済みの現状

教師はDA無しbest80、studentはR2 `vssd_local_global_128x2_v2_train` のbest180。block整合100＋finetune200で終了。early stoppingは有効だったが、停止理由は200ステップ上限到達。同じstrict16F validation15のGT lossは教師0.19620265、student0.29247786。

全系列・全点の評価はADT/DriveTrack/PStudio各5件の既知validation。各clip平均、subset内平均、3subset等重みで集計。0–1表記。

| 指標 | 教師best80 | 軽量best180 |
| :--- | ---: | ---: |
| Median-scaled 3D-AJ | 0.14899326 | 0.14965128 |
| Absolute metric-AJ | 0.26813288 | 0.19643878 |
| Absolute metric-APD3D | 0.39121931 | 0.30581274 |
| PStudio absolute metric-AJ | 0.36037736 | 0.16687923 |
| Occlusion accuracy | 0.81118275 | 0.81118275 |

絶対metric-AJは相対約26.74%低下し、同等精度の圧縮は未達。OA同値はflow可視性を共有するためであり、軽量visibility headの改善を示さない。PStudio `boxes_21` はmetric-AJが0.57279210→0.21467139、GT可視点3D誤差中央値が0.04410728m→0.67404354mで、低下が明瞭。

証跡: [15動画評価](../evidence/refiner_kd_20261002/accuracy_R2_best180_validation15.json)、[学習終了記録](../evidence/refiner_kd_20261002/training_R2_best180_status.json)、[公開評価レポート](https://github.com/yuki-inaho/vmamba3-3Dpointtracker/releases/download/mamba3-preview-20261001-step200/refiner_kd_v2_best180_20261002_metrics.json)。

## 優先度P0: 絶対精度の低下原因が未確定

1. **スケール誤差の直接診断がない。** median-scaled AJとabsolute metric-AJはスケール補正だけでなく閾値定義も異なる。「低下はすべてスケール予測の失敗」とは断定できない。clip単位のdepth比/log-depth偏り、フレーム間スケール変動、局所depth誤差、UV誤差が未分解。次回は同じ可視maskで分解し、近距離/遠距離・subset・系列長別に確認する。GTによるスケール合わせは診断用oracleとして分離し、正式スコアに混ぜない。

2. **損失・checkpoint選択とmetric-AJの対応が未検証。** 現行GT項はGT由来clip scaleで割った3D L1＋UV補正正則化。XYZ蒸留もscale正規化する。これはGTへの中央値整合ではなく、絶対座標の教師信号は残るが、距離スケールで動画間の勾配重みが変わる。対してabsolute metric-AJは整合なし・固定1/4/16/64/256cm閾値。best選択とearly stoppingは短window正規化GT lossに基づき、全系列metric-AJ最良を保証しない。次回は保存checkpoint間で両者の相関を測り、独立validationで選択/停止基準を決める。

3. **入力の絶対スケール情報と補正能力が未監査。** mixerへの埋め込みは `z_raw/z_ref` と中心深度で正規化したpatchを使用。メートル単位のdepthはreadoutで掛け戻し、Kはunprojectionに使う。これは確認済みの構造だが、低下原因と実験で確定したものではない。同じ画像に対するraw depth全体の共通倍率誤りをどう識別できるか、補正量の分布・一様倍率への応答・GT scaleの推論への漏洩がないことを確認する。

参照: [GT/KD損失](../../src/mamba3_tracker/train/distillation.py)、[選択的蒸留](../../src/mamba3_tracker/train/staged_distillation.py)、[入力/readout](../../src/mamba3_tracker/model/onnx_refiner.py)、[絶対評価](../../src/mamba3_tracker/eval/tapvid3d_eval.py)。

## 優先度P1: 学習比較・長系列・評価集合が未完

4. **R2改善の優位性は未証明。** 今回は1seed、window16、finetune200のpilot。改善後の同じデータ/予算によるR0/R1/R2対照比較、最終800ステップの複数seed、選択後のwindow32適応は未完。旧short実験とはquery/input/学習契約が異なるのでlossだけを直接比較しない。変更の効果を分離する比較が必要。

5. **短windowから全系列への一般化が未検証。** 学習16Fに対し評価系列は長い。系列長別・前半/後半の絶対誤差やscale driftを未分解。window32キャッシュ完成とwindow32適応訓練完了は別。再開時はmanifestのstatus/identityを確認し、時間軸分割で現行推論契約を変えない。

6. **15動画は未見/公式評価ではない。** trainとmonitor/minivalに保守的sequence単位の重複がある。現在の値は既知validationの診断用で、独立testや公式TAPVid-3D minival150全件の結果ではない。PointOdysseyの2D δ_avg/MTE/Survivalも未評価。teacherの学習履歴と台帳を確認し、独立validation/testを確保する。部分集合を全件評価と表示せず、少数動画の差だけで統計的な優位性を断定しない。

## 優先度P2: 配布受入・モデル予算・再開契約

7. **ONNX数値一致とタスク精度維持は別。** copy-fused deployは615,838パラメータ。CPU dynamic9条件、ORT CUDA12条件（実monitor3件を含む）の数値検証は済み。上の15件スコアはunfused PyTorchで、配布ONNXの公式全150件タスクスコアではない。最終checkpointのunfused/fused/ORT CUDA精度とGPU専有時の速度/VRAMを揃える必要がある。refinerの圧縮率をDINO/flow/depth込みの全体圧縮率と表示しない。

8. **scale変更でchunk契約・予算を壊さない。** 時間軸は保持し、点だけchunk。全B/F/Nのlower median＋1e-6で `z_ref` を一度計算して全chunkに共有する。点順入れ替えとchunk不変性を維持する。全点/フレーム共通scaleを将来導入するならONNX入力とcache identityも再監査。deploy目安600,000–625,000、上限650,000は未変更。推論に必要なheadを「学習専用」として数え落とさない。

9. **再開と公開の証跡を保全する。** `last.pt` はsource/config/input hash一致時だけresume可能。Releaseの `_train.pt` はbestモデルでoptimizer入りresume checkpointではない。契約変更は新run/新cache namespaceで記録する。軽量Releaseはexperimentalで教師best80推奨を置き換えていない。4列動画はDesktopの `tracking_comparison_kd_20261002/comparison_all.mp4`、3シーン481F。定性的動画は正式指標の代用ではない。

## 次回の着手順（未実施）

1. 教師/best180のhash、同一15件評価、データ重複、残りcache/取得ジョブのstatusを確認する。
2. PStudioを入口にscale偏り・時間変動・局所depth・UV誤差を分解する。
3. 保存checkpointの短window lossとabsolute metric-AJの関係を調べ、独立validationで選択/停止基準を決める。
4. ユーザー側の調査結果に基づき、条件を固定した対照実験と必要な長系列適応を設計する。
5. 最終モデルのPyTorch/fused/ORT CUDA精度・サイズ・速度を確認し、未達ならexperimentalを維持する。

本メモは課題整理のみ。新しい論文調査の結論・改善効果・追加訓練開始を意味しない。
