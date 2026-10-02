# 作業計画書 兼 記録書

**日付：** 2026年10月2日
**作業ディレクトリ・リポジトリ:** /home/kasm-user/Desktop/vmamba3-3Dpointtracker
**作業者：** root

## 1. 作業目的

### 1.1 ゴール要求分析

既存デモへ学習済み軽量refinerを加え、旧best200・新best80・ONNX・軽量モデルを同一動画で確認できるモードを提供する。ユーザーの追記を優先し、新しいテストコードは作らない。既存テスト、品質ゲート、実動画の短いinfer/render/auditで検証する。3列モードと既存Desktop成果物を保全する。Release更新、commit/push、再学習は非ゴール。

### 1.2 サブゴール構造

SG1: 任意の公開student checkpointをSHA固定し、同一DINO/WAFT/DA3/query/flow visibilityで4方法を推論する。
SG2: 4列レイアウト、全点のAJ、独立Desktop出力、再現コマンドを提供する。
SG3: 新規テストなしで既存互換性と実動画のエンコードを確認する。

### 1.3 トレーサビリティ方針

TR1=SG1/手順1–3/manifestとprediction report。TR2=SG2/手順4–5/media/layout/README。TR3=SG3/手順6/既存pytest・ruff・ty・ffprobe。

## 2. 作業内容

### フェーズ1 調査・設計

対象は scripts/demo_tracking_comparison.py（entry）、scripts/compare_release_tracking.py（infer/render/audit）、src/mamba3_tracker/viz/comparison.py（描画）。uv.lock/pyprojectあり、justfile/local AGENTS/CODEXなし。既存full-frame評価入力と旧デモのDINO特徴hashは一致しなかったので、キャッシュを同じ入力と偽って再利用しない。ThreeWayRefinerのDINO single-extractionにstudentを追加する。

### フェーズ2 実装

`--mode with-student`とcheckpoint指定を追加。新出力のdefaultはDesktop/tracking_comparison_kd_20261002。元manifestを検証し、student checkpoint+SHA+modeを追加した独立manifestを作る。3列defaultは不変。renderの幅は640×列数、高さ1080。axesは列数で比例配置。studentはCUDA FP32、既存nativeはBF16 mixer、既存ONNXはCPU FP32。studentのz_refは全点で計算して固定し、track chunkのみ分割する。GTは推論へ渡さない。

### フェーズ3 検証・記録

DriveTrack31フレームを実推論し、`--subset drivetrack`の短いrender/auditで4列を確認する。全3本の出力はコマンドを文書化し、必要なら後続実行する。短い確認を全3本の完了と偽らない。

## 3. 作業チェックリスト

### 手順 1: 設計・作業書レビュー
- [x] 🖐 **操作**: 本書をreview rubricと照合する。TR1–3。
- [x] 🔎 **確認**: 列数、入力共有、SHA、出力保全、検証範囲が具体的。
- [x] 🧪 **テスト**: 新規テスト作成はユーザー指示で省略。既存テスト2ファイルを使用。
- [x] 🛠 **エラー時対処**: cache不一致は実測記録しlive共有経路を採用、暗黙reuse禁止。

### 手順 2: CLIと独立manifest
- [x] 🖐 **操作**: compare_release_tracking.pyへmodeとstudent checkpoint検証を追加する。TR1。
- [x] 🔎 **確認**: default release互換、with-studentは別出力、異なるSHA上書き拒否。
- [x] 🧪 **テスト**: 既存CLI helpとinfer subset必須を維持。
- [x] 🛠 **エラー時対処**: checkpointなし/不一致/同じ出力なら明示エラー。

### 手順 3: 同一frontendのstudent推論
- [x] 🖐 **操作**: ThreeWayRefinerへoptional studentを追加する。TR1。
- [x] 🔎 **確認**: DINO1回、z_ref全点固定、同一visibility、score/provenanceにstudent追加。
- [x] 🧪 **テスト**: 実DriveTrack31Fで4方法のscoreとshapeを確認。
- [x] 🛠 **エラー時対処**: GPU不足なら既存GPUジョブを無断停止せず原因を記録する。

### 手順 4: 可変列数の描画・監査
- [x] 🖐 **操作**: rendering/summaryへ3列と4列の選択を追加する。TR2。
- [x] 🔎 **確認**: 4列2560×1080、同じ点色/表示ID/共有cube、student CUDA FP32表示。
- [x] 🧪 **テスト**: DriveTrack31F render/audit全decodeで確認。
- [x] 🛠 **エラー時対処**: 不足予測なら3列へfallbackせずinfer要求エラー。

### 手順 5: 実行手順を記録
- [x] 🖐 **操作**: READMEへ4列モードとcheckpoint/source指定を記載する。TR2。
- [x] 🔎 **確認**: renderのみはGPU不要、実推論の制約とvalidation scopeを明示。
- [x] 🧪 **テスト**: 記載commandを短動画で実行する。
- [x] 🛠 **エラー時対処**: eval checkpointとtrainer last.ptを混同しない。

### 手順 6: 既存品質ゲート
- [x] 🖐 **操作**: 既存pytestとruff/format/tyを実行する。TR3。
- [x] 🔎 **確認**: 新規テストファイルなし、既存3列CLI互換。
- [x] 🧪 **テスト**: uv run --no-sync pytest tests/unit/test_tracking_demo_cli.py tests/unit/test_tracking_comparison.py -q。
- [x] 🛠 **エラー時対処**: 既存dirty変更を保全し、対象ファイルだけ修正する。

## 4. 作業に使用するコマンド参考情報

### 追加要求: 3シーン完成・Release掲載・論文指標の説明

2026-10-02 22:16:39 JST、ユーザーの指摘を受け、31F確認だけを完成扱いにした判断を訂正する。旧動画と同じPStudio150F/DriveTrack31F/ADT300F、計481Fを4列で完成させる。追加学習は再実行しない。既存Releaseへ軽量モデルを別名assetで追加し、旧best80の推奨・既存assetsを保全する。教師並みの精度維持やPointOdyssey評価済みという主張は禁止。

### 手順 7: 不足2シーンの実推論
- [x] 🖐 **操作**: with-studentでpstudio/adtを同じstep180から推論する。TR1。
- [x] 🔎 **確認**: 3subset合計481F/全点、3reportsの同じstudentSHA。
- [x] 🧪 **テスト**: 新規テストなし、実infer成功と4method score。
- [x] 🛠 **エラー時対処**: OOMや欠損は記録し既存cache/旧成果を削除しない。

### 手順 8: 全3シーンを連結・監査
- [x] 🖐 **操作**: subset指定なしのwith-student render/auditを実行する。TR2。
- [x] 🔎 **確認**: comparison_all.mp4が481F、3clipsと4mediaの全decode成功。
- [x] 🧪 **テスト**: ffprobe/framecount/2560x1080とsnapshot確認。
- [x] 🛠 **エラー時対処**: 部分renderを全完了としない、古い31F summaryを最終成果と扱わない。

### 手順 9: 実験用軽量モデルをReleaseへ追加
- [x] 🖐 **操作**: step180をcopy-fuse/exportしCPU検証後、別名PT/ONNX/card/metrics/hash assetsを追加する。
- [x] 🔎 **確認**: deploy615838parameters、sourceSHA固定、既存assets保全、再downloadSHA一致。
- [x] 🧪 **テスト**: 実export parity/checker/dynamiccases、publish前modelcardが既知15clip評価/精度差を明示。
- [x] 🛠 **エラー時対処**: 未検証項目は未検証と明記、同名assetsをclobberしない、認証値非表示。

### 手順 10: 論文指標と現時点評価を説明
- [x] 🖐 **操作**: PointOdyssey原論文と元tracker論文の評価契約を調査し同じ15clip teacher/studentを集計する。
- [x] 🔎 **確認**: TAPVid-3D AJ/APD/OA/絶対metricとPointOdyssey2D delta/MTE/Survivalを区別する。
- [x] 🧪 **テスト**: 一次出典を確認、未実施PointOdyssey/test/minivalを実施済みとしない。
- [x] 🛠 **エラー時対処**: 同名APD/閾値/スケーリングが異なる場合は直接比較不能と明記する。

repo rootから `uv run --no-sync python scripts/demo_tracking_comparison.py --help`。
GPU推論はbashで `source scripts/cudnn_env.sh; export HF_HUB_OFFLINE=1`。
checkpointは result/refiner_kd_improved_20261002/pilot_R2_fullcache_s42/student_step180.pt を用いる（現時点の検証最良）。既存outputsは /home/kasm-user/Desktop/tracking_comparison_20261002。

## 6. 完了の定義

- [x] D4/TR1–2: 軽量モデル4列で元と同じ3シーン/481Fの連結と全decodeが完成。
- [x] D5/TR1–3: Releaseへ別名の実験モデルとprovenance/性能限界を掲載、download検証済み。
- [x] D6/TR3: 一次論文と現行評価を区別した説明とdocsが完成。

- [x] D1/TR1: with-studentモードで実DriveTrack31Fの4方法を推論しstudent SHAを固定。
- [x] D2/TR2: 4列の動画と全decode監査、既存3列出力の保全を確認。
- [x] D3/TR3: 既存テスト/品質チェックと再現コマンドを記録、新規テストなし。

## 7. 作業記録

**重要な注意事項：**

* 作業開始前に必ず `date "+%Y-%m-%d %H:%M:%S %Z%z"` コマンドで現在時刻を確認し、正確な日時を記録します。
* 各作業項目を開始する際と完了する際の両方で記録を行うこと。
* 作業内容は具体的なコマンドや操作手順を詳細に記載すること。
* 結果・備考欄には成功／失敗、エラー内容、解決方法、重要な気づきを必ず記入すること。
* 複数のフェーズがある場合は、フェーズごとに開始・完了の記録を取ること。
* コード変更を行った場合は、変更したファイル名と変更内容の概要を記録すること。
* エラーが発生した場合は、エラーメッセージと解決策を詳細に記録すること。

| 日付 | 時刻 | 作業者 | 作業内容 | 結果・備考 |
| :--- | :--- | :--- | :--- | :--- |
| 2026-10-02 | 22:03:43 JST+0900 | root | フェーズ1調査 | 既存3列とfullframe cacheを照合。ray/depth/K一致、DINO特徴不一致。live shared frontend採用。新規テストなし。 |
| 2026-10-02 | 22:03:43 JST+0900 | root | 手順 1 🖐 **操作**: 本書をre | Review PASS per rubric; source manifest caches compared: DINO differs, live same-extraction chosen. User forbids new tests, existing regression+real31F render/audit planned. Default3column preserved, independent Desktop output, no commit/push. |
| 2026-10-02 | 22:03:43 JST+0900 | root | 手順 1 🔎 **確認**: 列数、入力 | Review PASS per rubric; source manifest caches compared: DINO differs, live same-extraction chosen. User forbids new tests, existing regression+real31F render/audit planned. Default3column preserved, independent Desktop output, no commit/push. |
| 2026-10-02 | 22:03:43 JST+0900 | root | 手順 1 🧪 **テスト**: 新規テス | Review PASS per rubric; source manifest caches compared: DINO differs, live same-extraction chosen. User forbids new tests, existing regression+real31F render/audit planned. Default3column preserved, independent Desktop output, no commit/push. |
| 2026-10-02 | 22:03:43 JST+0900 | root | 手順 1 🛠 **エラー時対処**: c | Review PASS per rubric; source manifest caches compared: DINO differs, live same-extraction chosen. User forbids new tests, existing regression+real31F render/audit planned. Default3column preserved, independent Desktop output, no commit/push. |
| 2026-10-02 | 22:05:15 JST+0900 | root | 手順 2 🖐 **操作**: compa | CLI mode release/with-student, independent Desktop default, strict public student load/SHA pinned and mismatch overwrite rejection implemented. Existing infer subset guard retained, render/audit optional subset selection. No new test files. |
| 2026-10-02 | 22:05:15 JST+0900 | root | 手順 2 🔎 **確認**: defau | CLI mode release/with-student, independent Desktop default, strict public student load/SHA pinned and mismatch overwrite rejection implemented. Existing infer subset guard retained, render/audit optional subset selection. No new test files. |
| 2026-10-02 | 22:05:15 JST+0900 | root | 手順 2 🧪 **テスト**: 既存CL | CLI mode release/with-student, independent Desktop default, strict public student load/SHA pinned and mismatch overwrite rejection implemented. Existing infer subset guard retained, render/audit optional subset selection. No new test files. |
| 2026-10-02 | 22:05:15 JST+0900 | root | 手順 2 🛠 **エラー時対処**: c | CLI mode release/with-student, independent Desktop default, strict public student load/SHA pinned and mismatch overwrite rejection implemented. Existing infer subset guard retained, render/audit optional subset selection. No new test files. |
| 2026-10-02 | 22:08:29 JST+0900 | root | 手順 3 🖐 **操作**: Three | RealDriveTrack31F/256tracks infer complete20.7s, old/new/ONNX/student metricAJ=.111063/.154284/.154359/.136199. DINO extraction1 shared, studentSHA pinned/publicstep180. Same inputs and flowmask, full-N reference_depth fixed before chunk32. Output Desktop/tracking_comparison_kd_20261002 independent. ExistingCLI+viz11testsPASS. |
| 2026-10-02 | 22:08:29 JST+0900 | root | 手順 3 🔎 **確認**: DINO1 | RealDriveTrack31F/256tracks infer complete20.7s, old/new/ONNX/student metricAJ=.111063/.154284/.154359/.136199. DINO extraction1 shared, studentSHA pinned/publicstep180. Same inputs and flowmask, full-N reference_depth fixed before chunk32. Output Desktop/tracking_comparison_kd_20261002 independent. ExistingCLI+viz11testsPASS. |
| 2026-10-02 | 22:08:29 JST+0900 | root | 手順 3 🧪 **テスト**: 実Dri | RealDriveTrack31F/256tracks infer complete20.7s, old/new/ONNX/student metricAJ=.111063/.154284/.154359/.136199. DINO extraction1 shared, studentSHA pinned/publicstep180. Same inputs and flowmask, full-N reference_depth fixed before chunk32. Output Desktop/tracking_comparison_kd_20261002 independent. ExistingCLI+viz11testsPASS. |
| 2026-10-02 | 22:08:29 JST+0900 | root | 手順 3 🛠 **エラー時対処**: G | RealDriveTrack31F/256tracks infer complete20.7s, old/new/ONNX/student metricAJ=.111063/.154284/.154359/.136199. DINO extraction1 shared, studentSHA pinned/publicstep180. Same inputs and flowmask, full-N reference_depth fixed before chunk32. Output Desktop/tracking_comparison_kd_20261002 independent. ExistingCLI+viz11testsPASS. |
| 2026-10-02 | 22:09:11 JST+0900 | root | 手順 4 🖐 **操作**: rende | Real31F4column render/audit PASS, H264/yuv420p2560x1080, comparison_drivetrack.mp4 + comparison_all.mp4 full decodePASS, layout records4methods, shared IDs/cube. Original release4video SHA matches stored audit. New outputs Desktop/tracking_comparison_kd_20261002 only. Visual snapshot inspection follows. |
| 2026-10-02 | 22:09:11 JST+0900 | root | 手順 4 🔎 **確認**: 4列256 | Real31F4column render/audit PASS, H264/yuv420p2560x1080, comparison_drivetrack.mp4 + comparison_all.mp4 full decodePASS, layout records4methods, shared IDs/cube. Original release4video SHA matches stored audit. New outputs Desktop/tracking_comparison_kd_20261002 only. Visual snapshot inspection follows. |
| 2026-10-02 | 22:09:11 JST+0900 | root | 手順 4 🧪 **テスト**: Driv | Real31F4column render/audit PASS, H264/yuv420p2560x1080, comparison_drivetrack.mp4 + comparison_all.mp4 full decodePASS, layout records4methods, shared IDs/cube. Original release4video SHA matches stored audit. New outputs Desktop/tracking_comparison_kd_20261002 only. Visual snapshot inspection follows. |
| 2026-10-02 | 22:09:11 JST+0900 | root | 手順 4 🛠 **エラー時対処**: 不 | Real31F4column render/audit PASS, H264/yuv420p2560x1080, comparison_drivetrack.mp4 + comparison_all.mp4 full decodePASS, layout records4methods, shared IDs/cube. Original release4video SHA matches stored audit. New outputs Desktop/tracking_comparison_kd_20261002 only. Visual snapshot inspection follows. |
| 2026-10-02 | 22:09:11 JST+0900 | root | 手順 5 🖐 **操作**: READM | README4.6 now documents4column model/SHA pinning, GPU FP32 student/not studentONNX parity, separateoutdir, publiccheckpoint/notlast.pt, full3subset loop and short31Fcommands. Shortinfer/render/audit command successfully executed; missingremaining2subset not claimed complete. |
| 2026-10-02 | 22:09:11 JST+0900 | root | 手順 5 🔎 **確認**: rende | README4.6 now documents4column model/SHA pinning, GPU FP32 student/not studentONNX parity, separateoutdir, publiccheckpoint/notlast.pt, full3subset loop and short31Fcommands. Shortinfer/render/audit command successfully executed; missingremaining2subset not claimed complete. |
| 2026-10-02 | 22:09:11 JST+0900 | root | 手順 5 🧪 **テスト**: 記載co | README4.6 now documents4column model/SHA pinning, GPU FP32 student/not studentONNX parity, separateoutdir, publiccheckpoint/notlast.pt, full3subset loop and short31Fcommands. Shortinfer/render/audit command successfully executed; missingremaining2subset not claimed complete. |
| 2026-10-02 | 22:09:11 JST+0900 | root | 手順 5 🛠 **エラー時対処**: e | README4.6 now documents4column model/SHA pinning, GPU FP32 student/not studentONNX parity, separateoutdir, publiccheckpoint/notlast.pt, full3subset loop and short31Fcommands. Shortinfer/render/audit command successfully executed; missingremaining2subset not claimed complete. |
| 2026-10-02 | 22:10:34 JST+0900 | root | 手順 6 🖐 **操作**: 既存pyt | Existing11testsPASS4.29s; ruffcheck/format/tyPASS3targetfiles. New test files were not created or edited for this demo. Real31F4column full-decode/visual inspection and original4video SHA retentionPASS; docs/readme+docs index updated. Step2 earlier record22:05:15 should read actual command time22:05:55. No stage/commit/push/release changes. |
| 2026-10-02 | 22:10:34 JST+0900 | root | 手順 6 🔎 **確認**: 新規テスト | Existing11testsPASS4.29s; ruffcheck/format/tyPASS3targetfiles. New test files were not created or edited for this demo. Real31F4column full-decode/visual inspection and original4video SHA retentionPASS; docs/readme+docs index updated. Step2 earlier record22:05:15 should read actual command time22:05:55. No stage/commit/push/release changes. |
| 2026-10-02 | 22:10:34 JST+0900 | root | 手順 6 🧪 **テスト**: uv r | Existing11testsPASS4.29s; ruffcheck/format/tyPASS3targetfiles. New test files were not created or edited for this demo. Real31F4column full-decode/visual inspection and original4video SHA retentionPASS; docs/readme+docs index updated. Step2 earlier record22:05:15 should read actual command time22:05:55. No stage/commit/push/release changes. |
| 2026-10-02 | 22:10:34 JST+0900 | root | 手順 6 🛠 **エラー時対処**: 既 | Existing11testsPASS4.29s; ruffcheck/format/tyPASS3targetfiles. New test files were not created or edited for this demo. Real31F4column full-decode/visual inspection and original4video SHA retentionPASS; docs/readme+docs index updated. Step2 earlier record22:05:15 should read actual command time22:05:55. No stage/commit/push/release changes. |
| 2026-10-02 | 22:10:34 JST+0900 | root | 完了定義確認 | D1 accepted: realDriveTrack31F/256query inference4methods, sourceSHA16ec743e... andDINO extraction1 in Desktop reports/drivetrack.json. |
| 2026-10-02 | 22:10:34 JST+0900 | root | 完了定義確認 | D2 accepted: H264/yuv420p2560x1080, 2media fullDecodePASS, snapshot visually4columns, originalrelease4videoSHA unchanged. |
| 2026-10-02 | 22:10:34 JST+0900 | root | 完了定義確認 | D3 accepted: existing11testsPASS, ruff/format/tyPASS, no newdemo tests. README4.6+docs/tracking_comparison_student_20261002.md reproduce steps. Full3subset videos explicitly not generated. |
| 2026-10-02 | 22:22:21 JST+0900 | root | 手順 7 🖐 **操作**: with- | All3subset inference complete: PStudio150F460tracks, DriveTrack31F256tracks, ADT300F900tracks, total481F. Same step180SHA pinned, DINO1/clip and all4method scores recorded. PStudio63.3s/ADT168.6s. Full3 render/audit session8387 active. Scope completion not claimed before final mediaaudit. |
| 2026-10-02 | 22:22:21 JST+0900 | root | 手順 7 🔎 **確認**: 3subs | All3subset inference complete: PStudio150F460tracks, DriveTrack31F256tracks, ADT300F900tracks, total481F. Same step180SHA pinned, DINO1/clip and all4method scores recorded. PStudio63.3s/ADT168.6s. Full3 render/audit session8387 active. Scope completion not claimed before final mediaaudit. |
| 2026-10-02 | 22:22:21 JST+0900 | root | 手順 7 🧪 **テスト**: 新規テス | All3subset inference complete: PStudio150F460tracks, DriveTrack31F256tracks, ADT300F900tracks, total481F. Same step180SHA pinned, DINO1/clip and all4method scores recorded. PStudio63.3s/ADT168.6s. Full3 render/audit session8387 active. Scope completion not claimed before final mediaaudit. |
| 2026-10-02 | 22:22:21 JST+0900 | root | 手順 7 🛠 **エラー時対処**: O | All3subset inference complete: PStudio150F460tracks, DriveTrack31F256tracks, ADT300F900tracks, total481F. Same step180SHA pinned, DINO1/clip and all4method scores recorded. PStudio63.3s/ADT168.6s. Full3 render/audit session8387 active. Scope completion not claimed before final mediaaudit. |
| 2026-10-02 | 22:24:28 JST+0900 | root | 手順 9 進捗 | CPU trainedstep180 export PASS9cases +copy-fusion4; deploy615838params, ONNX2553222B/PT2492693B. ORTgpu12cases incl3realPASS, major computeCUDA/integerCPUonly.9unique prefixedassets uploaded, old19 preserved andbody pending additive edit. First redownload attempted in /tmp without --repo failed 'not a git repository'; retried explicit --repo. No token values printed. Hash/download acceptance still pending. |
| 2026-10-02 | 22:27:04 JST+0900 | root | 手順 8 🖐 **操作**: subse | Full3scene4column render/audit COMPLETE. PStudio150/DriveTrack31/ADT300=481F, fourH264/yuv420p2560x1080 files fullDecodePASS incl comparison_all481F. SameDesktoppath replaced short31F with complete3scenes. PStudio/ADT midpoint snapshot visual inspection. No oldrelease video overwritten. |
| 2026-10-02 | 22:27:04 JST+0900 | root | 手順 8 🔎 **確認**: compa | Full3scene4column render/audit COMPLETE. PStudio150/DriveTrack31/ADT300=481F, fourH264/yuv420p2560x1080 files fullDecodePASS incl comparison_all481F. SameDesktoppath replaced short31F with complete3scenes. PStudio/ADT midpoint snapshot visual inspection. No oldrelease video overwritten. |
| 2026-10-02 | 22:27:04 JST+0900 | root | 手順 8 🧪 **テスト**: ffpr | Full3scene4column render/audit COMPLETE. PStudio150/DriveTrack31/ADT300=481F, fourH264/yuv420p2560x1080 files fullDecodePASS incl comparison_all481F. SameDesktoppath replaced short31F with complete3scenes. PStudio/ADT midpoint snapshot visual inspection. No oldrelease video overwritten. |
| 2026-10-02 | 22:27:04 JST+0900 | root | 手順 8 🛠 **エラー時対処**: 部 | Full3scene4column render/audit COMPLETE. PStudio150/DriveTrack31/ADT300=481F, fourH264/yuv420p2560x1080 files fullDecodePASS incl comparison_all481F. SameDesktoppath replaced short31F with complete3scenes. PStudio/ADT midpoint snapshot visual inspection. No oldrelease video overwritten. |
| 2026-10-02 | 22:27:04 JST+0900 | root | 手順 9 🖐 **操作**: step1 | ExistingGitHubRelease now28assets:9new experimentalKDassets/19old preserved withsameSHA. Downloadall9into/tmp/kd-release-download-Kxn6Ka andallSHA match. Notes verified additive; teacherbest80 stillrecommended. Fuseddeploy615838params, ONNX2.55MB/PT2.49MB, source/trainpt/card/metrics/cpu+gpu reports/checksums. GPU12case incl3real PASS. Publicaudit saved docs/evidence/refiner_kd_20261002/public_release_kd_best180_audit.json. |
| 2026-10-02 | 22:27:04 JST+0900 | root | 手順 9 🔎 **確認**: deplo | ExistingGitHubRelease now28assets:9new experimentalKDassets/19old preserved withsameSHA. Downloadall9into/tmp/kd-release-download-Kxn6Ka andallSHA match. Notes verified additive; teacherbest80 stillrecommended. Fuseddeploy615838params, ONNX2.55MB/PT2.49MB, source/trainpt/card/metrics/cpu+gpu reports/checksums. GPU12case incl3real PASS. Publicaudit saved docs/evidence/refiner_kd_20261002/public_release_kd_best180_audit.json. |
| 2026-10-02 | 22:27:04 JST+0900 | root | 手順 9 🧪 **テスト**: 実exp | ExistingGitHubRelease now28assets:9new experimentalKDassets/19old preserved withsameSHA. Downloadall9into/tmp/kd-release-download-Kxn6Ka andallSHA match. Notes verified additive; teacherbest80 stillrecommended. Fuseddeploy615838params, ONNX2.55MB/PT2.49MB, source/trainpt/card/metrics/cpu+gpu reports/checksums. GPU12case incl3real PASS. Publicaudit saved docs/evidence/refiner_kd_20261002/public_release_kd_best180_audit.json. |
| 2026-10-02 | 22:27:04 JST+0900 | root | 手順 9 🛠 **エラー時対処**: 未 | ExistingGitHubRelease now28assets:9new experimentalKDassets/19old preserved withsameSHA. Downloadall9into/tmp/kd-release-download-Kxn6Ka andallSHA match. Notes verified additive; teacherbest80 stillrecommended. Fuseddeploy615838params, ONNX2.55MB/PT2.49MB, source/trainpt/card/metrics/cpu+gpu reports/checksums. GPU12case incl3real PASS. Publicaudit saved docs/evidence/refiner_kd_20261002/public_release_kd_best180_audit.json. |
| 2026-10-02 | 22:27:04 JST+0900 | root | 手順 10 🖐 **操作**: Point | PrimaryPointOdyssey2307.15055sec5.2: pixel delta_avg/MTE/Survival, not evaluated here. Tracker2609.34035sec4.1–4.2: officialminival150/median3DAJ vs absolutefixedcmMetricAJ; currentknown15 not comparable. Full15teacher/student means AJ .148993/.149651, APD .219474/.225796, absoluteAJ .268133/.196439, OA same. docs/kd_metrics_release_20261002.md/source links andpublicmetrics explain26.7% absoluteAJgap; noqualityequivalence claim. |
| 2026-10-02 | 22:27:04 JST+0900 | root | 手順 10 🔎 **確認**: TAPVi | PrimaryPointOdyssey2307.15055sec5.2: pixel delta_avg/MTE/Survival, not evaluated here. Tracker2609.34035sec4.1–4.2: officialminival150/median3DAJ vs absolutefixedcmMetricAJ; currentknown15 not comparable. Full15teacher/student means AJ .148993/.149651, APD .219474/.225796, absoluteAJ .268133/.196439, OA same. docs/kd_metrics_release_20261002.md/source links andpublicmetrics explain26.7% absoluteAJgap; noqualityequivalence claim. |
| 2026-10-02 | 22:27:04 JST+0900 | root | 手順 10 🧪 **テスト**: 一次出典 | PrimaryPointOdyssey2307.15055sec5.2: pixel delta_avg/MTE/Survival, not evaluated here. Tracker2609.34035sec4.1–4.2: officialminival150/median3DAJ vs absolutefixedcmMetricAJ; currentknown15 not comparable. Full15teacher/student means AJ .148993/.149651, APD .219474/.225796, absoluteAJ .268133/.196439, OA same. docs/kd_metrics_release_20261002.md/source links andpublicmetrics explain26.7% absoluteAJgap; noqualityequivalence claim. |
| 2026-10-02 | 22:27:04 JST+0900 | root | 手順 10 🛠 **エラー時対処**: 同 | PrimaryPointOdyssey2307.15055sec5.2: pixel delta_avg/MTE/Survival, not evaluated here. Tracker2609.34035sec4.1–4.2: officialminival150/median3DAJ vs absolutefixedcmMetricAJ; currentknown15 not comparable. Full15teacher/student means AJ .148993/.149651, APD .219474/.225796, absoluteAJ .268133/.196439, OA same. docs/kd_metrics_release_20261002.md/source links andpublicmetrics explain26.7% absoluteAJgap; noqualityequivalence claim. |
| 2026-10-02 | 22:27:04 JST+0900 | root | 完了定義D4 | D4 complete: original3scenes/481F/full4media decode/2560x1080, sameDesktopcomparison_all now complete. 31F-only initial completion corrected. |
| 2026-10-02 | 22:27:04 JST+0900 | root | 完了定義D5 | D5 complete:9newlightweightReleaseassets and additive notes verified,9redownloadSHA agree,19historicalassets preserved,teacherbest80 recommendation retained. |
| 2026-10-02 | 22:27:04 JST+0900 | root | 完了定義D6 | D6 complete: primaryPointOdyssey2D delta/MTE/Survival not evaluated; currentTAPVid3D15knownvalidation score summary andscope published/docs recorded. |
