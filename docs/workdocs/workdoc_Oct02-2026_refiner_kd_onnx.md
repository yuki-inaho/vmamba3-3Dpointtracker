# refiner知識蒸留の作業計画書 兼 記録書

**日付：** 2026年10月2日。作成時刻は17:14:21 JST+0900（08:14:21 UTC）。
**作業ディレクトリ・リポジトリ：** `/home/kasm-user/Desktop/vmamba3-3Dpointtracker`、yuki-inaho/vmamba3-3Dpointtracker。
**作業者：** Codex。次回実行者はこの作業書を単一の計画とし、変更判断を記録する。
**ブランチ：** `distill/refiner-kd-onnx-20261002`。起点main `6223405f44c72487dfd3d6669e490282a29b0821`。
**状態：** 2026-10-02 17:34 JST、ユーザーが訓練からONNX生成・検証までDoD達成を指示。実行フェーズを開始。各項目の証跡で進捗を更新する。

## 1. 作業目的

### 1.1 ゴール要求分析

ユーザーの直観的・直截的な目的は、学習済みの大きいrefinerの知識を、フォーク元と同程度に軽量なrefinerへ移し、推論時にはONNXで動かせる状態にすることです。モデルのパラメータ数を減らすだけでなく、tracking精度を維持できるかを実測します。

- 明示要求：関連論文のweb調査、auxiliary head、専用損失とモデル圧縮形式の設計、必要ならreparametrization、ONNX推論、専用ブランチ、temp以下のwrite/reviewスキルによる詳細作業書。
- 今回の成功条件：D-1〜D-8を実装・訓練・ONNX生成・CPU/CUDA・精度の実証で満たす。P-1/P-2の計画作成は完了済み。
- 後続の成功条件案：推論refiner600k〜625kを目安、650,000以下。aux除去/fusion済みFP32 ONNXが既存8入力4出力でCPU/CUDA動作し、教師との追跡精度差を独立性監査済みの評価集合で判定できる。
- 暗黙制約：uv使用、既存teacher・CPU/GPU環境・Releaseを保全。DRY/KISS/SOLID、t-wada式RED→GREEN、小型証跡、秘密非出力、暗黙fallback禁止。32GB GPUを使用して訓練と検証を行う。
- 非ゴール：DINO/WAFT/DA3の圧縮、RGB end-to-end ONNX、causal online化、Release asset差替え、main統合、commit/push。量子化と構造pruningは必須ではない。
- リスク：13倍圧縮で精度維持は未保証。teacherの未監督vis head、短windowと全Fの差、z_refの全B/F/N依存、前処理cache不足、sequence単位の漏洩、GPU EP fallback、fusion誤りを重点監査する。
- 設計上の仮定：650kの上限とAJ低下marginはユーザーの「同等」の操作的定義の提案。次回実行前にこの基準を固定し、結果を見て無断で緩和しない。

### 1.2 サブゴール構造

| ID | サブゴール | 成果物 | 検証 |
| :--- | :--- | :--- | :--- |
| SG-1 | 論文と現行構造を根拠に実験を設計 | `docs/knowledge_distillation_plan_20261002.md` | 一次資料、元62万/現812万の集計 |
| SG-2 | データ・teacher labelsを正しく再現 | split audit、cache budget/manifest | leakゼロ、同window・同z_ref、hash |
| SG-3 | 小型student・損失・auxを実装 | student、wrapper、unit tests | parameter上限、teacher freeze、loss mask |
| SG-4 | 対照実験と最終訓練 | smoke/pilot/full/long、resume/status | finite、同条件ablation、monitor選択 |
| SG-5 | ONNX圧縮モデルと実精度・速度を検証 | PT/ONNX、CUDA profile、paired metrics | numerical/quality/size gatesを別判定 |
| SG-6 | 完了と未達を追跡可能に記録 | 本作業書、review、docs evidence | 要求→手順→証跡→DoDを追跡 |

### 1.3 トレーサビリティ方針

| Trace ID | 要求・制約 | フェーズと手順 | 証跡とDoD |
| :--- | :--- | :--- | :--- |
| TR-1 | 専用ブランチ・論文・詳細方針 | 1 / 1–5 | plan、git HEAD、P-1/P-2 |
| TR-2 | データとlabelの整合・DA無し | 1/2/3 / 2,4,8,9,18,20,24 | `split_audit.json`,`cache_manifest.json`、D-2 |
| TR-3 | 62万程度のstudent、aux/head/loss | 2 / 6–7,10–15.5 | parameter/gradient/unit reports、D-1/D-3 |
| TR-4 | 同条件の短試験→本訓練 | 4 / 21–24.6（枝番含む） | ablation/run configs、status、D-4 |
| TR-5 | 再パラメータ化と安全な圧縮形式 | 2/3/5 / 12–15.4,19,24.6,25 | merge report、deploy manifest、D-5 |
| TR-6 | ONNX CPU/CUDA・追跡精度・実速度 | 5 / 26–29 | profile、AJ/CI、benchmark、D-6/D-7 |
| TR-7 | TDD・uv・fallback禁止・監査記録 | 全phase / 16–17,30–32 | tests/quality、作業記録、D-8 |

## 2. 作業内容

### フェーズ1 調査と実験設計

SG-1/SG-2、TR-1/TR-2。下記基準値とweb調査を再確認し、データの独立性、cache容量、サイズ・AJ基準を固定する。実装や訓練はしない。目安1〜2hだがデータ監査で変動する。

### フェーズ2 実装

SG-3/SG-5、TR-3/TR-5。named unit testsを先に書き、student A、loss/aux、cache、checkpoint/exportを小単位で実装する。Aを先に完成させる。B関連手順12–13は初回は保留し、手順23のA pilot後に採否を決める。不採用なら条件付き不実施を記録、採用なら12→13→13.1→13.2→16/17/19のB対象再検証→B pilotの順で戻る。目安1〜3日、未測定。

### フェーズ3 訓練前のテストとexport試験

SG-2/SG-3/SG-5、TR-2/TR-3/TR-5/TR-7。新規・既存unit、lint、型、teacher label pilot、未学習studentのFP32 exportを検証する。未学習モデルのexport成功をtracking精度合格と呼ばない。目安半日、未測定。

### フェーズ4 訓練実験

SG-4、TR-4。20-step smoke、32-step overfit、同条件ablation、monitor-selected full run、long-window適応の順。計算時間はpilot実測で見積もり、全組合せを無条件で走らせない。

### フェーズ5 受入検証

SG-5、TR-5/TR-6。選ばれたcheckpointでfusion、CPU/CUDA、全評価clip、latencyを測る。数値一致とteacher精度への非劣性を分離する。目安半日〜1日、前段cacheと全150本の可用性による。

### フェーズ6 記録と引渡し

SG-6、TR-7。小型reportをdocsへ残し、本作業書で未達・未実行を明示する。Gitで公開する対象は後続ユーザー指示を待ち、今回stage/commit/pushしない。

### 現行環境と維持する契約

- Python>=3.11,<3.13、`pyproject.toml`、`uv.lock`、root `.venv`。torch2.12.1+cu130、CPU ORT1.30、devにpytest/ruff/ty。justfile、AGENTS.md、CODEX.mdは調査時に無かった。Dockerも本タスクでは不要。
- GPUはRTX5090、32,607MiB。調査時は18MiB使用。`/workspace`空き105GiB、Desktop側463GiBは17:14:21 JSTの観測で、将来の使用可能量を保証しない。
- native teacherは8,117,988、上流v64相当Aは620,502、Bは設計見積620,548。teacher weight pathとSHAは第4章に固定。
- A: `temporal_mixer=vssd_cross,two_pool=true,dim=128,state_dim=64,num_heads=4,num_layers=2`。`collapse`経路を使いtoken-level F²経路は拒否。
- B: dim128、4 blocks、FFN hidden304、DW kernel7、dilation1/2/4/8、linear kernel7/3/1/identity branches、Fだけのmasked global mean。LayerNorm/SiLU/gatingはfusion外。train/deploy params別集計。
- 追加候補C: official Mamba-3幅160/state128、2層の部品計数・合算568,984。組み上げ/訓練/GPU/ONNXは未実施。768固定portable経路の一般化と5-head native smokeが必要で、本チェックリストの必須対象外。採用時は別write/review作業書を作成してから着手する。
- patch5、DINO projection64、DINO ViT-S/16 revision `114c1379950215c8b35dfcd4e90a5c251dde0d32`、DINO448、画像896、WAFT scale=-1/iters4、FB alpha.05/beta1、DA3 metric-large、photometric DA offを維持。
- geometry: bounded dlog±2、delta_uv±2px、補正UVでdepth再サンプル後exp(dlog)、Kでcamera XYZへ戻す。GTを推論へ渡さない。
- 入力8個は `ray,z_raw,visibility,uv,depth_map,dino_features,intrinsics,z_ref`。出力4個は `xyz,uv_refined,vis_logits,delta_uv`。DINO特徴384×28×28、他B/F/N/depth H/W dynamic、FP32/opset18/custom domain無し。
- `z_ref=lower_median(full B/F/N z_raw)+1e-6` はscalar。Nだけchunk化し、同じz_refと全Fを保持。padding/visibility/query maskを区別する。
- 初期smoke/pilot/fullはonline teacherを既定とする。microbatchのz_refを一度計算して両モデルへ渡す。B=1 label cacheを異なるB=32 medianへ流用しない。任意cacheは正確なFP32 z_refとbatch構成・順序をkey/manifestへ記録し、不一致は拒否する。
- teacher visibility headは未監督であり、vis KLやteacher logit confidenceは使わない。GT occlusion補助headは学習専用、評価visibilityはflow maskを維持。
- 既存monitor15と3比較動画はteacherの選択/評価に使用済み。minival150のgroup overlapと過去参照を監査し、完全未見と無条件に呼ばない。

### 実装先と責務

以下は初回計画時の実装先一覧。2026-10-02 18:40 JSTの再レビュー時点ではAのstudent/KD/cache/exportと短期train CLIは存在する。ファイルの存在を全modeの完成と混同せず、第3章の未チェック項目と第7章の証跡を確認する。Bは条件付き未実装、全F monitor/accuracyは未実装で明示例外、long用入力準備はwindow8制限の解消が必要。

本追記は専用ブランチ・作業書の確認とreview修正に限定し、既存コード・実行記録・ジョブを変更しない。追加訓練の起動や完了判定は行わない。

| 新規予定ファイル | 一操作で実装する責務 |
| :--- | :--- |
| `model/student_refiner.py` | A/Bの構造とgeometry、feature taps。パス先頭は `src/mamba3_tracker/` |
| `model/rep_temporal.py` | Bの分岐とcopy-to-deploy fusion |
| `train/distillation.py` | TeacherWrapper、auxiliary heads、RefinerDistillationLoss |
| `data/distillation_cache.py` | window/track/z_ref/teacher/input hash、strict atomic cache |
| `deployment/student_checkpoint.py` | version付きstudent schema、strict公開/訓練分離 |
| `scripts/prepare_refiner_kd.py` | mode audit/pilot/cache |
| `scripts/train_refiner_kd.py` | mode smoke/overfit/pilot/full/long |
| `scripts/export_student_refiner_onnx.py` | 学習artifactからdeploy PT/FP32 ONNXと数値report |
| `scripts/evaluate_student_refiner.py` | mode monitor/accuracy/gpu/benchmark、teacherとのpaired集計 |
| `configs/distill_refiner_vssd.yaml` | A0–A4のloss flagsと初期条件。Bは `distill_refiner_rep_tcn.yaml` |
| `tests/unit/test_refiner_distillation.py` | freeze/aux/loss/context/fairness |
| `tests/unit/test_rep_temporal.py` | 合成branchの等価性、非線形・stride不一致の拒否 |
| `tests/unit/test_distillation_cache.py` | key変化、hash/schema破損、容量・入力一致 |
| `tests/unit/test_student_deployment.py` | サイズ、schema、dynamic、chunk、CPU/GPU profile |

現行参照先は `depth_refined_tracker.py`、`train/loss.py:TrackingLossV35`、`scripts/train_depth_refined_tracker.py`、`deployment/checkpoint.py`、`model/onnx_refiner.py`、`deployment/runtime.py`、`scripts/verify_refiner_gpu.py`。共通処理は再利用し、公式Mamba専用loaderへstudentを黙って渡さない。

### 損失と実験条件の固定案

GTはV35そのまま、追加は `ramp*(.5 residual+.25 XYZ_KD+.05 feature+.05 temporal+.1 aux)`。rampはwarmup100 stepsで0→1。Huber delta.1は正規化後単位。dlogとUVは2で正規化、XYZはGT query anchor深度尺度clamp1e-3、featureは非affine LN後MSE、時間損失はteacherとの補正差分。auxは2中間補正headの平均KDと0.1 GT visibility BCE。teacher GT誤差からのwは `clip(exp(-norm(T-GT)/s_GT/.1),.05,1)`、GT lossはwで弱めない。全mask0→finite0/zero grad、モデルNaN→失敗停止。

weighted KDの分母はw.sumではなく有効位置mask M.sum().clamp_min(1)。低信頼wでgradientを弱める。時間wは隣接min、occlusion BCEは有効query/paddingの可視・不可視両方を使う。欠損GTのinactive値は有限値へ置換し、モデル/teacher NaNは黙ってmaskしない。`test_confidence_does_not_cancel` と `test_occlusion_invisible_is_supervised` を手順6でRED→GREENにする。

AdamW lr3e-4、WD.01、grad_clip1、seed42/43/44。短smoke20、overfit32、pilot200、full800/processed_clips20,000、warmup100、monitor毎10、early stop patience5/min_delta.001。window8後に16/32、必要時64で各200-step long適応、lr0.3倍。初期effective batch32、microbatch1/4/8/16/32を計測し安全な最大値を採用。ピークVRAM28〜30GB目安、最低2GB余裕。学習geometry/lossはFP32。

ablationのbest選択/early stopは全runで同じGT-only monitor lossを使い、KD込みtotalを比較しない。long候補の選択はmonitor15全FのAJで行う。初期student tensor hashとmicrobatchをA0〜A4で揃える。選択結果は `selected_config.yaml` に確定してからfullへ進める。

採用精度gate案はmetric-AJと3D-AJのteacherからの低下macro≤.003、各subset≤.005、paired sequence bootstrap10,000(seed42)の片側95%上限も同margin内。torch→deploy fusionはrtol1e-5/atol1e-6、ORTはrtol2e-4/atol5e-5。FP32 ONNX p50速度目標≤teacherの.7倍は精度と別判定。品質gate不合格はreportして止める。

## 3. 作業チェックリスト

以下は実行用チェックリストで、実施済み項目だけを証跡に基づいてチェックする。書類作成の完了は第6章P-1/P-2で別管理する。🛠欄は実際にエラーがない場合も「発生無し、手順確認」と記録してチェックする。条件付きBの不採用は未実施理由を残し、AでDoDを満たせばBを必須としない。

### フェーズ1 調査と設計

### 手順 1: 作業の起点を確認する

- [x] 🖐 **操作**: `git status --short --branch` を実行する。SG-1/TR-1。
- [x] 🔎 **確認**: 指定branch。既存未commit変更はユーザー所有として保全し、予定箇所の競合を報告する。
- [x] 🧪 **テスト**: RED不適用のread-only確認。HEADを作業記録へ記す。
- [x] 🛠 **エラー時対処**: branchが違う場合は変更を捨てず、対象branchの所在と重複編集を調査する。

### 手順 2: rawとsplitの可用性を棚卸しする

- [x] 🖐 **操作**: 既存split `growing_pool.json` とraw/minival inventoryを読み、調査メモを作る。SG-2/TR-2。
- [x] 🔎 **確認**: train1,027/monitor15の全path、clip/sequence ID、minival150の可用性と既存参照履歴を特定。
- [x] 🧪 **テスト**: RED不適用。件数・hash・欠落一覧を記録し、clip名だけで独立性を判定しない。
- [x] 🛠 **エラー時対処**: 欠落は一覧化して止める。3動画を150本の代替とするfallbackは禁止。

### 手順 2.1: 欠落minivalを別領域へ取得する

- [ ] 🖐 **操作**: `uv run --no-sync python temp/download_kd_minival.py --split minival --data-root /workspace/vmamba3_eval/raw --train-manifest /workspace/vmamba3_eval/source_manifest.json --state-json /workspace/vmamba3_eval/download_status.json --raw-budget-gb 30 --total-data-budget-gb 60 --reserve-gb 30` を起動する。SG-2/TR-2。HF revisionを1575ec135a22e924d1702cd15e76527bcb89cc6eに固定。
- [ ] 🔎 **確認**: 3subset各50本、計150本、CRC/必須NPZ key確認済み。既存train/cacheの保存先とは分離。
- [ ] 🧪 **テスト**: download_statusの欠落0と後続hash台帳を照合。取得中は他の設計・実装手順を進め、訓練・最終評価前に戻って完了を確認する。
- [ ] 🛠 **エラー時対処**: HTTP失敗は同一保存先で検証済みNPZから再開。raw30GB/評価領域60GB/空き30GB制限を超える場合は容量見積を記録して停止。旧データは削除しない。

### 手順 2.2: minivalのmetric depthを準備する

- [ ] 🖐 **操作**: `temp/prepare_kd_minival_depth.py` を既存DA3 scriptのwrapperとして実行し、新規 `/workspace/vmamba3_eval/depth` へ保存する。SG-2/TR-2。snapshot4010e39f3634a45bc60553321fb49fb760bd594e/model SHA bbea5b0b3ee389849cffa7ddae89de064a90abd2b055fc5aa99aac68db324776を起動前に検証。
- [ ] 🔎 **確認**: 全150本のdepth ready marker、frame数/CRC、process_res504、uint16 min/max圧縮。既存teacher depth cacheは無変更。
- [ ] 🧪 **テスト**: native/ONNX student/teacherは同一depthを読む。depth処理はmetric独立frame方式でchunk4〜8、GPU余裕12GB。取得中のrawも逐次処理し、全150readyで終了。
- [ ] 🛠 **エラー時対処**: 評価領域raw+入力+depth合計60GB上限を維持。認証・hash不一致・欠落・CRCを黙って代替しない。GPU OOM時は既存AdaptiveBatchSizeで縮小し再試行を記録する。

### 手順 3: 論文と圧縮対象を再確認する

- [x] 🖐 **操作**: `docs/knowledge_distillation_plan_20261002.md` の一次資料とsource参照を読み直す。SG-1/TR-1。
- [x] 🔎 **確認**: A優先/B条件付き、上流620,502、teacher8,117,988、未監督visの根拠がある。
- [x] 🧪 **テスト**: RED不適用。論文事実と適用提案を混同していないことを記録。
- [x] 🛠 **エラー時対処**: upstream更新で構造が違えばpinしたrevisionで比較し直す。現行mainの値に置き換えない。

### 手順 4: label cacheの容量とcontextを設計する

- [x] 🖐 **操作**: `cache_budget` の仕様を設計メモへ追記する。SG-2/TR-2。
- [x] 🔎 **確認**: 同window・同track・同microbatch z_ref、online既定、10-window任意cache pilot、追加20GB上限/空き10GB維持。
- [x] 🧪 **テスト**: RED不適用。全Fラベルのwindow切出しとB依存z_refの不一致を例に検算。
- [x] 🛠 **エラー時対処**: 容量見積不足ならpilot後までfull cacheを開始しない。旧cacheは削除しない。

### 手順 5: 評価条件を事前固定する

- [x] 🖐 **操作**: `result/refiner_kd_20261002/protocol.json` の設計値を記述する。SG-1/SG-5/TR-1/TR-6。
- [x] 🔎 **確認**: サイズ・AJ/CI・numerical gates、monitor/test役割、seedsとablation、速度条件が全て数値入り。
- [x] 🧪 **テスト**: RED不適用。testを見てmarginを変更しないという方針を記録。
- [x] 🛠 **エラー時対処**: holdout独立性が証明できない場合はそのscopeを明記し、完全未見主張のDoDを満たした扱いにしない。

### フェーズ2 実装

### 手順 6: studentと蒸留のRED testsを作る

- [x] 🖐 **操作**: `tests/unit/test_refiner_distillation.py` を作成する。SG-3/TR-3。
- [x] 🔎 **確認**: `test_vssd_parameter_budget`、`test_teacher_is_frozen`、`test_teacher_vis_is_not_distilled`、`test_aux_absent_from_deploy`、`test_all_masked_zero_grad`、`test_nonfinite_fails`、`test_same_batch_context`、`test_ablation_only_changes_loss_flags` の期待値が明示。未実装moduleのimportはtest関数内に限定し、個別GREEN判定を妨げない。
- [x] 🧪 **テスト**: `uv run --no-sync pytest tests/unit/test_refiner_distillation.py -q` は新module未実装でRED。
- [x] 🛠 **エラー時対処**: 依存importエラーと意図した未実装REDを区別する。GTを推論fixtureへ入れない。

### 手順 7: VSSD student Aを実装する

- [x] 🖐 **操作**: `src/mamba3_tracker/model/student_refiner.py` にAを実装する。SG-3/TR-3。
- [x] 🔎 **確認**: 620,502 params、geometry/8入力4出力、teacher共通tensorのshape一致コピー、collapseのみ、明示feature taps。
- [x] 🧪 **テスト**: `uv run --no-sync pytest tests/unit/test_refiner_distillation.py -k 'vssd_parameter or aux_absent' -q` をGREEN。未実装loss testsはまだREDで良い。
- [x] 🛠 **エラー時対処**: geometry差はteacherの入力/patch/GridSampleを比較する。非一致weightsをreshapeしない。

### 手順 8: cacheのRED testsを作る

- [x] 🖐 **操作**: `tests/unit/test_distillation_cache.py` を作成する。SG-2/TR-2。
- [x] 🔎 **確認**: `test_key_changes_with_window_tracks_zref`、`test_wrong_split_rejected`、`test_corrupt_hash_rejected`、`test_budget_prevents_write`、`test_batch_context_invalidates_key` を含む。
- [x] 🧪 **テスト**: `uv run --no-sync pytest tests/unit/test_distillation_cache.py -q` はmodule未実装でRED。
- [x] 🛠 **エラー時対処**: 外部rawへテストwriteしない。pytest tmp_pathを使う。

### 手順 9: strict label cacheを実装する

- [x] 🖐 **操作**: `src/mamba3_tracker/data/distillation_cache.py` を実装する。SG-2/TR-2。
- [x] 🔎 **確認**: teacher/input/split/window/track/z_ref/precision/schema key、atomic write/hash audit、容量超過・missを明示。
- [x] 🧪 **テスト**: `uv run --no-sync pytest tests/unit/test_distillation_cache.py -q` をGREEN。
- [x] 🛠 **エラー時対処**: partial fileは成功cacheとして扱わず当該新規fileだけ隔離する。旧前処理cacheを上書きしない。

### 手順 10: teacher wrapperと専用損失を実装する

- [x] 🖐 **操作**: `src/mamba3_tracker/train/distillation.py` を実装する。SG-3/TR-3。
- [x] 🔎 **確認**: teacher stopgrad/eval固定、auxはwrapper所有、V35基準を維持、各mask分母独立、同入力context、項別grad log。
- [x] 🧪 **テスト**: `uv run --no-sync pytest tests/unit/test_refiner_distillation.py -k 'not ablation_only_changes_loss_flags' -q` をGREEN。config依存testは手順11後にGREEN。teacher tensorの前後bitwise不変も検証。
- [x] 🛠 **エラー時対処**: gradがteacherへ流れたらdetach境界とoptimizer param groupを修正。teacher vis KLを追加しない。

### 手順 11: student設定を作る

- [x] 🖐 **操作**: `configs/distill_refiner_vssd.yaml` を作る。SG-3/SG-4/TR-3/TR-4。
- [x] 🔎 **確認**: teacher SHA、DINO pin、DA off、split絶対path、window/batch/context、A0–A4 flags、optimizerと上限を明示。
- [x] 🧪 **テスト**: 手順6の `test_ablation_only_changes_loss_flags` を設定読込後GREEN。GT-onlyとKDのdata/optimizer/初期tensor/microbatch条件一致。
- [x] 🛠 **エラー時対処**: 既存v64を黙って上書きしない。不明なconfig keyはfail-closed。

### 手順 12: 候補Bのfusion RED testsを作る

- [ ] 🖐 **操作**: A pilot後にB採用を判断した場合だけ `tests/unit/test_rep_temporal.py` を作る。SG-3/SG-5/TR-3/TR-5。
- [ ] 🔎 **確認**: F1/31/300、dilation1/2/4/8、端padding、random bias、identityのpre/post比較と `test_nonlinear_branch_rejected`。
- [ ] 🧪 **テスト**: `uv run --no-sync pytest tests/unit/test_rep_temporal.py -q` はmodule未実装でRED。
- [ ] 🛠 **エラー時対処**: Aのみで進む場合は条件付き不実施を記録する。shape不整合をpaddingで隠さない。

### 手順 13: 候補Bの時間blockを実装する

- [ ] 🖐 **操作**: B採用時のみ `model/rep_temporal.py` を実装する。SG-3/SG-5/TR-3/TR-5。
- [ ] 🔎 **確認**: kernel7/3/1/identityの線形和だけfusion、元training weightsを破壊しないdeploy copy、global poolはFのみ。
- [ ] 🧪 **テスト**: `uv run --no-sync pytest tests/unit/test_rep_temporal.py -q` をGREEN。student factoryへ接続する際もbudget≤650,000をテストする。
- [ ] 🛠 **エラー時対処**: BN/LNやgatingをkernelに統合しない。B接続/configは13.1/13.2で別に実装する。

### 手順 13.1: Bをstudent factoryへ接続する

- [ ] 🖐 **操作**: B採用時に `model/student_refiner.py` のfactoryへRepTCNを接続する。SG-3/TR-3。
- [ ] 🔎 **確認**: 共有head/geometry、deploy見積620,548、上限650k。Nをまたぐpool無し。
- [ ] 🧪 **テスト**: 先に `test_rep_tcn_parameter_budget` を追加してRED→GREEN。track permutation一致。
- [ ] 🛠 **エラー時対処**: teacher Mambaを等価なConvへ変換した扱いにしない。91frames局所/global meanの制約を明記。

### 手順 13.2: Bの設定を作る

- [ ] 🖐 **操作**: B採用時に `configs/distill_refiner_rep_tcn.yaml` を作る。SG-3/TR-3。
- [ ] 🔎 **確認**: Aと同data/optimizer/予算、構造とhintの2/4block対応だけ変更。B0=GT-only、B1〜B4=同じKD追加順。
- [ ] 🧪 **テスト**: 設定比較testを先に作りRED→GREEN。
- [ ] 🛠 **エラー時対処**: 複数の実験条件を同時に変えて構造差と混同しない。

### 手順 14: deployのRED testsを作る

- [ ] 🖐 **操作**: `tests/unit/test_student_deployment.py` を作る。SG-5/TR-5/TR-6。
- [ ] 🔎 **確認**: `test_strict_student_schema`、`test_no_teacher_aux_in_public`、`test_dynamic_axes`、`test_track_chunk_invariance`、`test_legacy_teacher_unchanged`、`test_cuda_profile_rejects_float_cpu`。
- [ ] 🧪 **テスト**: `uv run --no-sync pytest tests/unit/test_student_deployment.py -q` は未実装loader/exportでRED。
- [ ] 🛠 **エラー時対処**: GPU非搭載のunitはprofile fixtureで拒否を検証。実GPU gateをskip扱いで合格にしない。

### 手順 15: version付きstudent loaderを実装する

- [ ] 🖐 **操作**: `deployment/student_checkpoint.py` を実装する。SG-5/TR-5。
- [ ] 🔎 **確認**: weights_only、schema/architecture/config/shape/dtype/hash strict、aux除去リスト明示、trainingとpublicの分離。
- [ ] 🧪 **テスト**: `uv run --no-sync pytest tests/unit/test_student_deployment.py -k 'schema or public or legacy' -q` をGREEN。
- [ ] 🛠 **エラー時対処**: official teacher loaderをstudentへfallbackしない。未知architectureは拒否して記録。

### 手順 15.1: prepare CLIを実装する

- [ ] 🖐 **操作**: `scripts/prepare_refiner_kd.py` を実装する。SG-2/TR-2。
- [ ] 🔎 **確認**: 第4章のconfig/mode/limit/out-dir仕様、auditはread-only、pilotはtrainのみ、cacheは容量gate付き。
- [ ] 🧪 **テスト**: 手順8に `test_prepare_cli_rejects_test_as_train` を先に作りRED→GREEN。`--help` exit0。
- [ ] 🛠 **エラー時対処**: mode不明・teacher SHA違い・input欠落は明示失敗。データ取得を自動開始しない。

### 手順 15.2: train CLIを実装する

- [ ] 🖐 **操作**: `scripts/train_refiner_kd.py` を実装する。SG-4/TR-4。
- [ ] 🔎 **確認**: mode/ablation/seed/out-dir/resume仕様、online teacher既定、ablationはCLI指定かconfigに必須、GT-only monitorでbest選択。
- [ ] 🧪 **テスト**: 手順6に `test_train_cli_strict_resume` を先に作りRED→GREEN。teacher/split/context変更のresume拒否。
- [ ] 🛠 **エラー時対処**: 既存run dirを上書きせず拒否、resumeを明示要求。新candidateは別dir。

### 手順 15.3: student export CLIを実装する

- [x] 🖐 **操作**: `scripts/export_student_refiner_onnx.py` を実装する。SG-5/TR-5。
- [x] 🔎 **確認**: init-config/ckpt/selection-manifest排他、aux除去/fusion/strict deploy loading、architecture-aware memory estimate、FP32の8入力4出力。
- [x] 🧪 **テスト**: 手順14に `test_export_cli_requires_exactly_one_source` と `test_linear_workspace_not_teacher_quadratic` を先に作りRED→GREEN。
- [x] 🛠 **エラー時対処**: unknown student/fusion不一致は失敗。元teacher loaderとF² estimatorへfallbackしない。

### 手順 15.4: paired evaluation CLIを実装する

- [ ] 🖐 **操作**: `scripts/evaluate_student_refiner.py` を実装する。SG-5/TR-6。
- [ ] 🔎 **確認**: mode monitor/accuracy/gpu/benchmark、provider cpu/cuda明示、NPZ-only GPU replayとroot teacher準備を分離、GT/flow-vis protocol固定。
- [ ] 🧪 **テスト**: 手順14に `test_evaluator_rejects_missing_clip`、`test_benchmark_excludes_warmup`、`test_partial_is_not_full_accuracy` を先に作りRED→GREEN。
- [ ] 🛠 **エラー時対処**: GPU環境へnative teacher依存を強制installしない。rootで準備したhash付きfixtureをisolated ORTへ渡す。

### 手順 15.5: 型チェック対象を追加する

- [ ] 🖐 **操作**: `pyproject.toml` のty includeへ新student/KD/cache/CLI/testsの明示pathを追加する。SG-3/TR-7。
- [ ] 🔎 **確認**: 既存coverageを削らず、新コードも検査対象。uv.lockと依存versionは不変。
- [ ] 🧪 **テスト**: `uv run --no-sync ty check` の対象と結果を確認し、新規の型エラーを実装phaseで直す。
- [ ] 🛠 **エラー時対処**: 除外で回避しない。B未作成pathはB採用時に独立編集で追加。

### フェーズ3 訓練前の検証

### 手順 16: 新規unitを一括検証する

- [x] 🖐 **操作**: `uv run --no-sync pytest tests/unit/test_refiner_distillation.py tests/unit/test_distillation_cache.py tests/unit/test_student_deployment.py -q` を実行する。SG-3/TR-7。
- [x] 🔎 **確認**: 全GREEN、parameter budget、same-context、teacher bitwise不変、failure pathsが証跡にある。
- [x] 🧪 **テスト**: B採用時は同コマンドへ `tests/unit/test_rep_temporal.py` を明示追加してGREENにする。
- [x] 🛠 **エラー時対処**: 失敗testが残る間は実データ訓練へ進まない。

### 手順 17: 既存unitの回帰を確認する

- [x] 🖐 **操作**: `uv run --no-sync pytest tests/unit -q` を実行する。SG-3/TR-7。
- [x] 🔎 **確認**: baselineから新規不具合無し。既知外部依存skipは理由を記録する。
- [x] 🧪 **テスト**: 旧teacher/checkpoint/CPU/GPUのprotocol testsも対象。export gate代わりにskipしない。
- [x] 🛠 **エラー時対処**: 既存失敗と今回差分を切り分け、testを削除して通さない。

### 手順 18: splitとgroupの監査を実行する

- [ ] 🖐 **操作**: 第4章のprepare `--mode audit` を実行する。SG-2/TR-2。
- [ ] 🔎 **確認**: train/monitor/testの件数、同sequence混入、過去teacher集合、欠落、hashを `split_audit.json` に保存。
- [ ] 🧪 **テスト**: deliberate overlap fixtureを拒否する `test_wrong_split_rejected` はGREEN。
- [ ] 🛠 **エラー時対処**: teacherのtrain sequence混入はstudent splitの変更だけで消えない。独立holdout確保まで精度主張を留保する。

### 手順 19: 未学習studentでONNX exportの可否を検証する

- [x] 🖐 **操作**: 第4章のexport CLIの `--init-config` を実行する。SG-5/TR-5。
- [x] 🔎 **確認**: 標準opset18、8/4 contract、dynamic F/N、CPU一致、no-aux、deploy parameter≤650k。
- [x] 🧪 **テスト**: test_dynamic_axes/chunk_invarianceがGREEN。F1/8/31/128/257/300/600も検証する。
- [x] 🛠 **エラー時対処**: ONNX unsupported opはMatMul等で等価な標準演算へ直す。custom opや時間固定に黙って変更しない。

### 手順 20: teacher labelsの10-window pilotを実行する

- [ ] 🖐 **操作**: 第4章のprepare `--mode pilot` を実行する。SG-2/TR-2。
- [ ] 🔎 **確認**: teacher native BF16→FP32 labels、入力・window・track・z_ref一致、容量と秒/window、現空き容量をreport。
- [ ] 🧪 **テスト**: cache再読込がhash一致し、teacher直接計算との差を記録。許可したcast以外の変更無し。
- [ ] 🛠 **エラー時対処**: HF認証不備は環境変数の存在/アクセスだけ確認し秘密を出さない。cacheがbatch context不一致なら再利用を拒否する。

### フェーズ4 訓練

### 手順 21: 20-step GPU smokeを実行する

- [ ] 🖐 **操作**: 第4章のtrain `--mode smoke --ablation A1` を実行する。SG-4/TR-4。
- [ ] 🔎 **確認**: studentが更新、teacher bitwise不変、全loss/grad finite、peak VRAMとterm grad log、status完了。
- [ ] 🧪 **テスト**: step10も保存し、手順21.1でresume再現性を検証する。
- [ ] 🛠 **エラー時対処**: OOMはmicrobatchを下げ、windowは維持。NaNはartifactを保全して原因を調査、成功stepに数えない。

### 手順 21.1: smokeのresumeを検証する

- [x] 🖐 **操作**: smokeのstep10 checkpointから新dir `smoke_resume_A1_s42` へresumeしてstep20まで実行する。SG-4/TR-4。
- [x] 🔎 **確認**: 元smokeとstudent出力がrtol1e-5/atol1e-6以内、teacher/RNG/optimizer/split/config一致。
- [x] 🧪 **テスト**: smoke modeはstep10/20を保存。hash違いresumeは `test_train_cli_strict_resume` で拒否。
- [x] 🛠 **エラー時対処**: 元runを上書きしない。RNG/windowの再サンプリング差を特定してからfullを許可。

### 手順 22: 固定小集合で32-step overfitを実行する

- [x] 🖐 **操作**: 第4章のtrain `--mode overfit --ablation A1` を実行する。SG-4/TR-4。
- [x] 🔎 **確認**: teacher target残差に対するstudent誤差がstep0より低下、GT loss・clamp率の悪化も併記。
- [x] 🧪 **テスト**: tiny finite fixtureで少なくとも出力KD lossが初期より低下。汎化精度とは呼ばない。
- [x] 🛠 **エラー時対処**: residual normalization/grad、zero-init head、query maskを確認し、full runで解消しようとしない。

### 手順 23: 同条件ablation pilotを実行する

- [x] 🖐 **操作**: 第4章のtrain `--mode pilot --ablation A0` を1run実行する。SG-4/TR-4。
- [x] 🔎 **確認**: 各runは別dir、200-step上限、同seed/split/window/optimizer。A0–A4の比較と候補選択はmonitorだけ。
- [x] 🧪 **テスト**: A0のmonitor/statusを保存し、A1〜A4は23.1〜23.4で独立に実行する。
- [x] 🛠 **エラー時対処**: 結果未取得runを平均へ含めない。全項追加が悪化したら直前ablationと比較する。

### 手順 23.1: A1の200-step pilotを実行する

- [x] 🖐 **操作**: 第4章のpilotコマンドを `--ablation A1`、dir `pilot_A1_s42` として1run実行する。SG-4/TR-4。
- [x] 🔎 **確認**: A0と同じ初期tensor、microbatch、window/split/seed/optimizer。bestはGT-only monitor。
- [x] 🧪 **テスト**: run config/hash、status、項別loss/gradとclamp率を保存。test AJは読まない。
- [x] 🛠 **エラー時対処**: KD totalをablation間で比較しない。異なるbatchのrunを同条件表へ含めない。

### 手順 23.2: A2の200-step pilotを実行する

- [x] 🖐 **操作**: 第4章のpilotコマンドを `--ablation A2`、dir `pilot_A2_s42` として1run実行する。SG-4/TR-4。
- [x] 🔎 **確認**: A0と同じ初期tensor、microbatch、window/split/seed/optimizer。bestはGT-only monitor。
- [x] 🧪 **テスト**: run config/hash、status、項別loss/gradとclamp率を保存。test AJは読まない。
- [x] 🛠 **エラー時対処**: KD totalをablation間で比較しない。異なるbatchのrunを同条件表へ含めない。

### 手順 23.3: A3の200-step pilotを実行する

- [x] 🖐 **操作**: 第4章のpilotコマンドを `--ablation A3`、dir `pilot_A3_s42` として1run実行する。SG-4/TR-4。
- [x] 🔎 **確認**: A0と同じ初期tensor、microbatch、window/split/seed/optimizer。bestはGT-only monitor。
- [x] 🧪 **テスト**: run config/hash、status、項別loss/gradとclamp率を保存。test AJは読まない。
- [x] 🛠 **エラー時対処**: KD totalをablation間で比較しない。異なるbatchのrunを同条件表へ含めない。

### 手順 23.4: A4の200-step pilotを実行する

- [x] 🖐 **操作**: 第4章のpilotコマンドを `--ablation A4`、dir `pilot_A4_s42` として1run実行する。SG-4/TR-4。
- [x] 🔎 **確認**: A0と同じ初期tensor、microbatch、window/split/seed/optimizer。bestはGT-only monitor。
- [x] 🧪 **テスト**: run config/hash、status、項別loss/gradとclamp率を保存。test AJは読まない。
- [x] 🛠 **エラー時対処**: KD totalをablation間で比較しない。異なるbatchのrunを同条件表へ含めない。

### 手順 23.5: fullの候補設定を固定する

- [x] 🖐 **操作**: monitor pilot結果から `result/refiner_kd_20261002/selected_config.yaml` を作る。SG-4/TR-4。
- [x] 🔎 **確認**: A1〜A4で最良GT-only monitor loss、同値ならloss項の少ないcandidate。A0との優劣とB採否も記録。
- [x] 🧪 **テスト**: configにablation/parameter/teacher hash/pilot根拠を記す。B採用なら12→13→13.1→13.2→16/17/19再検証→B pilot→選択へ戻る。
- [x] 🛠 **エラー時対処**: 全KDがA0より悪ければKD効果未証明。予算/品質不適合を最良として採用しない。

### 手順 24: 選択した候補を最終訓練する

- [x] 🖐 **操作**: 第4章のtrain `--mode full` をmonitor選択済みの1candidateで実行する。SG-4/TR-4。
- [x] 🔎 **確認**: 上限800/20k clipsかearly stopでstatus完了、last/best、term logs、freeze/cache miss/VRAM reportを保存。
- [x] 🧪 **テスト**: seed43/44とlong-windowは24.2〜24.5。GT-only monitorでbestを選び、24.6でdeploy対象を固定。test AJは使わない。
- [x] 🛠 **エラー時対処**: resumeはteacher/split/cache/config/RNG hash strict。旧teacherのval .16135とのloss直接比較はしない。

### 手順 24.1: 同予算のGT-only対照を最終訓練する

- [x] 🖐 **操作**: 選択構造のA0（BならB0）を同じfull予算/seed42で `full_control_s42` に学習する。SG-4/TR-4。
- [x] 🔎 **確認**: KD候補と初期化/data/optimizer/最大step/early stop条件一致。
- [x] 🧪 **テスト**: pilot200対KD full800の差をKD効果としない。実steps/processed clipsを記録。
- [x] 🛠 **エラー時対処**: 対照を省いた場合はKD効果の検証を未達とし、圧縮成功と区別。

### 手順 24.2: seed43の再現性を確認する

- [x] 🖐 **操作**: selected_configでfull seed43を `full_s43` に学習する。SG-4/TR-4。
- [x] 🔎 **確認**: seedだけ違うconfig、best/status保存。
- [x] 🧪 **テスト**: seed42/43のmonitor GT loss/全F AJをreport。
- [x] 🛠 **エラー時対処**: testで良いseedを選ばず、seed42を基準とする。

### 手順 24.3: seed44の再現性を確認する

- [x] 🖐 **操作**: selected_configでfull seed44を `full_s44` に学習する。SG-4/TR-4。
- [x] 🔎 **確認**: 同条件、3 seedsのmonitor平均と範囲。
- [x] 🧪 **テスト**: failureを除いて都合の良い平均を作らない。
- [x] 🛠 **エラー時対処**: 失敗は未完としてresume条件を記録。

### 手順 24.4: window16で適応する

- [ ] 🖐 **操作**: seed42 bestからwindow16/200-step/lr0.3倍の適応を `long_w16_s42` に実行する。SG-4/TR-2/TR-4。
- [ ] 🔎 **確認**: teacherも16、対応flow/depth/DINO inputsをstrict準備。monitor全F AJを保存。
- [ ] 🧪 **テスト**: window8 labelsを流用しないcontext test。GT-only controlも同条件の別run手順24.4.1で比較。
- [ ] 🛠 **エラー時対処**: 前処理window cache不足なら準備前に停止。Fを無断で8へ戻さない。

### 手順 24.4.1: window16のGT-only対照を実行する

- [ ] 🖐 **操作**: 対照bestから同条件のwindow16適応を `long_control_w16_s42` に実行する。SG-4/TR-4。
- [ ] 🔎 **確認**: KDとはloss flagsだけ違い、input/予算/lr/seed一致、monitor全F評価を保存。
- [ ] 🧪 **テスト**: config比較と実stepsを記録。GT-only/KDのmonitorを比較。
- [ ] 🛠 **エラー時対処**: 対照欠落ならlong段階のKD効果は未検証と報告。

### 手順 24.5: window32で適応する

- [ ] 🖐 **操作**: window16 bestからwindow32/200-stepの適応を `long_w32_s42` に実行する。学習率はshort基準3e-4の0.3倍である9e-5に固定し、window16の学習率へ再度0.3を掛けない。GT-only対照にも同じ値を使う。SG-4/TR-2/TR-4。
- [ ] 🔎 **確認**: teacherも32、入力context一致、同じmonitor全F評価。
- [ ] 🧪 **テスト**: GT-only controlも別run手順24.5.1で同条件比較。必要時64は任意追加runとして記録。
- [ ] 🛠 **エラー時対処**: OOMならbatchを下げる。test採用条件を緩めない。

### 手順 24.5.1: window32のGT-only対照を実行する

- [ ] 🖐 **操作**: 対照window16 bestから同条件のwindow32適応を `long_control_w32_s42` に実行する。SG-4/TR-4。
- [ ] 🔎 **確認**: KDとはloss flagsだけ違い、input/予算/lr/seed一致、monitor全F評価を保存。
- [ ] 🧪 **テスト**: short/long対照双方の学習量を明記し、異なる予算の比較を避ける。
- [ ] 🛠 **エラー時対処**: 任意の64を試す場合も対照を追加。実施しなければ64未実施と記録。

### 手順 24.6: deploy対象checkpointを固定する

- [ ] 🖐 **操作**: short/long seed42のmonitor全F AJから `selected_checkpoint.json` を作成する。SG-4/SG-5/TR-4/TR-5。
- [ ] 🔎 **確認**: 実checkpoint絶対path/SHA/architecture/config、monitor manifest/hash。metric-AJ、同値なら3D-AJ、さらに同値ならshortを優先。
- [ ] 🧪 **テスト**: test未参照。seed43/44は再現性報告に使いseed42を置換しない。GT-only controlも同じ選択規則。
- [ ] 🛠 **エラー時対処**: 全F monitorが未完なら選択保留。8-frame GT lossだけをlong品質の証明にしない。

### フェーズ5 受入検証

### 手順 25: best studentのdeploy artifactを作る

- [ ] 🖐 **操作**: 第4章のexport CLIを `selected_checkpoint.json` に対して実行する。SG-5/TR-5。
- [ ] 🔎 **確認**: deploy PT/ONNX/manifest、pre/post fusion誤差、params≤650k、teacher/aux/optimizer無し、schema/hash一致。
- [ ] 🧪 **テスト**: rtol1e-5/atol1e-6 fusion、rtol2e-4/atol5e-5 CPU4出力、strict reloadがGREEN。
- [ ] 🛠 **エラー時対処**: fusion不一致を許容誤差拡大で解決しない。元training checkpointを保全する。

### 手順 26: CUDA EPを実機検証する

- [ ] 🖐 **操作**: 第4章のevaluate `--mode gpu` をGPU専用uv環境で実行する。SG-5/TR-6。
- [ ] 🔎 **確認**: 全4出力一致、CUDA計算eventsあり、主要float演算CUDA、CPUは整数shapeのみ。dynamic/chunkもreport。
- [ ] 🧪 **テスト**: provider名だけの成功、CPU fallback、未知CPU tensor typesを拒否するprofile testをGREEN。
- [ ] 🛠 **エラー時対処**: CUDA13/cuDNN9のlib pathとTF32 offを確認。CPUへ切替えてGPU合格と報告しない。

### 手順 27: 同入力で公式評価を実行する

- [ ] 🖐 **操作**: 第4章のevaluate `--mode accuracy` をteacherとdeploy studentに実行する。SG-5/TR-6。
- [ ] 🔎 **確認**: 監査済み全評価clip、同WAFT/DA3/DINO/query/flow-vis、metric-AJ/3D-AJとsubset別・CI、3動画は別scope。
- [ ] 🧪 **テスト**: lossではなくAJ/CIの事前marginで非劣性を判定。欠落clipで成功扱いを拒否するtest。
- [ ] 🛠 **エラー時対処**: 部分評価は部分としてreportする。testを再学習に使って同一の独立testと呼ばない。

### 手順 28: 実速度を計測する

- [ ] 🖐 **操作**: 第4章のevaluate `--mode benchmark` を1providerずつ実行する。SG-5/TR-6。
- [ ] 🔎 **確認**: threads4/同B,F,N,chunk、warmup20/100 trials、p50/p95、RSS/VRAM、H2D/D2H有無、ONNX bytesを記録。
- [ ] 🧪 **テスト**: GPU同期とwarmup除外を測定コードtestで確認。model-only≤teacher .7倍という目標と実測を分ける。
- [ ] 🛠 **エラー時対処**: 他GPU jobや初回compileで乱れた計測は条件を記録して再計測し、速い1回だけ選ばない。

### 手順 29: 受入判定を固定する

- [ ] 🖐 **操作**: `acceptance.json` にサイズ/数値/CPU/CUDA/精度/速度の各判定を記述する。SG-5/TR-6。
- [ ] 🔎 **確認**: 全mandatory gate通過か、未達項目と原因が明示。量子化やBの不実施も区別。
- [ ] 🧪 **テスト**: AJ margin等を結果取得後に変更していないことをprotocol hashと照合。
- [ ] 🛠 **エラー時対処**: 精度不足なら研究結果として未達を報告。サイズだけ合格で同等モデルと呼ばない。

### フェーズ6 記録と引渡し

### 手順 30: 小型証跡をdocsへ整理する

- [ ] 🖐 **操作**: `docs/evidence/refiner_kd_20261002/` へ集計JSONを保存する。SG-6/TR-7。
- [ ] 🔎 **確認**: run config/split hash/cache summary/params/metrics/acceptance/quality、秘密・host path・weights・巨大profile本体無し。
- [ ] 🧪 **テスト**: JSON finiteと文書リンク、artifact SHA一致を確認。証跡scopeが本体と一致。
- [ ] 🛠 **エラー時対処**: root .gitignoreを緩めない。weights/videos/cacheはローカルに保全。

### 手順 31: 最終品質を検証する

- [ ] 🖐 **操作**: `uv run --no-sync pytest tests/unit -q` を実行する。SG-6/TR-7。
- [ ] 🔎 **確認**: unitが成功。lint/format/ty/diffは31.1〜31.4で別判定。coverage外を成功と数えない。
- [ ] 🧪 **テスト**: lint/format/型/diffは31.1〜31.4で独立に実行し、exit codeとlogを記録。
- [ ] 🛠 **エラー時対処**: unrelated既存変更は修正せずreport。lint対象を無断で狭めない。

### 手順 31.1: 変更コードをlintする

- [ ] 🖐 **操作**: 第4章の `uv run --no-sync ruff check` を変更Pythonとnew testsへ実行する。SG-6/TR-7。
- [ ] 🔎 **確認**: exit0。Bならrep_temporalも対象。
- [ ] 🧪 **テスト**: 対象一覧と出力をquality.jsonへ記録。
- [ ] 🛠 **エラー時対処**: unrelated既存問題と新規問題を区別。対象を黙って狭めない。

### 手順 31.2: formatを検証する

- [ ] 🖐 **操作**: 第4章の `uv run --no-sync ruff format --check` をlintと同じ対象へ実行する。SG-6/TR-7。
- [ ] 🔎 **確認**: exit0。修正時は通常formatterでdiff確認。
- [ ] 🧪 **テスト**: 対象一覧とexit codeを記録。
- [ ] 🛠 **エラー時対処**: 無関係なユーザー変更を混ぜない。

### 手順 31.3: 型チェックを実行する

- [ ] 🖐 **操作**: 手順15.5でinclude追加済みの状態で `uv run --no-sync ty check` を実行する。SG-6/TR-7。
- [ ] 🔎 **確認**: exit0、student/KD/cache/export/evaluateがcoverageに入る。
- [ ] 🧪 **テスト**: include変更と検査対象を記録。
- [ ] 🛠 **エラー時対処**: 広範なignoreで隠さず境界APIを修正。

### 手順 31.4: diffを検証する

- [ ] 🖐 **操作**: `git diff --check` を実行する。SG-6/TR-7。
- [ ] 🔎 **確認**: exit0、task scopeのみ、temp/weights/resultはignore維持。
- [ ] 🧪 **テスト**: untracked docs/tempも末尾空白を別途検査。
- [ ] 🛠 **エラー時対処**: stage/commit/pushしない。binary混入を除去する前に対象を保全。

### 手順 32: DoDと未達を記録する

- [ ] 🖐 **操作**: 本作業書第6章と第7章を実績で更新する。SG-6/TR-7。
- [ ] 🔎 **確認**: future計画を完了扱いにせず、訓練・評価・GPU・独立性それぞれのstatusと保存先が読める。
- [ ] 🧪 **テスト**: `review-written-workdoc` で作業記録とDoDの整合を再確認。commit/pushは別のユーザー指示が必要。
- [ ] 🛠 **エラー時対処**: 未達DoDはuncheckedで残し、次の判断と証跡を記す。

## 4. 作業に使用するコマンド参考情報

すべてrepo rootから実行。下記は到達目標のCLI契約で、一部は実装済みだが全mode完成ではない。手順6〜15.4の証跡と実装を照合してから実行する。既存run dirへの再実行で成果物を上書きしない。flagsを変える場合は作業書・config・testsも更新する。

### 現在使用可能な環境確認 TR-1/TR-7

```bash
date '+%Y-%m-%d %H:%M:%S %Z%z'
git status --short --branch
git rev-parse HEAD
uv --version
df -h /workspace /home/kasm-user/Desktop
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
```

root .venvがある場合は `uv run --no-sync` で既存CUDA stackを維持する。新規cloneで必要なら `uv sync --frozen --extra onnx --group dev`。justfile無しなのでjustは使わない。uv/lockが無い場合は記録して止め、pipへのfallbackは禁止。

### 予定prepare CLI TR-2

```bash
uv run --no-sync python scripts/prepare_refiner_kd.py --config configs/distill_refiner_vssd.yaml --mode audit --out-dir result/refiner_kd_20261002/preflight
uv run --no-sync python scripts/prepare_refiner_kd.py --config configs/distill_refiner_vssd.yaml --mode pilot --limit 10 --out-dir result/refiner_kd_20261002/cache_pilot
```

auditはrawとteacher/split hashをread-onlyで検査、pilotはtrainの10 windowsだけを生成する。cache modeはpilot容量とcontext gate成功後にのみ別手順を追加して開始する。cache rootは `/workspace/vmamba3_data/tapvid3d_distillation_cache/`、manifestは新namespace。

### 予定train CLI TR-3/TR-4

```bash
uv run --no-sync python scripts/train_refiner_kd.py --config configs/distill_refiner_vssd.yaml --mode smoke --ablation A1 --seed 42 --out-dir result/refiner_kd_20261002/smoke_A1_s42
uv run --no-sync python scripts/train_refiner_kd.py --config configs/distill_refiner_vssd.yaml --mode overfit --ablation A1 --seed 42 --out-dir result/refiner_kd_20261002/overfit_A1_s42
uv run --no-sync python scripts/train_refiner_kd.py --config configs/distill_refiner_vssd.yaml --mode pilot --ablation A0 --seed 42 --out-dir result/refiner_kd_20261002/pilot_A0_s42
```

A1/A2/A3/A4もrunごとに同じ形式で別dirを使う。A0=GTのみ、A1=output KD、A2=+feature、A3=+temporal、A4=+aux。選択candidate名とseedを明記したコマンドを手順24へ追記してからfull/longを実行する。未選択のA4を自動採用しない。resumeは `--resume` にそのrunの `last.pt` の明示pathを渡す仕様。

### 予定export CLI TR-5

```bash
uv run --no-sync python scripts/export_student_refiner_onnx.py --init-config configs/distill_refiner_vssd.yaml --out-dir result/refiner_kd_20261002/export_init
```

学習後は `--init-config` と排他の `--ckpt` に選択済み `best_monitor.pt`、または `--selection-manifest` に手順24.6の選択JSONを渡す。JSONのpath/SHA/configをstrict検証。deploy PT/ONNX/manifestとreportを同dirへ保存する。既存 `scripts/export_refiner_onnx.py` はteacher専用のまま保全。

```bash
uv run --no-sync python scripts/train_refiner_kd.py --config result/refiner_kd_20261002/selected_config.yaml --mode full --seed 42 --out-dir result/refiner_kd_20261002/full_s42
uv run --no-sync python scripts/export_student_refiner_onnx.py --selection-manifest result/refiner_kd_20261002/selected_checkpoint.json --out-dir result/refiner_kd_20261002/deploy
uv run --no-sync python scripts/evaluate_student_refiner.py --config result/refiner_kd_20261002/selected_config.yaml --student-dir result/refiner_kd_20261002/deploy --mode accuracy --provider cpu --out-dir result/refiner_kd_20261002/accuracy
```

これも予定CLI。23.5/24.6の選択前に実行しない。long modeは `--window 16` または32を必須、`--resume` に直前best pathを指定し、lr/stepsをlong configに固定する。window変更はlong modeだけで明示許可し、通常resumeと別runにする。

### 予定evaluate CLI TR-6

teacherはconfigからSHA付きで指定。`--student-dir` は上記export dir、mode別out-dirを必須とする。評価対象はconfigの監査済みmanifestから読み、CLIで勝手にclipを減らさない。

```bash
uv run --no-sync python scripts/evaluate_student_refiner.py --config configs/distill_refiner_vssd.yaml --student-dir result/refiner_kd_20261002/export_init --mode benchmark --provider cpu --fixture-dir result/refiner_kd_20261002/gpu_init --out-dir result/refiner_kd_20261002/benchmark_init_cpu
```

現GPU環境は `result/onnx_gpu_20261002/.venv`。rootでnative teacher/portable student fixtureとNPZ期待出力を準備し、GPU環境はNPZ/ONNXのみreplayする。下記は予定CLIで、rootのCPU onnxruntimeを置換しない。cuDNN/CUDA libは `scripts/cudnn_env.sh` の既存方式、TF32 off。isolated envでnative teacher/DINOを起動しない。

```bash
PYTHONPATH=src uv run --no-project --python result/onnx_gpu_20261002/.venv/bin/python python scripts/evaluate_student_refiner.py --config result/refiner_kd_20261002/selected_config.yaml --student-dir result/refiner_kd_20261002/deploy --mode gpu --provider cuda --out-dir result/refiner_kd_20261002/gpu
```

### 品質管理 TR-7

```bash
uv run --no-sync pytest tests/unit -q
uv run --no-sync ruff check src/mamba3_tracker/model/student_refiner.py src/mamba3_tracker/train/distillation.py src/mamba3_tracker/data/distillation_cache.py src/mamba3_tracker/deployment/student_checkpoint.py scripts/prepare_refiner_kd.py scripts/train_refiner_kd.py scripts/export_student_refiner_onnx.py scripts/evaluate_student_refiner.py
uv run --no-sync ruff format --check src/mamba3_tracker/model/student_refiner.py src/mamba3_tracker/train/distillation.py src/mamba3_tracker/data/distillation_cache.py src/mamba3_tracker/deployment/student_checkpoint.py scripts/prepare_refiner_kd.py scripts/train_refiner_kd.py scripts/export_student_refiner_onnx.py scripts/evaluate_student_refiner.py
uv run --no-sync ty check
git diff --check
```

新規testsとBを実装した場合のrep_temporalもlint/format対象へ追加する。`ty`のincludeは手順15.5で追加済み範囲を確認し、後続moduleも漏らさず追加する。未実装modeのREDを完了扱いにしない。benchmarkは同じONNX hashに対応したfixture manifestを事前準備する。

### 後続実装の依存ゲート TR-2/TR-4/TR-6

- 手順15.1/15.2：window16/32の入力を別namespaceで準備し、追加容量を実測して上限を記録する。window8 cacheの流用、旧cache削除、暗黙のオンライン再計算は禁止。long対応をテストで確認するまで手順24.4以降は実行しない。
- 手順15.2/23.5：現read_configはroot key集合をstrict検証するため、selected_configへ選択metadataを単純追加すると拒否される。選択metadataのschemaと入力生成configの同一性検証を実装し、unknown key拒否を保ったRED→GREENを記録してからfullへ進む。hash検証を削除して回避しない。
- 手順15.4：全F monitor/accuracyの実装、全clip欠落・重複拒否、sequence単位paired bootstrapをテストしてから手順24.6/27へ進む。lossや部分動画をAJの代替にしない。
- 既知のsequence重複は第7章の監査結果を継承する。「clip重複ゼロ」と「完全未見」を区別し、独立性を新たに証明した扱いにしない。

### 固定teacherと既存データ TR-1/TR-2

- public teacher: `weights/mamba3-preview-20261002-best80/tracker_mamba3_daoff_best80.pt`
- SHA256: `c022511872c2d59a39bc1c39a3c62f44c142ee8ff9ca7dc7859881a78ae490b7`
- raw root: `/workspace/vmamba3_data`
- split: `/workspace/vmamba3_data/result/v64_daoff_unused100gb_train_20261002/growing_pool.json`
- evidence: `docs/evidence/daoff_best80_20261002/training_summary_20261002.json`、`run_config_20261002.json`
- old GPU protocol: `docs/onnx_gpu_video_20261002.md`
- 詳細論文・式・設計: `docs/knowledge_distillation_plan_20261002.md`

## 6. 完了の定義

今回の「計画作成」と将来の「モデル完成」を混同しない。

- [x] P-1 / SG-1/TR-1: 専用branch、一次資料リンク、teacher/upstreamサイズ、A/B・aux/loss・cache・ONNX方針の文書が存在する。
- [x] P-2 / SG-6/TR-7: temp作業書の全章とチェックリストをreviewし、Major/Blockerを安全に修正。最終PASS_WITH_NOTES、レビューと残留仮定を別ファイルへ記録済み。
- [ ] D-1 / SG-3/TR-3: 実モデルdeploy params≤650,000、train/deploy countsとbytesを実測、小型目標との比較がある。
- [ ] D-2 / SG-2/TR-2: split/group/teacher-training重複・test可用性監査、同window/input/z_ref、cache容量とhashの証跡がある。
- [ ] D-3 / SG-3/TR-3: teacher freeze/bitwise不変、GT-only対照、aux除去、各loss/mask/NaNのnamed testsがGREEN。
- [ ] D-4 / SG-4/TR-4: smoke/overfitとmonitor-selected full run、seed再現性/long-context検証が完了。statusに終了理由・best/last・残課題がある。
- [ ] D-5 / SG-5/TR-5: deploy PT/ONNX/manifestがstrict再読込可能。fusion数値gate、no teacher/aux/optimizer、opset18標準domain、dynamic8入力4出力。
- [ ] D-6 / SG-5/TR-6: CPU/CUDA数値一致と主要演算の実CUDA profile、track chunk/permutation不変、long F/全不可視の証跡がある。
- [ ] D-7 / SG-5/TR-6: 監査済み全評価集合でteacher/student AJ/CIが事前精度margin合格。速度目標の達否、測定条件とp50/p95は別判定で明記。
- [ ] D-8 / SG-6/TR-7: unit/lint/format/型/diff結果、小型docs証跡と本作業記録が整合。未評価を成功扱いせずfallback無し。

## 7. 作業記録

**重要な注意事項：**

- 作業開始前に必ず `date "+%Y-%m-%d %H:%M:%S %Z%z"` コマンドで現在時刻を確認し、正確な日時を記録します。
- 各作業項目を開始する際と完了する際の両方で記録を行うこと。
- 作業内容は具体的なコマンドや操作手順を詳細に記載すること。
- 結果・備考欄には成功／失敗、エラー内容、解決方法、重要な気づきを必ず記入すること。
- 複数のフェーズがある場合は、フェーズごとに開始・完了の記録を取ること。
- コード変更を行った場合は、変更したファイル名と変更内容の概要を記録すること。
- エラーが発生した場合は、エラーメッセージと解決策を詳細に記録すること。

チェック直後にcommand/exit code/artifact path/SHAを記す。新規runは開始・中断・resume・終了を分け、GPUの使用とcache byte数も記録。token値とbashrc内容は記録しない。

| 日付 | 時刻 | 作業者 | 作業内容 | 結果・備考 |
| :--- | :--- | :--- | :--- | :--- |
| 2026-10-02 | 17:06:02 JST+0900 | Codex | 計画作成開始。main起点・専用branchを確認 | `distill/refiner-kd-onnx-20261002`。モデル実装/訓練は始めない |
| 2026-10-02 | 17:09:42 JST+0900 | Codex | 現行loss/teacher/schemaとwrite/review指示を調査 | V35にvis監督無し。teacherのvis KLは不採用 |
| 2026-10-02 | 17:14:21 JST+0900 | Codex | 計画の数値と環境を確認、文書作成 | safe PT read: 50 tensors/8,117,988、B算術620,548。GPU32607MiB、workspace空105GiB |
| 2026-10-02 | 17:18:36 JST+0900 | Codex | 初稿全体をread、review開始 | 初期REVISE。cache batch context、CLI/ablation原子性、選択lossとB順序を改善対象にした |
| 2026-10-02 | 17:24:24 JST+0900 | Codex | apply_patch不一致を調査 | `Failed to find expected lines`。初回は部分文のcontext、次はhunk順序を修正して再適用。変更欠損・既存file破壊無し |
| 2026-10-02 | 17:27:49 JST+0900 | Codex | 修正後workdocと設計文書を全読、機械検査 | 57手順、4欄/空行/末尾空白問題0、future actions完了0。GPU env uv指定もread-only確認 |
| 2026-10-02 | 17:29:57 JST+0900 | Codex | 追加候補Cの調査とreview収束 | core CPU部品計数＋adapter合算568,984。Mamba-3 2026論文を追記。最終PASS_WITH_NOTES、実装/訓練/ONNX未実施 |
| 2026-10-02 | 17:32:07 JST+0900 | Codex | 文書作成・reviewの最終検証完了 | 57手順、形式/末尾空白エラー0、P-1/P-2のみ完了、Dとfuture actions完了0。git diff --check成功、staged変更無し、tempはignore維持 |

| 2026-10-02 | 17:34:11 JST+0900 | Codex | 実行phase開始、手順1開始 | ユーザーが全DoDまで実行を指示。指定branch、GPU空き、既存計画2ファイルのみ変更あり。active goalを確認 |

| 2026-10-02 | 17:34:38 JST+0900 | Codex | 手順1 操作完了 | HEAD 6223405、指定branch、docs2ファイルの計画変更を保全。競合・実行中訓練無し。 |

| 2026-10-02 | 17:34:38 JST+0900 | Codex | 手順1 確認完了 | HEAD 6223405、指定branch、docs2ファイルの計画変更を保全。競合・実行中訓練無し。 |

| 2026-10-02 | 17:34:38 JST+0900 | Codex | 手順1 テスト完了 | HEAD 6223405、指定branch、docs2ファイルの計画変更を保全。競合・実行中訓練無し。 |

| 2026-10-02 | 17:34:38 JST+0900 | Codex | 手順1 対処完了 | HEAD 6223405、指定branch、docs2ファイルの計画変更を保全。競合・実行中訓練無し。 |

| 2026-10-02 | 17:35:10 JST+0900 | Codex | 手順2開始/進捗 | raw train1027/monitor15は全件存在、minivalは0/150。取得source・group重複を調査開始。 |

### 追加レビュー記録

2026-10-02 17:52:56 JST+0900、学習を長時間走らせる前にexport構造の失敗を検出するため、依存の独立した15.3/19（未学習export）を15.1/15.2より先に実行する順序へ明示変更。学習済みartifactの25/26/27は従来順を維持。14のdynamic/CLI/profile testsは各機能の実装前に追加してREDを確認し、全て揃うまで14は未チェックとする。入力tensorの再利用cacheはteacher label cacheとは別であり、新規評価領域内20GB上限で準備する予定。旧cacheのevictionは禁止。

2026-10-02 17:39:23 JST+0900、手順2再開。既存取得scriptのCRC検証・容量上限・partial削除範囲を確認。新規領域 `/workspace/vmamba3_eval/raw` のみを対象とし、旧train manifestを保全するwrapperをtempへ追加。手順2.1は取得待ち中に設計・実装を並行する明示的な例外とする。

2026-10-02 17:38:42 JST+0900、write/reviewスキルと参照テンプレート・rubricを再確認。専用branchと57手順、P/D分離を確認した。チェックリスト冒頭の「全て未実行」を現状に合わせて修正し、window32の学習率をshort基準9e-5と明記。学習・ONNX・AJの完了を新たに主張する変更は行っていない。

| 2026-10-02 | 17:41:16 JST+0900 | Codex | 手順2 操作完了 | inventory.json: train1027/monitor15全件。clip重複0、保守的sequence群でmonitor14/15とminival141/150がtrainと重複。minival欠落取得は2.1へ分離、完全未見とは呼ばず全150公式比較を維持。 |

| 2026-10-02 | 17:41:16 JST+0900 | Codex | 手順2 確認完了 | inventory.json: train1027/monitor15全件。clip重複0、保守的sequence群でmonitor14/15とminival141/150がtrainと重複。minival欠落取得は2.1へ分離、完全未見とは呼ばず全150公式比較を維持。 |

| 2026-10-02 | 17:41:16 JST+0900 | Codex | 手順2 テスト完了 | inventory.json: train1027/monitor15全件。clip重複0、保守的sequence群でmonitor14/15とminival141/150がtrainと重複。minival欠落取得は2.1へ分離、完全未見とは呼ばず全150公式比較を維持。 |

| 2026-10-02 | 17:41:16 JST+0900 | Codex | 手順2 対処完了 | inventory.json: train1027/monitor15全件。clip重複0、保守的sequence群でmonitor14/15とminival141/150がtrainと重複。minival欠落取得は2.1へ分離、完全未見とは呼ばず全150公式比較を維持。 |

| 2026-10-02 | 17:41:16 JST+0900 | Codex | 手順3 操作完了 | 設計文書全204行を再読。一次資料9件と現行ソース、A620502/teacher8117988、vis未監督、B条件付き。論文結果と適用提案を区別。 |

| 2026-10-02 | 17:41:16 JST+0900 | Codex | 手順3 確認完了 | 設計文書全204行を再読。一次資料9件と現行ソース、A620502/teacher8117988、vis未監督、B条件付き。論文結果と適用提案を区別。 |

| 2026-10-02 | 17:41:16 JST+0900 | Codex | 手順3 テスト完了 | 設計文書全204行を再読。一次資料9件と現行ソース、A620502/teacher8117988、vis未監督、B条件付き。論文結果と適用提案を区別。 |

| 2026-10-02 | 17:41:16 JST+0900 | Codex | 手順3 対処完了 | 設計文書全204行を再読。一次資料9件と現行ソース、A620502/teacher8117988、vis未監督、B条件付き。論文結果と適用提案を区別。 |

| 2026-10-02 | 17:41:16 JST+0900 | Codex | 手順4 操作完了 | 既存計画のonline既定を確定。正確なmicrobatch z_refを共有、label cacheは任意上限20GB/空き10GB、全Fラベルの切出し不可。cache_budgetは次のprotocolと保存。 |

| 2026-10-02 | 17:41:16 JST+0900 | Codex | 手順4 確認完了 | 既存計画のonline既定を確定。正確なmicrobatch z_refを共有、label cacheは任意上限20GB/空き10GB、全Fラベルの切出し不可。cache_budgetは次のprotocolと保存。 |

| 2026-10-02 | 17:41:16 JST+0900 | Codex | 手順4 テスト完了 | 既存計画のonline既定を確定。正確なmicrobatch z_refを共有、label cacheは任意上限20GB/空き10GB、全Fラベルの切出し不可。cache_budgetは次のprotocolと保存。 |

| 2026-10-02 | 17:41:16 JST+0900 | Codex | 手順4 対処完了 | 既存計画のonline既定を確定。正確なmicrobatch z_refを共有、label cacheは任意上限20GB/空き10GB、全Fラベルの切出し不可。cache_budgetは次のprotocolと保存。 |

| 2026-10-02 | 17:42:41 JST+0900 | Codex | 手順5 操作完了 | protocol.json事前固定 SHA e83a01247263c84c27273427578b8c8093b48c6a6de42b514dbe12d949941054。全150、既知sequence重複scope、AJ/CIと数値/サイズgate・seeds・対照を維持。cache_budget.json保存。 |

| 2026-10-02 | 17:42:41 JST+0900 | Codex | 手順5 確認完了 | protocol.json事前固定 SHA e83a01247263c84c27273427578b8c8093b48c6a6de42b514dbe12d949941054。全150、既知sequence重複scope、AJ/CIと数値/サイズgate・seeds・対照を維持。cache_budget.json保存。 |

| 2026-10-02 | 17:42:41 JST+0900 | Codex | 手順5 テスト完了 | protocol.json事前固定 SHA e83a01247263c84c27273427578b8c8093b48c6a6de42b514dbe12d949941054。全150、既知sequence重複scope、AJ/CIと数値/サイズgate・seeds・対照を維持。cache_budget.json保存。 |

| 2026-10-02 | 17:42:41 JST+0900 | Codex | 手順5 対処完了 | protocol.json事前固定 SHA e83a01247263c84c27273427578b8c8093b48c6a6de42b514dbe12d949941054。全150、既知sequence重複scope、AJ/CIと数値/サイズgate・seeds・対照を維持。cache_budget.json保存。 |

| 2026-10-02 | 17:42:41 JST+0900 | Codex | 手順6開始/進捗 | named testsを先行追加。student geometry/size/freeze/aux/masks/confidence/contextをREDで確認する。 |

| 2026-10-02 | 17:42:41 JST+0900 | Codex | 手順6 操作完了 | pytest:11件が新規student/distillation module未存在で意図したRED (exit1)。fixtureは外部データ不要、全mask/confidence/occlusion/context/freezeを含む。 |

| 2026-10-02 | 17:42:41 JST+0900 | Codex | 手順6 確認完了 | pytest:11件が新規student/distillation module未存在で意図したRED (exit1)。fixtureは外部データ不要、全mask/confidence/occlusion/context/freezeを含む。 |

| 2026-10-02 | 17:42:41 JST+0900 | Codex | 手順6 テスト完了 | pytest:11件が新規student/distillation module未存在で意図したRED (exit1)。fixtureは外部データ不要、全mask/confidence/occlusion/context/freezeを含む。 |

| 2026-10-02 | 17:42:41 JST+0900 | Codex | 手順6 対処完了 | pytest:11件が新規student/distillation module未存在で意図したRED (exit1)。fixtureは外部データ不要、全mask/confidence/occlusion/context/freezeを含む。 |

| 2026-10-02 | 17:42:41 JST+0900 | Codex | 手順7開始/進捗 | 共通geometryを既存portableから再利用し、明示的feature tapsとVSSD二poolを実装する。旧teacher回帰も対象。 |

| 2026-10-02 | 17:46:01 JST+0900 | Codex | 手順7 操作完了 | student620502とaux/backbone無し、4出力、同batch z_ref、feature taps/gradient4tests GREEN。q_projのweight参照を実構造proj.weightへ修正。geometry共通化の旧回帰は手順17でも検証。 |

| 2026-10-02 | 17:46:01 JST+0900 | Codex | 手順7 確認完了 | student620502とaux/backbone無し、4出力、同batch z_ref、feature taps/gradient4tests GREEN。q_projのweight参照を実構造proj.weightへ修正。geometry共通化の旧回帰は手順17でも検証。 |

| 2026-10-02 | 17:46:01 JST+0900 | Codex | 手順7 テスト完了 | student620502とaux/backbone無し、4出力、同batch z_ref、feature taps/gradient4tests GREEN。q_projのweight参照を実構造proj.weightへ修正。geometry共通化の旧回帰は手順17でも検証。 |

| 2026-10-02 | 17:46:01 JST+0900 | Codex | 手順7 対処完了 | student620502とaux/backbone無し、4出力、同batch z_ref、feature taps/gradient4tests GREEN。q_projのweight参照を実構造proj.weightへ修正。geometry共通化の旧回帰は手順17でも検証。 |

| 2026-10-02 | 17:46:01 JST+0900 | Codex | 手順8 操作完了 | cache5testsは未実装moduleでREDを確認。context・batch・zref変更、split拒否、破損hash、容量超過。 |

| 2026-10-02 | 17:46:01 JST+0900 | Codex | 手順8 確認完了 | cache5testsは未実装moduleでREDを確認。context・batch・zref変更、split拒否、破損hash、容量超過。 |

| 2026-10-02 | 17:46:01 JST+0900 | Codex | 手順8 テスト完了 | cache5testsは未実装moduleでREDを確認。context・batch・zref変更、split拒否、破損hash、容量超過。 |

| 2026-10-02 | 17:46:01 JST+0900 | Codex | 手順8 対処完了 | cache5testsは未実装moduleでREDを確認。context・batch・zref変更、split拒否、破損hash、容量超過。 |

| 2026-10-02 | 17:46:01 JST+0900 | Codex | 手順9開始/進捗 | cache実装を追加、単一writer・atomic commit marker・no eviction・strict train-only/hash/context。 |

| 2026-10-02 | 17:49:48 JST+0900 | Codex | 手順9 操作完了 | cache5tests GREEN。context/batch/FP32z_ref hash、CRC代わりpayloadSHA、strict train-only/容量/atomic完了markerを検証。既存cacheは変更無し。 |

| 2026-10-02 | 17:49:48 JST+0900 | Codex | 手順9 確認完了 | cache5tests GREEN。context/batch/FP32z_ref hash、CRC代わりpayloadSHA、strict train-only/容量/atomic完了markerを検証。既存cacheは変更無し。 |

| 2026-10-02 | 17:49:48 JST+0900 | Codex | 手順9 テスト完了 | cache5tests GREEN。context/batch/FP32z_ref hash、CRC代わりpayloadSHA、strict train-only/容量/atomic完了markerを検証。既存cacheは変更無し。 |

| 2026-10-02 | 17:49:48 JST+0900 | Codex | 手順9 対処完了 | cache5tests GREEN。context/batch/FP32z_ref hash、CRC代わりpayloadSHA、strict train-only/容量/atomic完了markerを検証。既存cacheは変更無し。 |

| 2026-10-02 | 17:49:48 JST+0900 | Codex | 手順10 操作完了 | wrapper/loss実装。計16tests GREEN、teacher freeze/eval/stopgrad bitwise、vis未蒸留、全mask0、NaN拒否、confidence分母・occlusion不可視監督。 |

| 2026-10-02 | 17:49:48 JST+0900 | Codex | 手順10 確認完了 | wrapper/loss実装。計16tests GREEN、teacher freeze/eval/stopgrad bitwise、vis未蒸留、全mask0、NaN拒否、confidence分母・occlusion不可視監督。 |

| 2026-10-02 | 17:49:48 JST+0900 | Codex | 手順10 テスト完了 | wrapper/loss実装。計16tests GREEN、teacher freeze/eval/stopgrad bitwise、vis未蒸留、全mask0、NaN拒否、confidence分母・occlusion不可視監督。 |

| 2026-10-02 | 17:49:48 JST+0900 | Codex | 手順10 対処完了 | wrapper/loss実装。計16tests GREEN、teacher freeze/eval/stopgrad bitwise、vis未蒸留、全mask0、NaN拒否、confidence分母・occlusion不可視監督。 |

| 2026-10-02 | 17:49:48 JST+0900 | Codex | 手順11開始/進捗 | 固定protocolに沿った新規A設定を追加する。 |

| 2026-10-02 | 17:52:56 JST+0900 | Codex | 手順11 操作完了 | 専用config追加。teacher/split/protocol SHA、DAoff、8frameとDINO pin、online同z_ref、予算optimizer固定。ablationflags test GREEN。 |

| 2026-10-02 | 17:52:56 JST+0900 | Codex | 手順11 確認完了 | 専用config追加。teacher/split/protocol SHA、DAoff、8frameとDINO pin、online同z_ref、予算optimizer固定。ablationflags test GREEN。 |

| 2026-10-02 | 17:52:56 JST+0900 | Codex | 手順11 テスト完了 | 専用config追加。teacher/split/protocol SHA、DAoff、8frameとDINO pin、online同z_ref、予算optimizer固定。ablationflags test GREEN。 |

| 2026-10-02 | 17:52:56 JST+0900 | Codex | 手順11 対処完了 | 専用config追加。teacher/split/protocol SHA、DAoff、8frameとDINO pin、online同z_ref、予算optimizer固定。ablationflags test GREEN。 |

| 2026-10-02 | 17:52:56 JST+0900 | Codex | 手順12開始/進捗 | 条件付きBはA pilot採否まで保留。13/13.1/13.2も未着手。 |

| 2026-10-02 | 17:52:56 JST+0900 | Codex | 手順14開始/進捗 | schema/aux/hash/workspace3tests RED、既存teacherstrict/trackchunk2tests GREEN。dynamic/profile/exportCLI testsは該当CLI実装前に追加するためチェック未完。 |

| 2026-10-02 | 17:52:56 JST+0900 | Codex | 手順15開始/進捗 | strict student loader/save追加、public5tests GREEN。native外部特徴teacher bridgeはCUDA parity前で未検証。 |

| 2026-10-02 | 17:57:12 JST+0900 | Codex | 手順15.3 操作完了 | studentexportCLI排他sources/hash/strictreload、opset18標準op、8入力4出力。dynamic/CLI RED→GREEN確認。新規deploy tests8passed。 |

| 2026-10-02 | 17:57:12 JST+0900 | Codex | 手順15.3 確認完了 | studentexportCLI排他sources/hash/strictreload、opset18標準op、8入力4出力。dynamic/CLI RED→GREEN確認。新規deploy tests8passed。 |

| 2026-10-02 | 17:57:12 JST+0900 | Codex | 手順15.3 テスト完了 | studentexportCLI排他sources/hash/strictreload、opset18標準op、8入力4出力。dynamic/CLI RED→GREEN確認。新規deploy tests8passed。 |

| 2026-10-02 | 17:57:12 JST+0900 | Codex | 手順15.3 対処完了 | studentexportCLI排他sources/hash/strictreload、opset18標準op、8入力4出力。dynamic/CLI RED→GREEN確認。新規deploy tests8passed。 |

| 2026-10-02 | 17:57:12 JST+0900 | Codex | 手順19 操作完了 | export_init成功: 620502params、ONNX2534327bytes/PT2509593bytes。F1/8/31/128/257/300/600、B1/2、N1/7/32、depth17x21、全不可視、F600N900chunk32の9検査pass。精度/CUDAは未評価。 |

| 2026-10-02 | 17:57:12 JST+0900 | Codex | 手順19 確認完了 | export_init成功: 620502params、ONNX2534327bytes/PT2509593bytes。F1/8/31/128/257/300/600、B1/2、N1/7/32、depth17x21、全不可視、F600N900chunk32の9検査pass。精度/CUDAは未評価。 |

| 2026-10-02 | 17:57:12 JST+0900 | Codex | 手順19 テスト完了 | export_init成功: 620502params、ONNX2534327bytes/PT2509593bytes。F1/8/31/128/257/300/600、B1/2、N1/7/32、depth17x21、全不可視、F600N900chunk32の9検査pass。精度/CUDAは未評価。 |

| 2026-10-02 | 17:57:12 JST+0900 | Codex | 手順19 対処完了 | export_init成功: 620502params、ONNX2534327bytes/PT2509593bytes。F1/8/31/128/257/300/600、B1/2、N1/7/32、depth17x21、全不可視、F600N900chunk32の9検査pass。精度/CUDAは未評価。 |

| 2026-10-02 | 18:04:20 JST+0900 | Codex | 手順15.1開始/進捗 | prepareCLI追加。minivalをpilotへ指定するRED→GREEN。旧frozen cacheをmtimeも変更せずread-onlyで再利用、missは明示失敗、raw/depth/hash、clip固定queryrng、batch zref再計算用7inputs保存。 |

| 2026-10-02 | 18:04:20 JST+0900 | Codex | 手順20開始/進捗 | train10window native GPUteacher pilot成功、repeat bitwise一致・teacher不変。初回compile22.17s、以後1.58–1.66s/window。CPU返却flow UVをCUDAへ明示移動する修正後に成功。labelcacheは非採用online、入力cache再利用は別namespace。 |

| 2026-10-02 | 18:04:20 JST+0900 | Codex | 手順15.1開始/進捗 | 全train入力準備session53178、monitor入力準備99472を開始。旧cache readonly、inputcache20GB・disk空30GBgate。minivaldownload82710 live、85/150取得。 |

| 2026-10-02 | 18:05:26 JST+0900 | Codex | 手順15.1開始/進捗 | 行動40到達、カウントを0へreset。入力準備53178/99472とdownload82710の進捗を追跡。student本体・KD/cache・strictdeploy・CPU ONNX長系列まで進行、本訓練とCUDA/全AJは未完。 |

| 2026-10-02 | 18:12:59 JST+0900 | Codex | 手順15.5開始/進捗 | 新規student/KD/cache/CLI/testsをty対象追加。実Moduleの呼出型/JSON混在型を修正し全ty成功、除外無し。 |

| 2026-10-02 | 18:12:59 JST+0900 | Codex | 手順21開始/進捗 | 20-step GPU smoke_A1_s42開始。27新規unitとty成功、未学習ONNX9cases・native教師原型4出力bitwise一致の前提確認済み。microbatch候補1/4/8/16/32を計測する。 |

| 2026-10-02 | 18:14:12 JST+0900 | Codex | 手順21開始/進捗 | smoke20step完了、teacherunchanged/student更新、lossgradfinite、GTmonitor0.510047→0.485234、640clips、16.09s。最大14tracksfixtureでmicrobatch32 safe、peak938892800bytes。全データ最大Nのprofileはpilotで再実測。term別grad診断は追加中のため未チェック。 |

| 2026-10-02 | 18:14:12 JST+0900 | Codex | 手順21.1開始/進捗 | step10から別dirへresume開始 session74968。identity teacher/split/config/input/seed/microbatch/RNGを復元。 |

| 2026-10-02 | 18:19:39 JST+0900 | Codex | 手順21.1 操作完了 | step10→20 resumeと元step20の全trainable tensors差0。smoke_resume_parity.json成功、同一RNG/optimizer/teacher/split/context。 |

| 2026-10-02 | 18:19:39 JST+0900 | Codex | 手順21.1 確認完了 | step10→20 resumeと元step20の全trainable tensors差0。smoke_resume_parity.json成功、同一RNG/optimizer/teacher/split/context。 |

| 2026-10-02 | 18:19:39 JST+0900 | Codex | 手順21.1 テスト完了 | step10→20 resumeと元step20の全trainable tensors差0。smoke_resume_parity.json成功、同一RNG/optimizer/teacher/split/context。 |

| 2026-10-02 | 18:19:39 JST+0900 | Codex | 手順21.1 対処完了 | step10→20 resumeと元step20の全trainable tensors差0。smoke_resume_parity.json成功、同一RNG/optimizer/teacher/split/context。 |

| 2026-10-02 | 18:19:39 JST+0900 | Codex | 手順22 操作完了 | 32step/1024clips overfit成功。固定4窓GT .21962267→.03428874、residualKD .00208684→.00080565、XYZKD .00210795→.00045632。項別grad finite、clamp率0、teacher不変。汎化・全AJの証明ではない。 |

| 2026-10-02 | 18:19:39 JST+0900 | Codex | 手順22 確認完了 | 32step/1024clips overfit成功。固定4窓GT .21962267→.03428874、residualKD .00208684→.00080565、XYZKD .00210795→.00045632。項別grad finite、clamp率0、teacher不変。汎化・全AJの証明ではない。 |

| 2026-10-02 | 18:19:39 JST+0900 | Codex | 手順22 テスト完了 | 32step/1024clips overfit成功。固定4窓GT .21962267→.03428874、residualKD .00208684→.00080565、XYZKD .00210795→.00045632。項別grad finite、clamp率0、teacher不変。汎化・全AJの証明ではない。 |

| 2026-10-02 | 18:19:39 JST+0900 | Codex | 手順22 対処完了 | 32step/1024clips overfit成功。固定4窓GT .21962267→.03428874、residualKD .00208684→.00080565、XYZKD .00210795→.00045632。項別grad finite、clamp率0、teacher不変。汎化・全AJの証明ではない。 |

| 2026-10-02 | 18:19:39 JST+0900 | Codex | 手順15.4開始/進捗 | GPU/benchmark CLIと欠落・重複拒否/warmup除外 tests RED→GREEN。monitor/accuracy全F実装は明示未完、例外で停止。GPU fixture7shape準備済み、隔離ORT環境で実機検証開始。 |

| 2026-10-02 | 18:52:04 JST+0900 | Codex | 手順16 操作完了 | 新規testsおよびpaired gateのGREEN。2026-10-02全unit189件成功、追加schema/partial tests12件成功。Bは条件付き未採用。 |

| 2026-10-02 | 18:52:04 JST+0900 | Codex | 手順16 確認完了 | 新規testsおよびpaired gateのGREEN。2026-10-02全unit189件成功、追加schema/partial tests12件成功。Bは条件付き未採用。 |

| 2026-10-02 | 18:52:04 JST+0900 | Codex | 手順16 テスト完了 | 新規testsおよびpaired gateのGREEN。2026-10-02全unit189件成功、追加schema/partial tests12件成功。Bは条件付き未採用。 |

| 2026-10-02 | 18:52:04 JST+0900 | Codex | 手順16 対処完了 | 新規testsおよびpaired gateのGREEN。2026-10-02全unit189件成功、追加schema/partial tests12件成功。Bは条件付き未採用。 |

| 2026-10-02 | 18:52:04 JST+0900 | Codex | 手順17 操作完了 | uv run --no-sync pytest tests/unit -q:189passed、29warnings、14.60s。旧teacher/ONNX回帰込み。 |

| 2026-10-02 | 18:52:04 JST+0900 | Codex | 手順17 確認完了 | uv run --no-sync pytest tests/unit -q:189passed、29warnings、14.60s。旧teacher/ONNX回帰込み。 |

| 2026-10-02 | 18:52:04 JST+0900 | Codex | 手順17 テスト完了 | uv run --no-sync pytest tests/unit -q:189passed、29warnings、14.60s。旧teacher/ONNX回帰込み。 |

| 2026-10-02 | 18:52:04 JST+0900 | Codex | 手順17 対処完了 | uv run --no-sync pytest tests/unit -q:189passed、29warnings、14.60s。旧teacher/ONNX回帰込み。 |

| 2026-10-02 | 18:52:04 JST+0900 | Codex | 手順23 操作完了 | pilot_A0_s42完了、200step/monitorGT0.1212941284、同初期tensor・input・microbatch32。pilot_chain_status.json参照。 |

| 2026-10-02 | 18:52:04 JST+0900 | Codex | 手順23 確認完了 | pilot_A0_s42完了、200step/monitorGT0.1212941284、同初期tensor・input・microbatch32。pilot_chain_status.json参照。 |

| 2026-10-02 | 18:52:04 JST+0900 | Codex | 手順23 テスト完了 | pilot_A0_s42完了、200step/monitorGT0.1212941284、同初期tensor・input・microbatch32。pilot_chain_status.json参照。 |

| 2026-10-02 | 18:52:04 JST+0900 | Codex | 手順23 対処完了 | pilot_A0_s42完了、200step/monitorGT0.1212941284、同初期tensor・input・microbatch32。pilot_chain_status.json参照。 |

| 2026-10-02 | 18:52:04 JST+0900 | Codex | 手順23.1 操作完了 | pilot_A1_s42完了、200step/monitorGT0.1156595416、best190。全identityの比較条件一致。 |

| 2026-10-02 | 18:52:04 JST+0900 | Codex | 手順23.1 確認完了 | pilot_A1_s42完了、200step/monitorGT0.1156595416、best190。全identityの比較条件一致。 |

| 2026-10-02 | 18:52:04 JST+0900 | Codex | 手順23.1 テスト完了 | pilot_A1_s42完了、200step/monitorGT0.1156595416、best190。全identityの比較条件一致。 |

| 2026-10-02 | 18:52:04 JST+0900 | Codex | 手順23.1 対処完了 | pilot_A1_s42完了、200step/monitorGT0.1156595416、best190。全identityの比較条件一致。 |

| 2026-10-02 | 18:52:04 JST+0900 | Codex | 手順23.2 操作完了 | pilot_A2_s42完了、200step/monitorGT0.1245791333、best150。A1より悪化。 |

| 2026-10-02 | 18:52:04 JST+0900 | Codex | 手順23.2 確認完了 | pilot_A2_s42完了、200step/monitorGT0.1245791333、best150。A1より悪化。 |

| 2026-10-02 | 18:52:04 JST+0900 | Codex | 手順23.2 テスト完了 | pilot_A2_s42完了、200step/monitorGT0.1245791333、best150。A1より悪化。 |

| 2026-10-02 | 18:52:04 JST+0900 | Codex | 手順23.2 対処完了 | pilot_A2_s42完了、200step/monitorGT0.1245791333、best150。A1より悪化。 |

| 2026-10-02 | 18:52:04 JST+0900 | Codex | 手順23.3 操作完了 | pilot_A3_s42完了、200step/monitorGT0.1237190531、best150。A1より悪化。 |

| 2026-10-02 | 18:52:04 JST+0900 | Codex | 手順23.3 確認完了 | pilot_A3_s42完了、200step/monitorGT0.1237190531、best150。A1より悪化。 |

| 2026-10-02 | 18:52:04 JST+0900 | Codex | 手順23.3 テスト完了 | pilot_A3_s42完了、200step/monitorGT0.1237190531、best150。A1より悪化。 |

| 2026-10-02 | 18:52:04 JST+0900 | Codex | 手順23.3 対処完了 | pilot_A3_s42完了、200step/monitorGT0.1237190531、best150。A1より悪化。 |

| 2026-10-02 | 18:52:04 JST+0900 | Codex | 手順23.4 操作完了 | pilot_A4_s42完了、200step/monitorGT0.1288348157、best150。補助headが改善するとは主張しない。 |

| 2026-10-02 | 18:52:04 JST+0900 | Codex | 手順23.4 確認完了 | pilot_A4_s42完了、200step/monitorGT0.1288348157、best150。補助headが改善するとは主張しない。 |

| 2026-10-02 | 18:52:04 JST+0900 | Codex | 手順23.4 テスト完了 | pilot_A4_s42完了、200step/monitorGT0.1288348157、best150。補助headが改善するとは主張しない。 |

| 2026-10-02 | 18:52:04 JST+0900 | Codex | 手順23.4 対処完了 | pilot_A4_s42完了、200step/monitorGT0.1288348157、best150。補助headが改善するとは主張しない。 |

| 2026-10-02 | 18:52:04 JST+0900 | Codex | 手順23.5 操作完了 | selected_config.yamlでA1採用、base/pilotSHA固定、schema testGREEN。GTonlyより良い。Bは追加しない。最終AJ・latency未判定。 |

| 2026-10-02 | 18:52:04 JST+0900 | Codex | 手順23.5 確認完了 | selected_config.yamlでA1採用、base/pilotSHA固定、schema testGREEN。GTonlyより良い。Bは追加しない。最終AJ・latency未判定。 |

| 2026-10-02 | 18:52:04 JST+0900 | Codex | 手順23.5 テスト完了 | selected_config.yamlでA1採用、base/pilotSHA固定、schema testGREEN。GTonlyより良い。Bは追加しない。最終AJ・latency未判定。 |

| 2026-10-02 | 18:52:04 JST+0900 | Codex | 手順23.5 対処完了 | selected_config.yamlでA1採用、base/pilotSHA固定、schema testGREEN。GTonlyより良い。Bは追加しない。最終AJ・latency未判定。 |

| 2026-10-02 | 18:52:04 JST+0900 | Codex | 手順24 操作完了 | full_s42完了:early_stop240、7680clips、best190/0.1156595416、teacherunchanged、peak5483915776B、146.31s。全F精度とlongは後続。 |

| 2026-10-02 | 18:52:04 JST+0900 | Codex | 手順24 確認完了 | full_s42完了:early_stop240、7680clips、best190/0.1156595416、teacherunchanged、peak5483915776B、146.31s。全F精度とlongは後続。 |

| 2026-10-02 | 18:52:04 JST+0900 | Codex | 手順24 テスト完了 | full_s42完了:early_stop240、7680clips、best190/0.1156595416、teacherunchanged、peak5483915776B、146.31s。全F精度とlongは後続。 |

| 2026-10-02 | 18:52:04 JST+0900 | Codex | 手順24 対処完了 | full_s42完了:early_stop240、7680clips、best190/0.1156595416、teacherunchanged、peak5483915776B、146.31s。全F精度とlongは後続。 |

| 2026-10-02 | 18:55:55 JST+0900 | Codex | 手順24.1 操作完了 | full_control_s42完了、early_stop300、best250/GTmonitor0.1138488412。KD full240steps/best0.1156595416より対照が良い。上限・初期化・data・optimizerは同条件、実学習step差はearlystopによる。KD最終効果は未証明。 |

| 2026-10-02 | 18:55:55 JST+0900 | Codex | 手順24.1 確認完了 | full_control_s42完了、early_stop300、best250/GTmonitor0.1138488412。KD full240steps/best0.1156595416より対照が良い。上限・初期化・data・optimizerは同条件、実学習step差はearlystopによる。KD最終効果は未証明。 |

| 2026-10-02 | 18:55:55 JST+0900 | Codex | 手順24.1 テスト完了 | full_control_s42完了、early_stop300、best250/GTmonitor0.1138488412。KD full240steps/best0.1156595416より対照が良い。上限・初期化・data・optimizerは同条件、実学習step差はearlystopによる。KD最終効果は未証明。 |

| 2026-10-02 | 18:55:55 JST+0900 | Codex | 手順24.1 対処完了 | full_control_s42完了、early_stop300、best250/GTmonitor0.1138488412。KD full240steps/best0.1156595416より対照が良い。上限・初期化・data・optimizerは同条件、実学習step差はearlystopによる。KD最終効果は未証明。 |

| 2026-10-02 | 19:00:09 JST+0900 | Codex | 手順24.2開始/進捗 | full_s43完了270step、best220/GTmonitor0.1147727948、status complete。seed以外同selected config。全F AJは後続一括評価。 |

| 2026-10-02 | 19:08:32 JST+0900 | Codex | 手順24.2 操作完了 | seed43全F monitor15完了: macro3DAJ .1058393695/metricAJ .2437281765。best220固定、teacher同入力 .148993259/.268132877。seed42を置換せず再現性として報告。 |

| 2026-10-02 | 19:08:32 JST+0900 | Codex | 手順24.2 確認完了 | seed43全F monitor15完了: macro3DAJ .1058393695/metricAJ .2437281765。best220固定、teacher同入力 .148993259/.268132877。seed42を置換せず再現性として報告。 |

| 2026-10-02 | 19:08:32 JST+0900 | Codex | 手順24.2 テスト完了 | seed43全F monitor15完了: macro3DAJ .1058393695/metricAJ .2437281765。best220固定、teacher同入力 .148993259/.268132877。seed42を置換せず再現性として報告。 |

| 2026-10-02 | 19:08:32 JST+0900 | Codex | 手順24.2 対処完了 | seed43全F monitor15完了: macro3DAJ .1058393695/metricAJ .2437281765。best220固定、teacher同入力 .148993259/.268132877。seed42を置換せず再現性として報告。 |

| 2026-10-02 | 19:08:32 JST+0900 | Codex | 手順24.3 操作完了 | seed44full270step earlystop、best220/GT .1097330624。全F monitor macro3DAJ .1063759005/metricAJ .2411640121。seed42/43/44の全F平均metricAJ .23604185、範囲 .22323336〜.24372818。最終選択seed42方針維持。 |

| 2026-10-02 | 19:08:32 JST+0900 | Codex | 手順24.3 確認完了 | seed44full270step earlystop、best220/GT .1097330624。全F monitor macro3DAJ .1063759005/metricAJ .2411640121。seed42/43/44の全F平均metricAJ .23604185、範囲 .22323336〜.24372818。最終選択seed42方針維持。 |

| 2026-10-02 | 19:08:32 JST+0900 | Codex | 手順24.3 テスト完了 | seed44full270step earlystop、best220/GT .1097330624。全F monitor macro3DAJ .1063759005/metricAJ .2411640121。seed42/43/44の全F平均metricAJ .23604185、範囲 .22323336〜.24372818。最終選択seed42方針維持。 |

| 2026-10-02 | 19:08:32 JST+0900 | Codex | 手順24.3 対処完了 | seed44full270step earlystop、best220/GT .1097330624。全F monitor macro3DAJ .1063759005/metricAJ .2411640121。seed42/43/44の全F平均metricAJ .23604185、範囲 .22323336〜.24372818。最終選択seed42方針維持。 |

## 8. レビュー記録

2026-10-02 19:06:01 JST+0900、行動40到達で定型reminderを表示しカウント0へreset。前回以降はmetrics/selected-config/long入力・訓練/全F評価/追加GPUfixtureを実装、194unit GREEN、追加GPU形状test2件GREEN、最新ty成功。full A1 seed42/43/44は240/270/270stepでearlystop、best GT monitor=.115659542/.114772795/.109733062、teacherは不変。対照A0は300step/best.113848841。monitor全FのA0 macro 3D-AJ=.101076730/metric-AJ=.227728526、A1 seed42=.102392671/.223233361、teacher=.148993259/.268132877で精度不足。seed43全F評価52339とshort学習済みONNX export24615を開始、最終採用ではない。long入力session6124/PID2756593は16frame train83/1027時点、minival共通入力72060/PID2758514は3本時点、download2550009/depth2565298もliveを確認。minival準備はavailable-only明示で未取得をpartial扱い、全150揃うまで評価不可。既存ジョブの再起動はしない。最終選択、long訓練、全150AJ、最終GPU/速度、D1〜D8完了監査は未達。

2026-10-02 19:00:09 JST+0900、monitor全15clipの共通前処理完成（5,265,670,867 bytes）。monitor_full_s42/accuracy_report.jsonは全F/N完了、teacher macro 3D-AJ=.148993259/metric-AJ=.268132877、short A1=.102392671/.223233361。短窓studentに明確な劣化があり、精度同等未達。予定のlong16/32入力生成をsession6124で開始（train→validation、16→32の順）。旧cacheは保全。2clip preflightは初回scripts import path不足で停止し、repo/scriptsの明示path追加後に4.05s/3.28sで成功。全unit194passed/29warnings/14.51s、ty/ruff成功。GT-only全F評価session83966、seed44本訓練を開始。精度基準は緩和しない。

2026-10-02 18:53:12〜18:55:55 JST+0900、手順15.1/15.2のlong対応を実装。long入力の明示compute/root必須と旧cache書込拒否test、long訓練の短window拒否testを先にRED→GREEN、関連27tests/ty/ruff成功。long準備は `--window 16`（後に32）`--compute-frontend --new-input-root result/refiner_kd_20261002/long_inputs --new-input-budget-gb 150` を明示し、base configを使う。旧8frame cacheを変更せず、WAFT固定hash/設定とBF16 DINOを再計算。Desktop追加150GB上限、空100GB以上。train1027/validation15の両方を準備する。long phase移行は通常resumeと分離し、`--initialize-from` に直前phaseのpublic best、`--initialize-sha256` に実SHAを指定してoptimizerをresetする仕様へ更新。`--resume` はlong内部の中断復帰だけに使用し、同じinitialization指定も必須。lr上限9e-5、100step warmup、200step上限、window対応monitor GT lossでphase内best、全F monitor AJでphase間選択。実行前に各manifest window一致を検証する。

2026-10-02 18:44:50 JST+0900以降、手順15.4の全F評価経路を実装中。新規 `prepare_student_evaluation.py` は既存eval_metric3d._inferを使ってWAFT/DA3/DINOを一度計算し、refiner未実行の共通8入力を保存する。monitor15本→選択後minival150本の順。追加入力保存先はDesktop側 `result/refiner_kd_20261002/evaluation_inputs/` とし、partitionごと200GB上限・空100GB維持、workspace既存60GB制限は変更しない（18:44観測Desktop空461GB）。精度試験中も全F/Nとscalar z_refを固定、Nだけchunk化。前処理はFP32 DINO、native teacher mixer BF16/geometry FP32、student FP32を明記。短window訓練のBF16 DINOとは精度設定が異なるが、比較するteacher/student入力は完全共有。全F評価CLIとpreparation追加のテストは実装直後に追加したため先行RED無し、最終gateは実データ実行で別途確認する。

2026-10-02 18:42:35 JST+0900、active goalのDoD完遂指示により実行を継続。前turnは文書改善のprogress。download PID2550009/depth PID2565298の稼働を確認、pilot chainは完了。18:43:55より手順15.4のpaired sequence bootstrapを実装、先行RED3件→GREEN。全unit189件成功（14.60s）。A0/A1/A2/A3/A4のmonitor GT lossは0.121294/0.115660/0.124579/0.123719/0.128835、全200step、初期tensor・input・microbatch32等identity一致。A1採用、Bは現時点追加しない。18:44:50より15.2/23.5のselected_config provenance対応を実装し、base config全内容一致・base/pilot SHA・monitor最良選択をstrict検証。選択schemaのテストは実装直後に追加したため先行RED未取得というTDD逸脱を記録。fullはselected_config、A1 seed42/43/44とA0 seed42を同予算で実行する。全F評価実装は並行するが、test結果を候補選択に使わない。

2026-10-02 18:40:32 JST+0900、専用ブランチ・temp作業書のwrite/review依頼に対する再確認開始。全書を再読し、未実装とする古い説明、benchmarkのfixture引数不足、selected_config/long/全F評価の依存ゲートを修正した。既存の実装・訓練記録は保全し、新しいD項目はチェックしない。途中のapply_patchはcontext不一致で2回失敗し、正しい実在行へ限定して再適用した。コード・データへの変更は無し。

`review-written-workdoc` のrubricでreview-and-fix済み。初期REVISE→最終PASS_WITH_NOTES。指摘、適用変更、再読後判定、残留仮定は `temp/review_Oct02-2026_refiner_kd_onnx.md` に記録した。同一Codexによる自己レビューで、独立した別エージェント監査ではない。DoDではP-1/P-2のみ完了、D-1〜D-8は未達。手順単位の進捗は第3章と第7章を正とする。
| 2026-10-02 | 20:27:31 JST+0900 | root | 手順 24.6 進捗 | ユーザー追加研究/改善許可に従い temp/workdoc_Oct02-2026_kd_accuracy_improvement.md をwrite/review PASSで追加。旧16input1027/15complete保存、旧32prep停止・保全。queryGT不整合/unused272440実測を修正、新v2deploy615838/strictreanchor/FP32DINO/stagedKD対照へ。親protocolhash/margin全150本とD1–D8不変、改善途中でDoD完了にしない。 |
