# DA無し訓練とモデル公開と比較デモの継続メモ

2026年10月2日の一連の作業を、次回の担当者が再調査せず継続するためにまとめます。
新しい保守的な未使用データ集合を取得し、DA無しキャッシュと訓練を完了、最良step80をReleaseの推奨にしました。
同じ3動画で旧モデル・新モデル・新ONNX CPUを比較し、別途旧新ONNXのCUDA実行も検証しました。
best80は追加で同じ3本の実入力をGPU版ORTにreplayし、全481Fの実GPU動作とAJも確認しました。
ただし公式minival全150本、固定9本の受入評価、GPU性能benchmarkは未評価です。

## 作業の位置づけ

対象は `yuki-inaho/vmamba3-3Dpointtracker` のofficial Mamba-3 adapter構成です。
論文のVSSD-2pool v64 + visibility head v94そのものではありません。
リポジトリは `/home/kasm-user/Desktop/vmamba3-3Dpointtracker`、データは `/workspace/vmamba3_data`、動画はDesktop直下の別directoryに置いています。
ここに記載する絶対pathは今回の端末の作業記憶で、別端末で同じpathが存在することは保証しません。

既存のONNX枝をローカルの `integration/onnx-best200-20261002` に取り込みました。
デモ関連6ファイルはcommit `7698044780b760ad26a9d5a870961d2c6e518adb` で公開済みです。
今回のdocs・GPU検証ツール・残っていた訓練修正もmain統合の対象です。
統合の順序は、docsの作成・検証・commitの後に通常のmain統合と非force pushです。
branch名はbest200由来ですが、推奨推論モデルはbest80です。Git tagや旧モデルを同じ名前の別内容へ書き換えません。

## データの取得と未使用判定

取得量は **99,949,386,750 bytes**、decimal 100 GB上限内の **1,042 clips** です。
ADT 590、DriveTrack 362、PStudio 90。sourceは `ZhengGuangze/TAPVid-3D`、revisionは `1575ec135a22e924d1702cd15e76527bcb89cc6e` に固定しました。

旧DA-off学習で実際に使用された230 clipsのIDはReleaseに含まれません。
そのため旧選択アルゴリズムのseed42/fraction0.15で得られる **663候補全体** を除外し、公式minival **150 clips全体** も除外しました。
新集合は公式FULL_EVAL内で、両除外集合との重複0を監査済みです。
これは公開情報から可能な保守的な非重複保証であり、非公開の過去実験全てについて未使用と証明したわけではありません。

`/workspace/vmamba3_data/manifests/` の保存先は次のとおりです。

| ファイル | 用途 |
| :--- | :--- |
| `unused_train_manifest.json` | 訓練に渡す1,042 clipsの集合 |
| `download_ledger.jsonl` | clipごとのarchive、revision、bytes、frames、SHA256台帳 |
| `excluded_prior_seed42_fraction015.json` | 保守的な旧候補除外集合 |
| `download_summary.json` | exact bytesとsubset件数、完了状態 |
| `final_audit.json` | 全raw rehash、CRC、重複0、depth確認の当初snapshot |
| `training_final_cache_observation.json` | 訓練終了後のfiles/bytes/budget/SHA観測 |

manifest SHA256は `1c4f6c02dd208767dd1b28037ee506000aaeb21872a5a38943d7c9cd62e2cc75`、台帳SHA256は `be722dc03a60882cf8afc3db9751580c314263215d73e5455bb24af6882dcfba`。
[公開用の小型集計](evidence/daoff_best80_20261002/data_cache_summary.json) は元JSONから必要な項目だけを抽出しています。今回の文書化で100 GB全体の再rehashはしていません。

## DA無しキャッシュと認証

**DA無しはphotometric data augmentationを無効にする意味です。Depth Anything 3を無効にする意味ではありません。**
WAFT、DA3 metric-large、DINOv3を凍結した前段として引き続き使います。

| 内容 | 完了時の件数と容量 | 保存先 |
| :--- | :--- | :--- |
| DA3 depth | 1,042 NPZ + ready markers、81,211,408,573 bytesのNPZ | `tapvid3d_da3/` |
| WAFT flow | 元seed42 windowの14,418 tensorsをstrict reread。variant込み19,136 files、122,947,058,624 bytes | `tapvid3d_frozen_cache/flow/d6e17af69987e592/` |
| DINO patchとCLS | 元8,225 keysをstrict reread。variant込み10,801 files、13,052,500,853 bytes | `tapvid3d_frozen_cache/dino/5a5f7acefff4d43a/` |

flowはworker-context試験で別キーが増え、旧120 GB LRUで元459 entriesが不足しました。
元キーを再導出して不足だけ修復し、variantを削除せず全14,418 entriesを再読込しました。
flow上限160 GB、DINO上限20 GB、workspaceデータ総量上限360 GBへ明示的に変更しています。raw上限は100 GBのままです。
詳細は `daoff_flow_cache_repair_status.json` と `daoff_dino_cache_repair_status.json`、訓練後snapshotを参照してください。

DINO認証エラーはユーザーの `~/.bashrc` 更新後に解消しました。
`final_audit.json` の `blocked_authentication` は当初snapshotとして残っており、現在の完了状態は後続の `postauth_cache_audit.json`、DINO status、repair、訓練終了の観測を合わせて読みます。
fallback encoderは使っていません。認証値は表示・保存・commitしません。
今回のモデルが要求するのは `facebook/dinov3-vits16-pretrain-lvd1689m` のrevision `114c1379950215c8b35dfcd4e90a5c251dde0d32` です。
会話中に示されたViT-B/16へ差し替えるとcheckpointの構成と合わないので使用していません。

訓練windowは8 frames、fixed seed42、画像896 px、DINO画像448 px、WAFT scale=-1/iters=4です。
最終訓練のloaderは **spawn workers 2 × CPU threads 8**。単一processの8-thread resizeと画像bitwise/keyが一致することを確認しました。
resizeのthread/contextを変えると凍結cacheのcontent hashが変わるため、再開時もこの条件を保持してください。

## 訓練設定と終了結果

短いsmokeと並行loader probeを経て、公開best200をwarm startにDA無しで訓練しました。
train **1,027 clips**、監視validation **15 clips**（各subset5）、AMUSE optimizer、最大batch **32**、trainable parameters **8,117,988**。
既存設定は最大800 steps・20,000 clip消費ですが、patience5/min_delta0.001のearly stopping条件を変更せず **step130 / 3,950 clips消費** で正常終了しました。
800 stepsを強制完走した結果ではありません。ユーザーの選択に従い、final130ではなく最良の **best80** を公開しました。

| 監視validation | total loss | 正規化3D L1 |
| :--- | ---: | ---: |
| 初期step0 | 0.1828703870 | 0.2011452418 |
| 最良step80 | 0.1613510862 | 0.1774710458 |
| 最終step130 | 0.1656259870 | 0.1821751678 |

同じ監視集合でtotal lossは初期から **11.77%低下**。
lossは可視点のdepth正規化XYZ L1とdelta_uv二乗の正則化で、logの `pos_2D` は2D追跡誤差ではありません。
visibility学習lossは0で、可視性は共通のWAFT forward/backward maskです。
旧Releaseのlossとはtrain/validation membershipが違うため直接比較できません。

再開後step11–130はDINO/flow cache misses **0**、flow OOM retries **0**、全scalarとmodel/optimizer tensorがfiniteでした。
GPU allocated peakは **24,579.76 MiB**。RTX 5090の32 GBを常時使い切る保証ではなく、depth-grid別batchで最大32を使い、allocatorのreserved量と実使用量を区別します。
PStudioの短いmotion診断48.7%は改善なしで、公式追跡精度ではありません。
数値・全loss履歴・15 clipsのIDは [training_summary_20261002.json](evidence/daoff_best80_20261002/training_summary_20261002.json) を正とします。

ローカルの再開元は `/workspace/vmamba3_data/result/v64_daoff_unused100gb_train_20261002/`。
`best_80.pt`、`latest.pt`、`training_status.json`、`loss_history.json`、`growing_pool.json`、`cfg.json`、TensorBoardを保全しています。
実行設定は `/workspace/vmamba3_data/configs/v64_daoff_unused100gb_train_20261002.yaml`。公開の [run_config](evidence/daoff_best80_20261002/run_config_20261002.json) は推論用で、訓練再開用のフル設定ではありません。

## Releaseの更新とモデル保全

[公開Release](https://github.com/yuki-inaho/vmamba3-3Dpointtracker/releases/tag/mamba3-preview-20261001-step200) の推奨をbest80に変更しました。
tagのstep200は歴史的名称です。CIが旧assetの固定SHAを参照するため旧11 assetsを保持し、新8 assetsを追加、合計19 assetsにしています。
全旧assets/bodyのbackupは `result/release_update_20261002/backup/`。削除や同名内容のclobberは行いませんでした。

| 推奨asset | SHA256 |
| :--- | :--- |
| `tracker_mamba3_daoff_best80.pt` | `c022511872c2d59a39bc1c39a3c62f44c142ee8ff9ca7dc7859881a78ae490b7` |
| `tracker_mamba3_step80.onnx` | `7b1530c38e01496e60f67ef1cd6b9eb2ebb250ec8108cb908d9c07217edef604` |

公開PTは52 tensorsで元best80とbitwise一致。凍結DINO backbone211 tensors、optimizer、RNG、host固有pathを除外し、`weights_only=True`で読込確認しています。
50 tracker tensorsにDINO前処理2 buffersを含むので「52 tensors」と「50 tracker tensors」は矛盾しません。
公開ファイルを別directoryへ再downloadし、SHAとPT/ONNXの由来を確認しました。
[public_download_audit.json](evidence/daoff_best80_20261002/public_download_audit.json) と [public_export_report](evidence/daoff_best80_20261002/public_export_report_20261002.json) が証跡です。
モデル本体は `weights/mamba3-preview-20261002-best80/` に残し、Gitに入れません。

## ONNXとGPU版Runtimeの検証範囲

ONNXは **refinerのみ**。DINO、WAFT、DA3を含むRGBからのend-to-end ONNXではありません。
Triton SISOを等価なFP32標準演算へ展開し、元のtracker重みを変更せずopset18・1,547 nodes・単一ファイルへexportしました。
方式・入出力・O(F²)制約・再変換コマンドは [ONNX化の説明](onnx_export_design_20261002.md) にあります。

best80のCPU ORTは4動的人工入力と点chunk一致、full checkerを検証済みです。
旧best200にはCPU ORTとC++の10正常/12異常ケースの既存証跡があります。
`docs/evidence/onnx_best200_cpu/` の108tests/版/測定時間は今回の端末で再実行した当時snapshotで、最新の単体テスト件数と混同しないでください。
同directoryの `paired_preflight.json` も初期の依存・固定9本入力不足のsnapshotで、現在のGPU runtime実動作を報告するものではありません。

追加要求により **onnxruntime-gpu 1.30.0 / CUDA13 / cuDNN9 / RTX 5090** でも旧best200・新best80を各4動的人工入力で実検証しました。
`use_tf32=0`、固定 `rtol=2e-4, atol=5e-5` でportable FP32、CPU ORT、点chunk1との比較が全て成功。
profileで主計算のCUDA実行を確認し、CPU併用はint64の形状処理だけでした。全node GPU-onlyとは主張しません。
[best200 GPU report](evidence/onnx_gpu_20261002/best200_gpu_report.json) と [best80 GPU report](evidence/onnx_gpu_20261002/best80_gpu_report.json) を参照してください。
rootのCPU ORT/uv.lockは変更せず、GPU環境は `result/onnx_gpu_20261002/.venv/` へ隔離しています。

その後、[GPU実動画検証](onnx_gpu_video_20261002.md) で同じ3本の全481F/全tracksをlatest best80でreplayしました。
入力signatureは旧比較と一致、4出力はCPU ORTと固定閾値で一致しました。metric-AJは各clipで同値、3D-AJ最大差7.51e-7、主要計算CUDA/CPUint形状のみをprofileで確認しました。
新7回帰ケースを追加し全156単体テスト成功。前段main統合時の149件とは検証時点が異なります。
新しい原本は `result/onnx_gpu_video_20261002/`、小型証跡は `docs/evidence/onnx_gpu_video_20261002/` です。元Desktop動画・reportsは不変です。

## 統合対象のソース変更と品質確認

今回の訓練で使用し、残っていた変更は次の3点です。推論のarchitectureや公開重みは変更しません。

- `depth_source.py`: workspaceへ移したdepth rootを、canonical source名・model・rootが一致する `depth_source.json` stampに限って許可する。stampなし/不正/別modelは拒否。
- `train_depth_refined_tracker.py`: motion診断も設定したdepth rootを使う。従来のhome固定pathによる診断不一致を解消する。
- `dataset.py`、`bucket_batch.py`、`fixed_da.py`: workerのthread/contextを明示し、訓練とcache prewarmでloader optionを共有する。複数threadのworkerはspawnを要求し、resize/key不一致を防ぐ。

GPU検証CLIとprofile判定の6回帰ケースも追加しました。
main統合時の単体テストは **149 passed / 26 warnings**。warningsは既存のforkとNumPy scalar変換のdeprecationです。後続のGPU実動画検証では156件へ増えています。
対象15 Python filesのruff、既存incremental ty gate、GPU checker/testのty、対象12 filesのformat check、diff checkが成功しています。
大きな既存legacyファイルを丸ごと整形してはいません。[品質記録](evidence/integration_20261002/quality.json) に検証範囲を記録しています。

## 同一動画のトラッキング比較

結果は `/home/kasm-user/Desktop/tracking_comparison_20261002/` に保存しました。
`comparison_all.mp4` は約32秒で、PStudio→DriveTrack→ADTの順です。個別動画3本・一覧PNG・snapshot・予測NPZ・各report・固定manifestも同じ場所にあります。
動画4本はH264/yuv420p、1920×1080、15fps、全481 framesをdecode確認しています。元動画の再生速度を保証するものではありません。

各subsetで訓練validationの先頭1本を推論前に固定し、全frames・全queriesを評価しました。
描画は同一32 IDs、色、カメラ、等軸メートルcube、過去30frames、GT破線/予測実線です。
論文図9の表示を参考にしていますが、DELTAとの比較や論文モデルの再現ではありません。
WAFT・DA3・DINO入力とflow visibilityを旧/新/ONNXで共有しました。

| 動画 | frames / points | 旧metric-AJ | 新native metric-AJ | 新ONNX CPU metric-AJ | XYZ差p95 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| PStudio boxes_21 | 150 / 460 | 0.51933 | 0.57270 | 0.57275 | 0.000711 m |
| DriveTrack | 31 / 256 | 0.11106 | 0.15428 | 0.15436 | 0.015293 m |
| ADT cook_seq143_0 | 300 / 900 | 0.33982 | 0.35881 | 0.35875 | 0.002018 m |
| 3 subsetsの等重み平均 | — | 0.32340540 | 0.36192914 | 0.36195172 | — |

この3本では絶対metric-AJは全subsetで上昇しました。
ただし中央値スケーリング後の3D-AJはPStudio **0.237754→0.230885**、DriveTrack **0.047180→0.029449** と低下し、ADTは **0.175505→0.222341** と上昇しました。
すべての指標で改善したとはいえません。
native BF16 mixerとONNX CPU FP32のAJ差は最大 **0.00016924**、GT可視XYZ差の最大値はDriveTrackで **0.110558 m**。完全一致ではありません。

これらはbest80の選択に使ったvalidationからの3本で、重み更新用1,027 clipsとは非重複ですが **独立testではありません**。
公式minival150本や旧Releaseの固定9本、論文のmetric-AJ0.256と直接比較できません。
全指標、clip ID、XYZ/UV差、動画SHAは [比較summary](evidence/tracking_comparison_20261002/summary.json) に保存しています。

## 次回の開始手順と注意点

まず本書・[docs索引](README.md)・workspaceのREADMEを読み、Git status、training_status、GPUプロセスを確認してください。
現在の学習はearly stopで終了しているため、再開コマンドを自動実行しないでください。
ユーザーの追加目的がなければ再学習、Release更新、cache削除、動画上書きを行いません。

```bash
cd /home/kasm-user/Desktop/vmamba3-3Dpointtracker
git status --short --branch
nvidia-smi
jq . /workspace/vmamba3_data/result/v64_daoff_unused100gb_train_20261002/training_status.json
uv run --locked --extra onnx pytest tests/unit -q
```

動画を再生成する許可がある場合だけ、保存済み予測を使って次を実行します。
raw/depth、モデル、pool manifestなどの元pathとSHAが必要で、空directoryからbootstrapするデモではありません。

```bash
uv run python scripts/demo_tracking_comparison.py --stage render
uv run python scripts/demo_tracking_comparison.py --stage audit
```

GPU推論にはbashから `source scripts/cudnn_env.sh` を実行し、認証済みのDINOアクセスを用意します。
token値をechoせず、秘密を含む `~/.bashrc` の内容をGit/docsにコピーしません。
詳細な生成手順は [root READMEのデモ節](../README.md#46-tracking-comparison-demo)、GPU再検証は [ONNX方式文書](onnx_export_design_20261002.md#gpu版runtimeの再検証) を参照してください。

継続が必要な未評価項目は、公式minival全150本、固定9本のnative/ONNX受入gate、GPU性能benchmark、他GPU/他OSです。
今回の成果だけでこれらの合格や本番採用を宣言しません。
