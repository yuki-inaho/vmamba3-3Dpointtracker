# 容量を制限した TAPVid-3D データ取得と段階的学習

README 4.1 の全データは train 4419 clips と minival 150 clips、約525 GB展開です。
この環境では全量を保存できないため、ユーザー追加指定の約60 GBを目安にまず部分学習を行います。
公式の全train学習と minival の absolute metric Average Jaccard 0.256 達成は別の未完了要件です。

## 現在の固定partition

`scripts/download_tapvid3d_subset.py` は Google公式splitの `FULL_EVAL_FILES` から各subsetを同じ比率でサンプルします。
標準設定は seed 42、比率0.15で、ADT 286 / DriveTrack 361 / PStudio 16、合計663 train clipsです。
名前をsortしてからsubset別seedでshuffleし、shard番号に依存せず選択するため、最初のshardのみ使う偏りを避けます。
150 minival は評価専用として別リストに保持し、trainに混入しないことをmanifest生成時にassertします。

manifestは `temp/tapvid3d_train_subset_manifest.json` です。
`files_by_subset` は `{subset: [filename, ...]}`、`train_files` は `[{subset, filename}, ...]`、
`evaluation_files_by_subset` は公式minival全150件です。全663 trainの取得は起動条件にしません。
`configs/v64_amuse.yaml` の `data.split.growing: true` により、raw NPZと検証済みdepthの
両方が準備できたclipを学習へ継ぎ足します。開始時に各subsetから5clipずつ検証用へ分離・固定し、
残る1clip以上/subsetで訓練を始めます。検証メンバーは `result/v64_amuse/growing_pool.json` に保存し、
再開しても変更しません。minivalは学習poolの対象外です。

5 optimizer stepsごとに新しいready clipを確認し、DataLoaderを作り直して取り込みます。
更新時にAMUSEの状態やstepをリセットしません。履歴は `data_additions.jsonl`、TensorBoardでは
`data/train_clips` です。選択数・取得数・学習準備完了数はそれぞれ区別して記録します。

## 実行と分割・再開

```sh
# 選択・件数・非混入を確認。ダウンロードなし。
uv run python scripts/download_tapvid3d_subset.py --dry-run

# 三subsetの全shardから選択trainとminivalを取得。
uv run python scripts/download_tapvid3d_subset.py --split all \
  --raw-budget-gb 30 --total-data-budget-gb 60 --reserve-gb 30

# subsetを分けて取得する例。manifestとseedは同一にする。
uv run python scripts/download_tapvid3d_subset.py --subsets adt --split all
uv run python scripts/download_tapvid3d_subset.py --subsets drivetrack pstudio --split all

# 評価データのみ、またはtrainのみを後から追加可能。
uv run python scripts/download_tapvid3d_subset.py --split minival
uv run python scripts/download_tapvid3d_subset.py --split train
```

環境構築中でも本体の環境ファイルを変更せず取得できます。

```sh
uv run --no-project --with requests python scripts/download_tapvid3d_subset.py --split all
```

README指定の https://huggingface.co/datasets/ZhengGuangze/TAPVid-3D を使います。
24 gzip tar shards をstreamingで読み、manifestに一致するNPZだけ保存します。
全tarはディスクに置きませんが、gzipは任意位置から解凍できないため、選択率15%でもネットワーク転送は最大約474 GBです。
中断後は同じコマンドを再実行してください。完成NPZのZIP CRCと実loaderの必須キーを検証してskipし、
中断したshardを先頭から再走査します。部分NPZは完成と扱わず、`.npz.part` に書いて検証後atomic renameします。
再開単位はclipです。gzipの転送バイト単位での再開ではありません。

`temp/tapvid3d_subset_download_status.json` はshard走査履歴、取得件数、missing名、容量ブロック名を持ちます。
ログの `verified` は画像とGTが入ったNPZのCRC/必須キー検証済みです。
同じmanifest pathでseedや比率を変えることは拒否します。別partitionを追加する場合は新しいmanifest/state pathを指定してください。

```sh
uv run python scripts/download_tapvid3d_subset.py --seed 43 --train-fraction 0.15 \
  --train-manifest temp/tapvid3d_partition_2.json \
  --state-json temp/tapvid3d_partition_2_status.json
```

別seedのpartitionは既存partitionと重複し得ます。完全に非重複の次partitionが必要な場合は、
既存manifestのtrain名集合を除外して新たにmanifestを作り、train loaderにも新manifestを明示する必要があります。

## 約60 GBの容量配分

- raw NPZは30 GB（10進）上限。旧方式で取得済みの正規NPZは再利用します。
- depthとDINO/flowの一時cacheも含め、`~/data` の総量約60 GBをguardします。
- tarはstreaming読取のみで保存しません。NPZ内部のJPEG bytesを画像フォルダへ別途展開しません。
- 環境・重み・checkpoint用にディスク全体で30 GBの空きを残します。
- 並列書込み用に総量から1 GBの余裕を確保します。深度は元の504解像度・uint16精度を維持します。
- 予算により未取得が残っても、準備済みpoolの学習は進めます。選択663件や公式4419件を全取得したとは扱いません。
- DINO/flow cacheはLRUで古い自分の生成物だけを削除し、raw/GT/学習checkpointは削除しません。

## 現環境の進捗

データrootは `/home/kasm-user/data/tapvid3d`。初期のminival取得でDriveTrack50/PStudio50とADTの一部を取得し、
三subset混合train要件を受けて統合downloaderへ移行しました。既取得minivalは再検証してskipします。
正確な現在件数は `temp/tapvid3d_subset_download_status.json` とログ・実ファイルを照合してください。
これはsubset訓練のデータであり、公式全4419clipの再現を完了したという意味にはなりません。

## 人工データによるsmoke train

ダウンロード完了を待たず、既知の3D軌跡と深度・画像を人工生成して訓練経路を検証できます。
実際の2pool Mamba3 refiner、凍結済みDINOv3、FlowVisHead、AMUSEを使います。
WAFT/DA3の測定値は解析的に生成するため、このコマンドでは両ネットワークを実行しません。
取得済みDINOv3の正規revisionをオフラインで読みます。

```sh
# リポジトリrootで実行。cuDNNとCUDA runtimeの参照先を設定。
. scripts/cudnn_env.sh
uv run python scripts/smoke_train_synthetic.py --steps 6

# latestの平均重みとoptimizerを復元し、8stepまで継続。
uv run python scripts/smoke_train_synthetic.py --steps 8 --resume

# 初めから別runを作る場合。
uv run python scripts/smoke_train_synthetic.py --steps 6 \
  --out-dir result/synthetic_smoke_2

# 人工データでDINOキャッシュの出力一致とAMUSE更新を確認。
uv run python scripts/smoke_train_synthetic.py --steps 3 --frozen-cache \
  --out-dir result/synthetic_cache_smoke

uv run tensorboard --logdir result --host 0.0.0.0 --port 6006
```

GPUを使わない試行は `--device cpu` を明示します。
このブランチのuv環境はGPUで検証した `torch==2.12.1`、`torchvision==0.27.1`、
`nvidia-cudnn-cu13==9.26.0.51` とTensorBoardを固定しています。
`uv sync` で `uv.lock` と同期してください。
`result/synthetic_smoke/summary.json` にlossと完了stepを記録し、
`refiner/` と `visibility/` の各ディレクトリへ `best_<step>.pt` 最大3個と `latest.pt` を保存します。
`checkpoints.json` は検証lossの順位、`tensorboard/` はtrain/val loss・gradient・LRです。
人工smokeの結果を公式TAPVid-3Dの精度として扱いません。

## 実データの継ぎ足し学習

以下を別プロセスで動かします。全ダウンロード完了を待つ必要はありません。

```sh
# 1: raw取得。予算内の正規trainとminivalを追加。
uv run python scripts/download_tapvid3d_subset.py --split all \
  --raw-budget-gb 30 --total-data-budget-gb 60

# 2: 深度batch生成。subset交互の順番で初期train6clipずつを優先。
. scripts/cudnn_env.sh
uv run python scripts/precompute_da3_depths.py --watch \
  --train-manifest temp/tapvid3d_train_subset_manifest.json \
  --auto-batch --chunk-frames 16 --max-batch-frames 128 --gpu-reserve-gb 8 \
  --decode-workers 8 --cpu-threads 8 --stats-json temp/da3_gpu_stats.jsonl --total-data-budget-gb 60

# 3: readyデータで学習開始し、後から届くclipを追加。
. scripts/cudnn_env.sh
uv run python scripts/train_depth_refined_tracker.py \
  --config configs/v64_amuse.yaml --data-root ~/data
```

HF_TOKENは環境へexportしてください。この環境で `~/.bashrc` の設定を使う場合は
`bash -ic 'uv run python scripts/train_depth_refined_tracker.py --config configs/v64_amuse.yaml --data-root ~/data'`
で値を表示せず継承できます。

depthは `.npz.partial` へ書き、原子renameとCRC・frame数の検証後に `.npz.ready.json` を作ります。
学習poolはready marker付きの完成物だけを使います。再開コマンドは同じものです。
`--watch` は後から到着するNPZを繰り返し確認します。データ不足で止まった場合はPID/logを確認してから再開します。

refinerのbestは固定heldout15clipのlossで選び、visibilityのbestは独立heldoutのaccuracyで選びます。
minivalは最終評価専用です。AMUSEは独自warmupと平均化を持つため外部WSD schedulerを使いません。
検証・保存では平均重みXを用い、再開時にtrainモードYへ戻します。

## 処理clip数による学習上限

batch sizeを変更した後も、訓練量をoptimizer update数だけで比較しません。
`train.clip_budget` は実際にoptimizerへ渡したclip数をcheckpointへ保存し、上限に達すると停止します。
`configs/v64_amuse.yaml` は元のbatch=1・20,000 updateを20,000 clip相当として扱います。
checkpoint 3,750までの旧runとbatch32 runの実測は4,989 clip相当なので、再開時は
`resume_clip_count: 4989` から開始します。新しいcheckpoint以降は保存値が優先され、
`data/processed_clips` と `data/remaining_clips` がTensorBoardへ出ます。
`steps: 4400` は異常時の安全上限で、通常はclip budgetかEarly stoppingで先に終了します。

## 論文指標を使う固定reference評価

`scripts/eval_metric3d.py` はvendored official TAPVid-3D evaluatorで、二つの指標を並べて出します。

- `overall.average_jaccard`: median scale補正とdepth-relative thresholdを使う公式leaderboardの3D-AJ。
- `overall.metric_average_jaccard`: scale補正なし、1 cm〜2.56 mの固定thresholdを使う論文headlineのabsolute metric-AJ。

論文の完全minival 150 clipsのheadlineは後者の **0.256** です。
現在の容量制約では、rawとDA3 readyが揃うofficial minival各subset 3 clips、計9 clipsを
`configs/v64_metric_reference_minival.json` に固定しました。これは訓練集合とは分離しています。
同じ公式metric実装でチェックポイント間の変化を見るreferenceであり、9/150 clips・部分訓練230 clipsの値を
0.256の再現/比較値と扱いません。full 150/150が揃った後だけ最終評価として比較します。

```sh
# readyなofficial minivalから固定referenceを再作成する場合。
uv run python scripts/create_metric_reference_manifest.py \
  --data-root ~/data --depth-root ~/data/tapvid3d_da3 --per-subset 3 \
  --out configs/v64_metric_reference_minival.json

# 停止済みcheckpointを評価する。学習と同じWAFT/DA3設定はckpt横のcfg.jsonから読む。
. scripts/cudnn_env.sh
bash -ic 'uv run python scripts/eval_metric3d.py --method v35 \
  --ckpt result/v64_amuse/latest.pt --depth da3l --split minival \
  --clip-manifest configs/v64_metric_reference_minival.json \
  --out-dir result/v64_amuse/reference_eval/step_XXXX'

# 0.256との差とreference範囲を機械可読で保存する。
uv run python scripts/summarize_reference_metric.py \
  --metrics result/v64_amuse/reference_eval/step_XXXX/metrics.json \
  --manifest configs/v64_metric_reference_minival.json \
  --out result/v64_amuse/reference_eval/step_XXXX/reference_progress.json
```

`metrics.json`、`summary.md`、`reference_progress.json` は常に同じcheckpoint pathと9本のmanifestを記録します。
full評価では`--clip-manifest`を外し、150 clips・各subset50・failure 0を確認してから
`metric_average_jaccard >= 0.256` を判定してください。

## 大バッチ学習とEarly stopping

`configs/v64_amuse.yaml` は同じdepth gridのclipを最大32本まとめるbucket samplerを使います。
元depthの解像度と精度を保持し、4 DataLoader workers・worker別seed・prefetch 1で画像を準備します。
bucket末尾の小さいbatchも捨てず、poolに追加されたclipを次のloader更新で取り込みます。
checkpointは開始・再開後の最初の更新と10更新ごと、学習ログは毎更新記録します。
stepはoptimizer更新回数です。batchを変更すると同じstep数でも処理するclip数が変わります。

`flow.batch_size: 32` から `flow.max_batch_size: 64` まで、複数clipの隣接frame pairをGPUでまとめます。
clip間をまたぐpairは作りません。forward/backward flowを現在のbatchだけRAMに置き、追跡時に再利用します。
元recipeのphotometric augmentationは有効です。この方式はdiskのfrozen cache方式と同時指定できません。
空きVRAMと自processの再利用可能なallocator cacheを考慮し、4 GBを残します。
バッチ容量は各GPU callのpeakから見積もり、小さな末尾batchに前callのpeakを流用しません。
flowがOOMなら同じpairからbatchを半減して再試行します。refinerのbatch変更は別out_dirのprobeで確認してください。
TensorBoardの `data/batch_clips`、`system/flow_executed_batch`、`system/flow_oom_retries`、
`system/gpu_memory_mib` で実際のclip数、flow batch、再試行数、更新ごとのpeakを確認できます。

refinerは固定heldout15clipのlossを500更新ごとに検証し、`early_stop_min_delta: 0.001` を超える改善が
`early_stop_patience: 5` 回連続でない場合に停止します。改善時は待ち回数を0へ戻します。
小さな改善もK-best順位には反映し、停止の判断とcheckpoint選択を分けます。
`latest.pt` は停止した実stepと待ち回数を保存し、`best_*.pt` は最大3個保持します。
停止理由・step・bestのpathは `training_status.json` に保存します。
新しいpolicyを設定した古いcheckpointの再開では、保存済み検証履歴から待ち回数を復元します。
同じpolicyで停止済みのrunは同じコマンドで再開しても追加更新しません。
継続したい場合は `early_stop_patience: 0` とするかpolicyを変更してください。

visibilityの `configs/v94_amuse.yaml` はheldout accuracy最大化、250更新ごとの検証、同じpatience/deltaを使います。
人工smokeの短い固定step検証には早期停止を適用しません。

実GPUの別out_dirのprobeではbatch32でfiniteなAMUSE更新、更新ごとのpeak約35.5 GiB、
GPU全体の使用約44.9 GiBを確認しました。値は48 GBのA6000での短い実測です。
現在のrunはlatest step3700から再開し、固定heldoutのメンバー・学習済み重み・AMUSE状態を引き継ぎます。
容量待ちのdepth workerは停止してVRAMを学習へ戻しています。容量を確保して深度生成を再開するときは、
学習のflow batch上限とreserveも再調整してください。全学習の完了や公式minival精度を示す検証ではありません。

## 凍結encoderと光学フローを再利用する高速方式

```sh
. scripts/cudnn_env.sh
uv run python scripts/train_depth_refined_tracker.py \
  --config configs/v64_amuse_cached.yaml --data-root ~/data
```

DINOのパッチ特徴・CLSとWAFTのdense flowを一時ファイルへ保存し、同一入力を再利用します。
保存時にdtypeや精度を変更しません。DINOのtrainable projectionと3D refinerは毎回forward/backwardします。
入力の値・形状・dtype、モデル重み、scale/iters等の設定、autocast dtypeがcache keyに含まれ、
重みや条件の異なる生成物は混用しません。

cacheは `~/data/tapvid3d_frozen_cache/`、DINO最大2 GB・flow最大4 GBのLRUです。
モデルrevisionが変わった古いcacheも各上限に含めます。runを再開しても生成物を再利用できます。
総データ予算で保存できない場合も学習はその入力を通常計算して続けます。
TensorBoardの `cache/dino_hits`、`cache/flow_hits` とmissesで再利用を確認できます。

固定入力の第1段階では、ユーザー許可により raw/depth を含む `~/data` 全体を約120 GBまで
使えます。`configs/v64_amuse_cache_phase.yaml` はDINO 20 GB・flow 40 GBを上限にし、
raw/depth/checkpointを消さず cache のLRUだけを退避します。通常の60 GB分割取得configの
上限は変えません。

高速方式では再利用率を上げるため **photometric augmentationを無効** にします。
元recipeの `configs/v64_amuse.yaml` はaugmentationを保持します。両runは別out_dirへ保存し、
高速方式を論文のAdamW・全データrecipeと同一の再現条件とは扱いません。
DINOのtrainable multi-layer fusionにはこのcacheを使いません。

## DAなしcache段階からDA付き微調整への引継ぎ

固定入力の高速化だけを設定しても、DA付きrunをそのまま続ければ再利用できません。
以下の runner は、停止済み `result/v64_amuse/latest.pt` を重みだけ読み込んで、第1段階を
実行します。第1段階が early stopping・clip budget・steps のいずれかで正常終了した後、
heldout loss 最良の `best_*.pt` を固定reference 9 clipで評価し、それが成功した場合にのみ
DA付き第2段階を開始します。optimizer/AMUSE状態は段階ごとに新規なので、DAなしの探索状態を
DA付き微調整へ誤って持ち込みません。

```sh
. scripts/cudnn_env.sh
bash -ic 'uv run python scripts/train_v64_staged.py --data-root ~/data \
  --cache-config configs/v64_amuse_cache_phase.yaml \
  --finetune-config configs/v64_amuse_da_finetune.yaml' \
  > temp/v64_staged_train.log 2>&1 &

# GPUを使わず、開始元・DA設定・引継ぎ先だけを検証する。
uv run python scripts/train_v64_staged.py --dry-run \
  --cache-config configs/v64_amuse_cache_phase.yaml \
  --finetune-config configs/v64_amuse_da_finetune.yaml
```

- 第1段階: `configs/v64_amuse_cache_phase.yaml` → `result/v64_amuse_cache_phase/`。
  DAを切り、DINO本体とWAFTを凍結cacheから供給する。時間窓とquery選択は維持するため、
  生フレームは同じでも監督サンプルを一種類に固定しない。
- 評価: `result/v64_amuse_cache_phase/reference_eval/best_<step>/` に、9/150 clipの
  `metrics.json` と `reference_progress.json` を保存する。paperの0.256との比較値ではなく、
  段階間の同一reference監視値である。
- 第2段階: `configs/v64_amuse_da_finetune.yaml` → `result/v64_amuse_da_finetune/`。
  `init_best_from` は第1段階の `checkpoints.json` の首位だけを解決する。DA時は入力値が
  変わるため persistent frozen cache を使わず、GPU cross-clip flow batchを使う。

第1段階のprocessが非zero終了、評価が失敗、best checkpointが欠落した場合はrunnerが例外で
終わり、第2段階は起動しません。中断後は同じrunnerを実行すれば第1段階の`latest.pt`から
再開します。第2段階の結果を再計測するときは、学習終了後に同じreference manifestを渡します。

## 公式Mamba-3の事前学習済みmixerを使う実験

`state-spaces/mamba` の公開最小Mamba-3は
[mamba3-siso-187m](https://huggingface.co/state-spaces/mamba3-siso-187m)です。
これはFineWeb-Eduで学習した言語モデルで、幅768/state128/head24です。
元のtrackerは幅128/state64/head4の独自VSSD cross-attentionで、同じSSMの考え方を使っていても
checkpointのtensor名と投影構造が異なります。`strict=False`はshape不一致を変換しません。
[visionMamba3](https://github.com/MasahiroOgawa/visionMamba3)の公開README/tree/releasesには
このtrackerへ転用できる事前学習checkpointの配布先がありませんでした（2026-10-01確認）。

`OfficialMamba3Adapter` は128→768→128の新規projectionの間に、公式`Mamba3`をそのまま使います。
最小checkpointの先頭2層のmixerと入力RMSNormを読み込み、言語embedding/MLP/headを除きます。
2層合計18 tensors・7,645,280 parametersが完全一致し、mixerの読み込みcoverageは100%です。
refiner全体の学習対象は620,502から8,117,988 parametersに変わります。
実際の読み込み結果・tensor mapping・source revision・ファイルsha256はrun内の
`pretrained_mamba3_report.json`へ保存します。無理なslice/reshapeは行いません。
trackerの入力・出力層は旧latest step3780から引き継ぎ、以前のVSSD層を公式mixerへ交換します。
forward/reverseの公式scanは重みを共有し、平均で両方向の時間情報を与えます。
これはVSSD-2poolと異なる実験構成で、論文headlineの完全再現runとは区別します。

```sh
# 取得revisionを固定。モデル本体は357 MiB、weights/はgitignore対象。
uv run python scripts/download_mamba3_weights.py
. scripts/cudnn_env.sh

# 人工データで実pretrained mixer・AMUSE更新・DINO cache一致を検証。
uv run python scripts/smoke_train_synthetic.py --steps 3 \
  --temporal-mixer official_mamba3 --frozen-cache \
  --out-dir result/synthetic_official_mamba3_smoke

# runnerの現在の標準設定は公式mixerの二段階学習。
bash -ic 'uv run python scripts/train_v64_staged.py --data-root ~/data' \
  > temp/v64_official_mamba3_staged.log 2>&1 &

# 従来VSSDの二段階実験はconfigを明示する。
uv run python scripts/train_v64_staged.py \
  --cache-config configs/v64_amuse_cache_phase.yaml \
  --finetune-config configs/v64_amuse_da_finetune.yaml --data-root ~/data
```

新runは`result/v64_official_mamba3_cache_phase/`と
`result/v64_official_mamba3_da_finetune/`へ保存します。元runとcheckpointは保全します。
両段階のheldoutは`result/v64_amuse/growing_pool.json`から完全に同じ15本を継承します。

第1段階は`fixed_window_seed: 42`でclipごとの8-frame時間窓を固定し、query選択は毎回変えられます。
最初に全ready trainのDINO/forward-backward flowをGPU batchでprewarmし、生成が終わってから
refinerを更新します。初回生成と学習のflowはFP32、DINOは同じBF16条件を使ってkeyを一致させます。
flowのmissは32 pairから最大64 pairで複数clipをまとめ、clip境界をまたぐpairは作りません。
cache 20 GB +40 GB、raw/depth込み120 GB guardを維持します。
この段階では固定時間窓以外のフレームを学習しないため、第2段階でランダム時間窓とDAを戻します。

正常終了した第1段階bestを9/150の固定referenceで評価し、成功してからDA微調整へ引き継ぎます。
微調整終了後も同じreferenceでbestを再評価します。ステージの失敗・中断・best欠落は次段階を
起動しません。各runにK-best3+latest、早期停止、clip countとTensorBoardを保存します。

## GPUを活用した生成と学習の並行実行

深度の `--auto-batch` は16 frameからバッチを拡大し、最大128 frame・空きVRAM・8 GBの
学習用余裕を考慮します。OOM時は同じframe位置から半分のバッチで再処理し、frameを飛ばしません。
独立frameを処理する `da3metric-*` のCUDA実行に限定しています。通常のDA3の複数viewモデルは
文脈が変わるため、このオプションでは明示エラーになります。
504解像度、公式のforward/出力変換、uint16保存を維持し、不要な可視化RGBだけ省きます。
JPEGは8 CPU threadsでデコードし、前clipの圧縮・CRC・ready保存を次clipのGPU処理と重ねます。
書込み待ちqueueは1clipまで。通常のバッチごとの `empty_cache()` は呼びません。

高速学習configではflow cache missを4 pairから最大8 pairへまとめてGPUで計算し、
同じwindowの追跡はCPUの一時bufferと保存cacheを再利用します。DINOの最大batchは32 frameです。
cacheの容量集計・LRU処理はバッチごとに一度行い、flowのCPU保存1batchと次のGPU計算を重ねます。
CPU tensorのsliceは保存前に複製し、元batch全体を何度も保存することを避けます。
depth/fast trainingのCPU計算threadsは8に制限し、複数processのthread過剰起動を避けます。
学習用VRAMの余裕は `frozen_cache.gpu_reserve_gb`、バッチ上限は `flow_max_batch` / `dino_batch` で指定します。
これらは同時に動くdepth workerも考慮するための余裕です。生成物のLRU上限と総60 GB guardは維持します。

生成速度とpeak GPU memory、batch、OOM回数は `temp/da3_gpu_stats.jsonl`、
flowのbatchとOOM回数はTensorBoardの `cache/flow_batch` / `cache/flow_oom_retries` に記録します。
GPUのバッチ計算は単一pair計算と微小な丸め差が出ることがあります。実GPUの14 pair比較では
平均絶対差約0.001 px、深度64 frameでは標準16 frame経路とのp99相対差0でした。
計測は他の学習・前処理と同時実行した短い確認で、全学習の速度倍率を保証する測定ではありません。

`--frozen-cache` の人工smokeは通常計算・初回cache・再読込のDINO出力完全一致をassertし、
`summary.json` に一致結果とhit/missを保存します。実データでは
`result/v64_cache_probe/` の2step AMUSE試行とTensorBoardでも動作を確認しました。
この短い試行は高速化率や公式精度の測定ではありません。
