# ONNX Runtime GPU 実動画の検証結果

2026年10月2日、最新best80のONNXを **onnxruntime-gpu 1.30.0 / RTX 5090 / CUDA13 / cuDNN9** で、比較動画と同じ3本の実入力に対して実行しました。
全481フレーム・全追跡点で成功し、4出力すべてがCPU版ORTと事前に固定した許容誤差内で一致しました。
metric-AJは3本それぞれCPU版と同値です。3D-AJはADTだけ約7.51e-7異なります。出力のビット一致を意味するものではありません。

対象はONNX **refinerのみ** です。WAFT flow、DA3 depth、DINO特徴はnative環境で生成し、同一入力をCPUとCUDAへ渡しました。
RGBからの全pipelineをONNX化した検証や、公式minival全体の評価ではありません。
[GPU実行report](evidence/onnx_gpu_video_20261002/gpu_report.json) と [AJ summary](evidence/onnx_gpu_video_20261002/summary.json) が実測の証跡です。

## 入力と比較条件

既存Desktopの比較manifestで推論前に選定した3本を、そのまま使いました。各subsetの訓練validation先頭1本なので、独立したtest集合ではありません。
PStudioは150F/460点、DriveTrackは31F/256点、ADTは300F/900点です。フレームや追跡点を間引いていません。

prepareでnativeとCPU ORTを再実行し、ray、z_raw、visibility、uv、depth_map、dino_features、intrinsics、z_refの8入力をFP32 NPZへ保存しました。
旧比較で記録済みの7入力は、3本ともshape・dtype・tensor SHAがすべて一致しました。
旧reportに独立hashのないz_refも、全B/F/Nに対する下側中央値+1e-6を検査しています。点chunkごとに計算し直していません。
[fixture manifest](evidence/onnx_gpu_video_20261002/fixture_manifest.json) にraw/depth/input/predictionのSHAと入力signatureを保存しています。

新しい検証CLIは [verify_refiner_gpu_video.py](../scripts/verify_refiner_gpu_video.py) です。
native/CPUの入力捕捉、隔離GPU Runtimeでのreplay、既存のscore関数によるAJ計算を3段階に分離しました。
点chunkは8、attention作業領域の推定予算は1024MiBで、時間方向は分割しません。
CUDAはTF32を無効化し、全4出力に `rtol=2e-4, atol=5e-5` を適用しました。測定後に閾値を緩めていません。
可視性は全手法で同じ二値WAFT forward-backward maskを使用し、GTやvis headへ置換していません。

## CUDAの実行確認

provider一覧だけでなく、実行profileに主要演算のCUDA kernelがあることを確認しました。
MatMul/FusedMatMul、GridSample、LayerNormalization、Sin/Cos、Exp、ReduceMeanなどはCUDAでした。
203点chunkの実行でCUDA **118,755 node-events**、CPU **17,864 node-events** を記録しています。これは実行event数で、graphのunique node数ではありません。

CPUで実行されたConcat/Gather/Mul/Reshape/Slice/Transpose/Unsqueezeは **int64の形状情報のみ** です。
主要計算のCPU割当やCPU浮動小数tensor計算は検証CLIで拒否しています。全graphがGPU-onlyという主張ではありません。
起動時のplugin EP device登録警告は出ましたが、CUDA providerがactiveで、実kernelのprofileを確認しています。CPU-onlyへのfallbackではありません。

## 数値差とAJ

GPUとCPUの最大絶対成分差はXYZ **2.670288e-5 m**、uv_refined **1.220703e-4 px**（896座標）、vis_logits **1.264736e-6**、delta_uv **1.601875e-7 px** でした。
atolだけではなく、atol+rtol×参照値の条件で全出力・全点が成功しています。
各出力別・各clip別の正確な値はGPU report、GT可視点の差の分布はAJ summaryを参照してください。

| subset | CPU metric-AJ | CUDA metric-AJ | CPU 3D-AJ | CUDA 3D-AJ |
| :--- | ---: | ---: | ---: | ---: |
| PStudio | 0.572746338 | 0.572746338 | 0.230767708 | 0.230767708 |
| DriveTrack | 0.154358507 | 0.154358507 | 0.029279799 | 0.029279799 |
| ADT | 0.358750327 | 0.358750327 | 0.222315567 | 0.222316317 |
| 3本の等重み平均 | 0.361951724 | 0.361951724 | 0.160787691 | 0.160787941 |

同じmaskのocclusion accuracyもCPUとCUDAで完全一致しました。
nativeのBF16 Mamba-3とFP32 ORTは同じ数値にはなりません。nativeの3本平均metric-AJは0.361929142、3D-AJは0.160891730です。
既存native対CPU比較と同様、subsetによる3D精度の悪化をGPUの動作確認で打ち消すものではありません。
[元の比較と限界](session_memory_20261002.md#同一動画のトラッキング比較) も参照してください。

## 再実行

repo rootからBashで実行します。既存のDesktop比較manifest/raw/depth/モデルと、認証済みのpinned DINO cacheが必要です。
人工入力だけで再検証したい場合は [ONNX方式文書のGPU手順](onnx_export_design_20261002.md#gpu版runtimeの再検証) を使ってください。

```bash
# rootのCPU ORTとGPU ORTを同じvenvへ混在させない。
# 初回だけ、既存GPU環境がなければ作成する。
uv venv --python 3.12 result/onnx_gpu_20261002/.venv
uv pip install --python result/onnx_gpu_20261002/.venv/bin/python \
  onnxruntime-gpu==1.30.0 numpy==1.26.4 beartype==0.22.9 jaxtyping==0.3.11

source scripts/cudnn_env.sh

# 新しい出力先を指定し、今回の証跡も元Desktop動画も上書きしない。
uv run --locked --extra onnx python scripts/verify_refiner_gpu_video.py \
  --stage prepare --out-dir result/onnx_gpu_video_recheck

PYTHONPATH="$PWD/src" result/onnx_gpu_20261002/.venv/bin/python \
  scripts/verify_refiner_gpu_video.py --stage verify \
  --out-dir result/onnx_gpu_video_recheck

uv run --locked --extra onnx python scripts/verify_refiner_gpu_video.py \
  --stage score --out-dir result/onnx_gpu_video_recheck
```

prepare中、同一プロセスでWAFTを2回目に初期化すると相対checkpoint pathが崩れる既存挙動を検出しました。
新CLIだけでWAFT builderを明示的なworking-directory contextへ入れ、正常時も例外時も元cwdへ戻す対応をしています。
対応前の失敗ログは `result/onnx_gpu_video_prepare_20261002.log`、再実行の成功ログは `result/onnx_gpu_video_prepare_retry_20261002.log` に保全しました。
元のtrain/evalコードは変更していません。入力hash・coverage・dtypeとcwd復帰の新7回帰ケースを含み、全単体テスト **156 passed / 26既存warnings**、対象ruff/format/tyと既存incremental tyが成功しました。
[品質記録](evidence/onnx_gpu_video_20261002/quality.json) を参照してください。

## 保存先と未評価範囲

raw profile・8入力NPZ・CPU/native/GPU予測・各reportは `result/onnx_gpu_video_20261002/` にあります。
小型のJSONだけをdocsに保存し、モデル・データ・動画・大きなprofileをGitへ追加しません。
rootのCPU ORT、uv.lock、元PT/ONNX、元Desktop動画4本とreport3本のSHAは不変です。
元の動画表示はnative/CPU比較のままです。今回GPU版の動画は生成していません。

**公式固定9本の受入gate、minival全150本、性能benchmark、他GPU/他OSは未評価** です。
report中のprofiling付き時間にはuploadや実行等を含みますが、warmup・反復・同条件CPU計時がないため性能benchmarkとして使えません。
今回確認したのは、best80 refinerのこの3本での実GPU動作と数値・AJの一致です。
