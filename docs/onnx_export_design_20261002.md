# Mamba-3 refinerをONNX化した方式と検証

今回のONNX化は、CUDA/TritonのMamba-3 SISOを **同じ重みを使うFP32標準演算の計算グラフ** に展開して実現しました。
別世代のMambaへ置き換えたり、再学習したり、custom CUDA operatorをONNXへ埋め込む方法ではありません。
対象は旧best200と新best80のofficial Mamba-3 adapter refinerです。RGBからの全pipelineや論文のVSSD-2pool構成を丸ごとONNX化したものではありません。

## 元のkernelを直接exportしなかった理由

native実装の [OfficialMamba3Adapter](../src/mamba3_tracker/model/official_mamba3.py) は128次元のtracker特徴を768次元へ投影し、公式Mamba-3 SISOをCUDA BF16で実行して128次元へ戻します。
正順と逆順は同じ重みを共有し、出力を平均します。

直接exportを試した [失敗記録](evidence/onnx_best200/direct_export_attempt.json) では、torch.exportのFakeTensorからcustom kernelのdata pointerへアクセスしようとしてTorchExportErrorになりました。
opaque custom opにして追跡だけ通しても、標準ONNXと通常のORT/C++で実行できる計算定義は別途必要です。
そこでnative学習コードを残し、deployment専用の [onnx_refiner.py](../src/mamba3_tracker/model/onnx_refiner.py) を作りました。

## 保持した計算と標準演算への対応

portable版はadapterの投影、norm、Mamba-3固有の状態更新、双方向平均、refinerのheadsと射影を保持します。
学習済み50 tracker tensors、8,117,988 parametersを厳格に読み込み、元PTとbitwise一致を確認してからexportします。

| 元の処理 | portableでの表現 |
| :--- | :--- |
| up/downとin/out projection | 同じLinear重み、MatMulとAdd |
| RMSNorm | 二乗、ReduceMean、Reciprocal/Sqrt、同じweightとeps=1e-5 |
| 入力依存のAとdt | clamp、reciprocal、softplus、同じdt_bias。負のAとA×dtを保持 |
| rotary state | tanh、CumSum、周期の正規化、Sin/Cosとペア回転。rope_fraction=0.5 |
| 台形離散化 | sigmoid(trap)と隣接dtのshiftから積分係数を構築 |
| SISO recurrent scan | 因果mask、安定なsegment CumSum、Exp decay、QKとVのMatMul |
| 対角項の補正 | 元の非回転QKとshift係数の補正を明示的に減算 |
| D skipとZ gating | D×Vを加算後、SiLU(Z)を乗算 |
| 双方向scan | frame反転で同じmixerを実行、反転し直して正順と0.5平均 |
| depth/DINO samplingとXYZ | 標準GridSample、depth補正、Kによる視線への逆射影 |

SISOを単なる注意機構へ学習し直したわけではありません。
recurrent stateを展開し、各時刻のQK相互作用に減衰と台形積分係数を掛ける、公式参照式と同じ計算をattention状の行列として実装しています。
台形法のshiftには次時刻の係数が現れますが、対角補正により未来依存を除去します。
減衰のsegment和は大きなprefix累積和同士を引く方法を避け、丸め誤差を抑える形にしています。

[test_onnx_refiner.py](../tests/unit/test_onnx_refiner.py) はpinned upstreamの `mamba3_siso_step_ref` とframe数 **1/8/31/128/257** で比較し、prefixの因果性も確認します。
参照ファイル全体のSHA256を検証してから当該関数だけを読込むため、参照の版違いを黙って使いません。
upstreamコードrevisionは `e9594ce1c732d97440f0332fdc43170a2294dbfa`、参照ファイルSHAは `b68ec350f557a4124516f7d1c916ec756a48389fed4387791aadb74492531e85` です。
このFP32数式検証とnative BF16の実動画精度は異なる検証です。

## export対象と入出力

DINO、WAFT flow、DA3 depthは外部で計算します。
ONNXには次の **8入力** を渡し、全てfloat32です。B=batch、F=frames、N=tracks。

| 名前 | 形状 | 内容 |
| :--- | :--- | :--- |
| `ray` | B,F,N,2 | Kで正規化した視線 |
| `z_raw` | B,F,N | 元uvで取得したメートル深度 |
| `visibility` | B,F,N | 二値のflow可視性。GTや別の学習headで代用しない |
| `uv` | B,F,N,2 | 896 px座標系の2D追跡 |
| `depth_map` | B,F,Hd,Wd | メートル深度grid |
| `dino_features` | B,F,384,28,28 | 凍結ViT-S/16、448入力のpatch特徴。feat_projより前 |
| `intrinsics` | B,3,3 | 896 px画像に合わせた歪み/skewなしのK |
| `z_ref` | scalar | 分割前の全B/F/Nに対するz_rawの下側中央値+1e-6 |

出力は `xyz (B,F,N,3)`、`uv_refined (B,F,N,2)`、`vis_logits (B,F,N)`、`delta_uv (B,F,N,2)`。
`z_ref` はhost側で一度だけ計算します。偶数個の場合、NumPyの通常medianの中央2値平均ではなく **torch.medianと同じ下側中央値** を使います。
点chunkごとにmedianを計算し直すと元の計算と違うので禁止します。

[checkpoint loader](../src/mamba3_tracker/deployment/checkpoint.py) はweights_only読込、finite、keys、shape、dtypeを検査します。
native復元では公開仕様で除外されたDINO backbone全体だけを欠落許可し、trackerの欠落や部分的なbackbone欠落は拒否します。
portable版では外部DINO関連を除いた50 tracker tensorsをstrict loadします。

[exporter](../scripts/export_refiner_onnx.py) は `torch.onnx.export(..., dynamo=False, opset_version=18, external_data=False)` を使います。
B/F/Nとdepth gridを動的にし、checkpoint SHA・step・config・DINO revision・z_ref定義をmetadataへ記録、full checker・標準domain・単一ファイルを確認します。
best80のartifactは1,547 nodes、32,669,333 bytes、SHA256 `7b1530c38e01496e60f67ef1cd6b9eb2ebb250ec8108cb908d9c07217edef604`。

## メモリと利用上の制約

portable版はF×Fの行列を作るので、nativeの固定サイズrecurrent stateと違い、frame数に対し **O(F²)** の作業領域を使います。
これはCPU/CUDAのどちらでORTを動かしても同じgraph上の制約です。

[runtime.py](../src/mamba3_tracker/deployment/runtime.py) は独立したtrack Nだけを分割します。
frameを分割して状態をリセットする方法には切り替えません。
推定attention作業領域が予算を超える場合はエラーにし、track chunkを小さくします。推定値はtotal RSS/VRAMの上限保証ではありません。
長い動画全体や大きなtrack chunkでは追加のメモリ確認が必要です。

portable側はFP32、native mixerはBF16なのでビット一致を約束しません。
ONNX化で元の学習kernelやDAの設定を変えていません。
CPU配布用の `session_for` / `run_refiner_onnx.py` はCPU EP専用のままです。GPUの実チェックは別CLIで行い、CPU-only配布・CIの依存を壊さないようにしています。

## CPUでの再変換

repo rootから公開best80を指定します。出力directoryは明示し、元のPT/ONNXを上書きしません。

```bash
uv sync --locked --extra onnx --python 3.12
uv run --locked --extra onnx python scripts/export_refiner_onnx.py \
  --ckpt weights/mamba3-preview-20261002-best80/tracker_mamba3_daoff_best80.pt \
  --expected-sha256 c022511872c2d59a39bc1c39a3c62f44c142ee8ff9ca7dc7859881a78ae490b7 \
  --out-dir result/onnx_best80_reexport --threads 4 --long-frames 128
```

exporterはB/F/N/depth gridが異なる4人工入力をportable FP32とCPU ORTで比較し、track chunk1と一括も比較します。
固定閾値は `rtol=2e-4, atol=5e-5`。結果は `export_report.json` と `SHA256SUMS`。
新best80の実証跡は [export_report.json](evidence/daoff_best80_20261002/export_report.json)、旧best200のC++経路は [既存CPU文書](onnx_best200_cpu_20261001.md) を参照してください。

## GPU版Runtimeの再検証

2026年10月2日に **onnxruntime-gpu 1.30.0 / CUDA13 / cuDNN9 / RTX 5090** で実検証しました。
GPU wheelをrootのCPU ORTと同じ環境へ重ねて入れず、別venvを使います。
[ORT公式CUDA EP文書](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) のCUDA/cuDNN互換条件に合わせています。

```bash
# portable FP32の基準入力と基準出力を作る。DINOダウンロードは不要。
uv run --locked --extra onnx python scripts/verify_refiner_gpu.py --stage prepare \
  --ckpt weights/mamba3-preview-20261002-best80/tracker_mamba3_daoff_best80.pt \
  --onnx weights/mamba3-preview-20261002-best80/tracker_mamba3_step80.onnx \
  --out-dir result/onnx_gpu_20261002/best80

# 初回だけ作成する。既存venvの削除やroot CPU環境の置換は不要。
uv venv --python 3.12 result/onnx_gpu_20261002/.venv
uv pip install --python result/onnx_gpu_20261002/.venv/bin/python \
  onnxruntime-gpu==1.30.0 numpy==1.26.4 beartype==0.22.9 jaxtyping==0.3.11

# BashからCUDA13/cuDNN9ライブラリを揃えて実行する。
source scripts/cudnn_env.sh
PYTHONPATH="$PWD/src" result/onnx_gpu_20261002/.venv/bin/python \
  scripts/verify_refiner_gpu.py --stage verify \
  --onnx weights/mamba3-preview-20261002-best80/tracker_mamba3_step80.onnx \
  --out-dir result/onnx_gpu_20261002/best80
```

旧best200も、prepareのPTを `weights/mamba3-preview-20261001-step200/tracker_mamba3_daoff_best200.pt`、ONNXを `result/onnx_best200_cpu/tracker_mamba3_daoff_best200.onnx`、out-dirを `result/onnx_gpu_20261002/best200` に変更して同じ2段階で検証済みです。

[verify_refiner_gpu.py](../scripts/verify_refiner_gpu.py) は `use_tf32=0`、CUDA EP先頭指定で実行します。
provider一覧だけを見て成功とせず、初期化後にCUDAが先頭にあること、profileに実際のCUDA計算eventがあることを要求します。
主要演算のCPU割当とCPU浮動小数tensor処理を拒否し、整数の形状処理を別に記録します。
missing provider、CUDA初期化失敗、誤差超過は失敗です。CPU-only sessionでの成功をGPU合格として報告しません。

| チェック | 旧best200 | 新best80 |
| :--- | :--- | :--- |
| B/F/Nとdepth gridが違う4人工入力 | 全て成功 | 全て成功 |
| portable FP32とCUDA ORTの最大絶対差 | 5.722046e-6 | 9.059906e-6 |
| CPU ORTとCUDA ORTの最大絶対差 | 5.722046e-6 | 6.198883e-6 |
| CUDAのchunk1と一括の数値一致 | 成功 | 成功 |
| 固定rtol2e-4/atol5e-5 | 成功 | 成功 |
| 主要計算の実CUDA割当 | 確認済み | 確認済み |
| CPU側のtensor型 | int64のみ | int64のみ |

差の最大値は4出力を通した数値最大で、XYZ・UV・logitで単位が異なります。各出力別の値はJSONを参照してください。
各モデルの17回の実行（4一括+13点chunk）でCUDA **9,945 node-events**、CPU **1,496 node-events** が記録されました。
これは実行event数で、graphのunique node数ではありません。
MatMul/FusedMatMul、GridSample、LayerNormalization、Sin/Cos、ExpなどはCUDAでした。
CPUのConcat/Gather/Mul/Reshape/Slice/Transpose/Unsqueezeはint64の形状情報だけを扱います。**全node GPU-onlyではありません。**

起動時にplugin EP deviceの登録に関する警告が出ましたが、CUDA providerのactive状態とkernel profileで実行を確認しています。
警告を消すためにCPUへ切り替えたものではありません。
profile原本とfixtureは `result/onnx_gpu_20261002/` に保全し、公開reportにはprofile SHAと演算割当集計を含めています。
[best200_gpu_report.json](evidence/onnx_gpu_20261002/best200_gpu_report.json) と [best80_gpu_report.json](evidence/onnx_gpu_20261002/best80_gpu_report.json) が証跡です。

## 精度について確認したことと残ること

CPUとGPUの人工入力検証は、標準演算と配布runtimeの数値一致の確認です。
新best80では同一3本の全frames/all queriesでnative BF16とONNX CPU FP32を比較しましたが、validationから選んだ3本で独立testではありません。
詳細と指標が悪化したsubsetは [継続メモ](session_memory_20261002.md#同一動画のトラッキング比較) と [比較summary](evidence/tracking_comparison_20261002/summary.json) に記録しています。

追加の [CUDA ORT実動画検証](onnx_gpu_video_20261002.md) では、best80で同じ3本の全481 frames/all queriesをreplayしました。
GPU対CPUの全4出力が同じ固定閾値で成功し、各clipのmetric-AJは同値、3D-AJ最大差は7.51e-7でした。profileの主要演算CUDA/整数形状のみCPUも確認済みです。

**固定9本の受入gate、公式minival全150本、GPU性能benchmark、他GPU/他OSは未評価です。**
今回のGPU検証を、これらの合格や本番採用の証明として扱いません。
