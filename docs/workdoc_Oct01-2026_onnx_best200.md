# 作業計画書 兼 記録書：最終採用best200のONNX化と精度確認

**日付：** 2026年10月01日 21:42 JST（12:42 UTC）

**作業ディレクトリ・リポジトリ：** `vmamba3-3Dpointtracker`のroot（`git rev-parse --show-toplevel`で取得）

**作業者：** 作成・レビュー：Codex。実装・評価：引き継ぎ先エージェント。

**Git基準：** `main` / `fbaf7b205864e6966ae4bf3587d552316641e700`。この作業書のcommitはその後続。

**今回の依頼：** 作業書の作成・レビューと、再現に必要な試作コード・tests・依存設定・小さな検証記録のcommit・push。個人情報・認証情報を含めない。ONNXの追加実装や実データ評価再実行は次の担当者へ引き継ぐ。

**保存先について：** ユーザーの指定により`docs/`に保存。write-workdoc-uvの標準保存先`temp/`より、この指定を優先した。既存の実験資料は`doc/`にある。

## 1. 作業目的

### 1.1 ゴール要求分析

- **直観的な目的：** これまでの学習結果であるDAなしbest200を最終モデルとして使い、学習済み重みと追跡精度を維持したまま、ONNXで推論できるようにする。
- **明示要求：** best200を採用する。DAコードは残す。ONNX生成を試し、精度も確認する。ユーザーが挙げた実装・issueを参照する。別エージェントが作業できる具体的な作業書を作り、write/reviewスキルを適用してcommit・pushする。
- **追加で許容された選択肢：** ONNXが実用化できなければggmlでの自作も検討できる。その方式を採用する場合はpixiで依存を管理する。ggml方式の実装は必須ではなく、現時点ではONNXを優先する。
- **暗黙制約：** 同じcheckpoint・DINO・深度・flow・評価集合・可視性条件で比較する。DA学習を再開しない。元checkpointを変更しない。独自SSMや別のMamba世代への置き換えで「変換できた」としない。個人情報・トークンをログ・Gitへ保存しない。暗黙fallbackを行わない。DRY/KISS/SOLIDに従い評価処理の重複を抑え、挙動変更はt-wada式の失敗→成功を記録する。
- **成功条件：** 元重みの一致、ONNX checker、動的shape、ONNX Runtimeでの出力一致、同じ9clipでのAJ/絶対metric-AJ比較、品質ゲート、再現手順・制約の記録が揃う。詳細は6章。
- **非ゴール：** 再学習、DA追加学習、全150clipの論文再現、DINOv3/WAFT/DA3の一括ONNX化、INT8等の量子化、未検証のTensorRT対応、速度向上の保証。
- **リスク：** 元のMamba-3はTriton/BF16を使う。標準演算のFP32経路との丸め差がある。生成済みONNXは人工入力でのみ確認済みで、実データの指標は未確認。CPU用参照式はフレーム数に対して二乗のメモリを使う。

### 1.2 サブゴール構造

| ID | サブゴール | 目的との対応 | 成果物 | 検証方法 |
|---|---|---|---|---|
| SG-1 | 入力モデル・作業状態の固定 | best200の保全 | 出所・SHA256・Git/WIP台帳 | 元checkpointとGit状態の照合 |
| SG-2 | Mamba-3を保つ変換経路の確定 | ONNXで使う | 演算・入出力仕様、代替案の判断 | 公式参照式との比較 |
| SG-3 | ONNX生成と推論の実装 | 移植可能なモデル | exporter、ONNX、検証JSON | checker・CPU Runtime・動的shape |
| SG-4 | 元モデルとの精度比較 | 追跡精度の維持 | 同一9clipの比較JSON | AJ、絶対metric-AJ、OA、XYZ差 |
| SG-5 | 品質・再現性・公開状態の整備 | 他担当者が再現できる | tests、lock、docs、commit | ruff/ty/pytest、Git照合 |

### 1.3 トレーサビリティ方針

| Trace ID | 要求・制約 | 手順 | 必須の証跡 |
|---|---|---|---|
| TR-1 | best200を最終採用しDAを停止した状態を維持 | 1–3、25、29 | checkpoint hash、DA停止記録、最終モデルの出所 |
| TR-2 | 同一Mamba-3重み・式を使う | 4–6、10–11、15 | 参照式の版、全trainable重みの一致、単体試験 |
| TR-3 | ONNXとして推論できる | 11–12、16–17 | artifact hash、checker、入出力schema、動的shape試験 |
| TR-4 | 同一条件で精度を測る | 7–8、13–14、18–20 | per-clip比較、固定manifest、AJとmetric-AJ、誤差分布 |
| TR-5 | uv・品質ツール・監査記録 | 9、21–29 | lock、ruff/ty/pytest、作業記録、remote commit |
| TR-6 | 参照先を実装世代・演算差も含めて判断 | 4–6 | 根拠リンク、採用/不採用理由。ggmlならpixi仕様 |

## 2. 作業内容

### フェーズ0：計画・現物照合（目安15–30分、SG-1/TR-1/TR-5）

開始時刻とGit状態を記録し、既存WIPと生成物を特定する。このフェーズではコードを変更しない。
今回の公開commitには試作コードと依存設定を含める。別cloneには`result/`と学習データは存在しない。
コードは本書の公開commit以降を取得し、submoduleを初期化する。checkpointは2.2節の既存Releaseから取得する。
データ・深度が不在ならその事実を記録し、既存のデータ処理手順を使って準備する。ランダム重みで代用しない。

### フェーズ1：調査（目安30–60分、SG-2/TR-2/TR-6）

公式Mamba-3のforward/step参照式と、ユーザー提示の実装を読む。別モデルの重み・演算を
そのまま転用できるとは仮定しない。既知の失敗とCPUメモリ制約を照合する。実装はまだ変更しない。

### フェーズ2：設計の確定（目安15–30分、SG-2/TR-2/TR-3/TR-4）

既存試作の標準ONNX・FP32経路を第一候補とする。入出力、全点共通のz_ref、track chunk、
メトリクス、暫定採用基準、失敗時の扱いを記録する。TensorRT/ggmlが必要なら別方式として理由を記録する。

### フェーズ3：実装・TDD（目安1–3時間、SG-3/SG-4/TR-2–TR-5）

既知の評価不具合を先に失敗するテストで固定する。メトリクス計算と正規化を修正し、
変換・Runtime・paired評価の責務を整理する。trainingのTriton経路や既存評価の挙動は変更しない。

### フェーズ4：検証（目安30–90分、SG-3/SG-4/TR-3–TR-5）

演算単体→人工入力→同一実データ9clipの順で確認する。元モデルのBF16経路とONNX FP32の
差を別に記録する。実データ比較は途中で失敗したままなので、現時点で「精度維持」と結論しない。

### フェーズ5：記録・commit・push（目安15–30分、SG-5/TR-1/TR-5）

結果、採用条件、非対応、重み取得方法、操作コマンドをdocsへ記録する。コードとlockは検証後に
選択してcommitする。ONNX等のバイナリはGitに混入させない。Releaseの追加更新は、この作業書の必須作業に含めない。

### 2.1 リポジトリと実行環境

- Python 3.11–3.12、uv、PyTorch `2.12.1`、torchvision `0.27.1`。依存は`pyproject.toml`/`uv.lock`。
- GPU比較にはCUDAが必要。元`OfficialMamba3Adapter`はCPU実行を拒否する。ONNXの標準演算経路はCPU実行できる。
- GPU用に`source scripts/cudnn_env.sh`を使う。Hugging Face認証は`~/.bashrc`の設定を利用できるが、内容を表示しない。
- `AGENTS.md`、`CODEX.md`、`justfile`は本書作成時の対象root/親ディレクトリには存在しなかった。次担当者は開始時に再確認する。
- このworkspaceのデータは`~/data/tapvid3d`、深度は`~/data/tapvid3d_da3`。同じworkspaceでは追加ダウンロード不要。別環境では存在を確認し、必要な9clipと対応深度を準備する。
- 原データ＋深度＋cacheのユーザー予算は約120 GB。作成時のfilesystem空きは約60 GiB。
- cgroupの`memory.max`は`82999996416` bytes、約83 GB。ホストの総RAM表示を予算に使わない。

### 2.2 最終採用モデルと学習停止状態（TR-1）

| 項目 | 現物・値 |
|---|---|
| 最終モデル | DAなしのofficial-Mamba-3 tracker、best step200 |
| checkpoint | `result/v64_official_mamba3_cache_phase/best_200.pt` |
| ローカルcheckpoint SHA256 | `a004e267db0e12e09082682f5afaa02122a63eb425d4558f836f62bd0492af2b` |
| trainable parameters | 8,117,988 |
| DINO除外後のtracker state tensors | 50。公開PTにあるDINO前処理buffer等と数を混同しない |
| 固定15clipの検証loss | 0.10282776057720185 |
| 固定9clipの3D-AJ | 0.11774170602947298 = 11.7742% |
| 固定9clipの絶対metric-AJ | 0.24733599408801457 = 24.7336% |
| 固定9clipのOA | 0.6771292301191316 = 67.7129% |
| DAなし訓練の終了 | 300更新、Early stopping。最終採用はlatest300ではなくbest200 |
| DA追加学習の停止 | ユーザー指示で2026-10-01 12:21:07 UTCに停止。latest30を保全 |
| DA最後の検証loss | 0.10470321277777354。開始時bestを上回る改善なし |
| DA停止の証跡 | `result/v64_official_mamba3_da4_cached/user_stop.json` |

DAの`training_status.json`は最後の保存時の`reason: running`を保持している。
これだけで実行中と判断せず、`user_stop.json`と対象processの有無を照合する。
DAのconfig、sampler、cache生成コード、latest30、best_0は削除しない。DAを再開しない。

既存記録は[DAなし結果](../doc/training_result_20261001.json)、
[20更新時のDA検証記録](../doc/da4_validation_20261001.md)、
[データ処理手順](../doc/partitioned_data.md)。20更新記録は当時のsnapshotであり、現在の進行状態ではない。

公開PTは[既存Release](https://github.com/yuki-inaho/vmamba3-3Dpointtracker/releases/tag/mamba3-preview-20261001-step200)
の`tracker_mamba3_daoff_best200.pt`。32,490,569 bytes、SHA256
`bcddf045311b76a91bc772e35901acbdb99f397b752ee0ba4beacb3fb1ded5f4`。
公開PTとローカルcheckpointのファイルhashは異なる。optimizer/DINO backboneを除いた公開PTを使うときは、
tracker重みとcfgの一致を検証し、公開PTのファイルhashを別の出所として記録する。
ローカルcheckpointがない別cloneでの取得先はRelease assetであり、ランダム初期化で代用しない。
公開PTのtop-level keysは`cfg/export/model/step/weights_mode`、step200、model tensors52。
`dino.imagenet_mean/std`の2bufferを含み、凍結DINO backboneの重みは含まない。
exporterは`dino.*`除外で50 tensorsをロードできるが、paired評価のnative strict-loadは公開PTへ未対応。
別cloneでのpaired評価には、同じHF backboneを取得し、欠落をその凍結backboneに限定して検証するloader整備が必要。

### 2.3 引き継ぐ試作と公開対象（TR-5）

本書作成開始時、下記7ファイルがONNX関連WIPだった。**追加のユーザー指示により、7ファイルも本書と一緒にcommit・pushする。**
表のmodified/untrackedは公開前のsnapshot。これらは未完成の試作であり、Gitへ保存しても精度検証完了を意味しない。
同じworkspaceでも別cloneでも、この公開commitの試作から引き継げる。
勝手な`reset --hard`、`clean`、全ファイル`git add .`を使わない。

| 状態 | ファイル | 現在の責務 |
|---|---|---|
| modified | `pyproject.toml` | optional extra `onnx`、新コードのty/ruff設定 |
| modified | `uv.lock` | ONNX依存の解決結果 |
| modified | `scripts/check_training_quality.sh` | 新コードのruff対象追加。extra同期の問題は未修正 |
| untracked | `src/mamba3_tracker/model/onnx_refiner.py` | FP32・標準演算のMamba-3/V35 refiner |
| untracked | `scripts/export_refiner_onnx.py` | export、checker、動的shape、Runtime照合 |
| untracked | `scripts/eval_onnx_refiner.py` | 同じDINO/flow/depthによるpaired比較。既知の不具合あり |
| untracked | `tests/unit/test_onnx_refiner.py` | 公式recurrent参照式、causality、median、異常configの6ケース |

現物照合用のSHA256は下記。今回は依存設定の既存メール欄だけ除去し、試作の演算・評価コードは変更せず保存する。次担当者の作業による差は正常だが、開始時に差があれば記録する。

```text
abc4fb1c7f0ca4324780f3c0761e872a81810de6d541ffbf409e71a97e4fe71c  pyproject.toml
259ade098c242f8b200cb907030ee76fa6bc9d85c91ffccc3b448647b02ae075  uv.lock
010abc463df755562360df7f08ee2cc5b0a7725544f32918da9069f2efdc0d1b  scripts/check_training_quality.sh
29b4af83391461101739e30e31bac642ca8e40031479704a034f44f95eb1bb9b  scripts/export_refiner_onnx.py
3c38728404b06acd352d9391801edc2ef8c0f5b99373e7e24203b98add2c1ef9  scripts/eval_onnx_refiner.py
7d1469048bdee260abb22627e42dfb1c29d5b7cf5078c771f2c047604b09b94f  src/mamba3_tracker/model/onnx_refiner.py
e9e02efa18ce6f1072a8d8c3e828418287ecfd1c4cce8f7e073d7a361bbd2750  tests/unit/test_onnx_refiner.py
```

### 2.4 生成済みartifactと検証状況（TR-2/TR-3/TR-4）

| 証跡 | 状態 |
|---|---|
| `result/onnx_best200/direct_export_attempt.json` | 元CUDA adapterの直接exportは`TorchExportError` |
| `result/onnx_best200/direct_export_failure.txt` | 原因：FakeTensor/FunctionalTensorからTritonがdata pointerを取得しようとした |
| `result/onnx_best200/tracker_mamba3_daoff_best200.onnx` | 標準演算で生成済み。32,669,266 bytes、opset18、1,547 nodes |
| ONNX SHA256 | `0a8bcc72d17c32bb5e4000c8fb03a84f397e9a5314cde8bc1e64beac6bcf148a` |
| `result/onnx_best200/export_report.json` | checker成功、50 tracker tensorsをstrict loadしbitwise一致 |
| operator domain | 空文字のみ。custom CUDA/ATen domainなし |
| Runtime | onnxruntime1.30.0、CPUExecutionProvider |
| 人工入力のXYZ最大絶対差 | FP32 portableとの比較で最大5.7220458984375e-6 m |
| shape確認 | (B,F,N)=(1,1,1),(1,8,7),(2,13,3)、深度gridも変更 |
| 公式recurrent参照との単体照合 | 1/8/31 framesで最大約1.2e-6、6 unit tests成功 |
| 新コードのruff/ty | 本書公開前の再確認も両方成功。具体的な範囲と方法は7章に記録 |
| 全unit test / lock | 公開前確認で59 passed（3.38秒）、`uv lock --check`成功。品質launcher自体のextra問題は未修正 |
| 実データ9clip比較 | **未完了**。最初のclip後の集計でKeyError。`comparison.json`は存在しない |
| 実データ比較ログ | `temp/paired_onnx_reference.log` |

人工入力での数µm差は、portable FP32対ONNX FP32の結果であり、元CUDA/BF16モデルとの実データ精度差ではない。
ONNXは`result/`、ログは`temp/`のためGit対象外。次担当者はartifactの存在とhashを再確認する。

再現用の小さな証跡は`docs/evidence/onnx_best200/`にも保存する：
`export_report.json`、`direct_export_attempt.json`（ANSI装飾除去）、`da_user_stop.json`、`handoff_manifest.json`。
manifestには試作7ファイルのhash・サイズ、元JSONのhash、依存version、公開前品質確認を記録する。
raw動画・深度/cache・PT/ONNXバイナリ・認証設定・生のstack traceはGitへ入れず、入力の取得方法・hash・生成コマンドを残す。
公開PTは既存Releaseを使う。ONNXはexporterから再生成する。実データ評価が完了するまで正式ONNXモデルとは扱わない。

### 2.5 既知の未修正不具合（最優先、TR-4/TR-5）

1. **絶対metric-AJが未計算。** `scripts/eval_onnx_refiner.py`の`score()`は
   `compute_clip_metrics()`だけを呼ぶ。この戻り値に`metric_average_jaccard`はないため、
   最初のclipで`KeyError: 'metric_average_jaccard'`になる。
   `scripts/eval_metric3d.py`の既存処理と同じく`compute_clip_metrics_absolute()`も呼んでmergeする。
2. **median-scaled AJ用のintrinsicsスケールが不足。** 試作`score()`は元のKをそのまま渡している。
   既存評価は`intrin = [fx,fy,cx,cy] * 256/min(H_original,W_original)`を使う。
   この補正を揃えないと、とくにDriveTrackで「同じ評価」のつもりでも別の数値になる。
3. **optional extraが品質コマンドで落ち得る。** 試作は`onnx` extraを追加したが、
   `scripts/check_training_quality.sh`の`uv run`はextra指定なし。
   `uv sync/run`のexact同期でONNX関連依存が除かれるとtyのimportや評価が壊れる。
   quality/script実行のextra方針を明示的に統一する。
4. `_infer`の引数順を誤った最初の試行は修正済み。現在はkeyword呼出し。
   元関数の順序は`method, flow_model, model, clip, image_size, fb_alpha, fb_beta, da3_depth_root, max_frames, device`。
5. 真のMamba-3対応を検証せず、既存のMamba-1/2、VMamba、MambaVisionのscanに差し替えない。
   それはbest200の同一モデル変換という要求を満たさない。
6. **公開PTのpaired native loadingは未対応。** 試作は`native.load_state_dict(state['model'], strict=True)`を使う。
   公開PTでは凍結DINO backboneが欠落するため、この呼出しは失敗する。公開PTを使う場合は
   同じHF model/revisionのbackboneを復元し、tracker50 tensorsと前処理2bufferを全て照合する。
   `strict=False`だけで任意のmissing/unexpected keysを見逃さない。ローカル完全checkpointの場合は従来のstrict loadを維持する。

## 3. 作業チェックリスト

各操作は一つずつ実行する。結果は7章へ即時記録する。下記は次担当者の作業で、未実行の項目は未チェックのまま。

### フェーズ0：計画・現物照合

### 手順 1: 開始時のGit状態を記録する（SG-1/TR-1/TR-5）
- [ ] 🖐 **操作**: rootで`git status --short --branch`を実行する。
- [ ] 🔎 **確認**: mainの現在のcommit、本書、上記7ファイルのWIP有無を記録できる。
- [ ] 🧪 **テスト**: 読取専用。別担当者の変更を含む場合は所有・範囲を記録して区別する。
- [ ] 🛠 **エラー時対処**: repoが違う場合は`git rev-parse --show-toplevel`で確認し、上記rootへ移動する。resetで合わせない。

### 手順 2: 元checkpointを同定する（SG-1/TR-1）
- [ ] 🖐 **操作**: `sha256sum result/v64_official_mamba3_cache_phase/best_200.pt`を実行する。
- [ ] 🔎 **確認**: ローカル元hashが2.2節と一致。DA best_0やlatestを誤選択していない。
- [ ] 🧪 **テスト**: 出所が一致しないcheckpointは、ONNXとの比較対象にしない。
- [ ] 🛠 **エラー時対処**: 別cloneで不在なら公開Release PTを取得し、その別hash・cfg・tracker重みを記録してから進む。

### 手順 3: DA停止状態を確認する（SG-1/TR-1）
- [ ] 🖐 **操作**: `result/v64_official_mamba3_da4_cached/user_stop.json`を読む。
- [ ] 🔎 **確認**: user選定、保存step30、最終採用best200が記録されている。対象training processは停止している。
- [ ] 🧪 **テスト**: DAコードとcheckpointが保全されていることを確認する読取作業。学習コマンドは実行しない。
- [ ] 🛠 **エラー時対処**: 別cloneで停止記録が不在なら本書の履歴を記録する。processが見つかれば対象cmdlineを特定し、無関係なjobを止めない。

### フェーズ1：調査

### 手順 4: 公式Mamba-3の式を照合する（SG-2/TR-2）
- [ ] 🖐 **操作**: 4.1節のupstream module・forward reference・step referenceを読む。
- [ ] 🔎 **確認**: B/Cのnorm+bias、tanh角度、RoPE、DT、ADT、trap、D、Z、双方向平均の役割が記録される。
- [ ] 🧪 **テスト**: `test_attention_matches_official_recurrence`と`test_attention_is_causal_despite_shifted_trapezoid`を対応ケースとして特定する。
- [ ] 🛠 **エラー時対処**: submodule不在なら明示的にinitする。参照版が変わる場合は差分とrevisionを記録する。

### 手順 5: 提示された4参照先の適用範囲を整理する（SG-2/TR-6）
- [ ] 🖐 **操作**: 4.1節の参照リンクを読み、適用できる工夫とモデル世代の差を調査記録へ記す。
- [ ] 🔎 **確認**: CUDA制約、pure Torch化、parallel scan、ggmlのAPIとMamba-3固有要件を区別できる。
- [ ] 🧪 **テスト**: 調査のため自動テスト不要。直接exportの実ログを根拠に、ONNX全般が不可能とは結論しない。
- [ ] 🛠 **エラー時対処**: issueのコメントがHTMLに出なければGitHub APIのissue commentsを読む。憶測で作者回答を補わない。

### フェーズ2：設計

### 手順 6: 変換・精度比較の仕様を確定する（SG-2/TR-2/TR-3/TR-4/TR-6）
- [ ] 🖐 **操作**: 5章を現物に照合し、採用方式・入出力・閾値・chunk方針を本書に追記する。
- [ ] 🔎 **確認**: 標準ONNX FP32が第一候補で、速度/精度/メモリの制約が明示される。ggml採用ならpixiを含む別設計が必要。
- [ ] 🧪 **テスト**: 5.4節の暫定精度閾値を、検証前に固定して記録する。結果を見て閾値を緩めない。
- [ ] 🛠 **エラー時対処**: unsupported演算やメモリ不足の根拠を保存し、明示した別方式を設計する。無断のモデル差し替えはしない。

### フェーズ3：実装・TDD

### 手順 7: 評価不具合の回帰テストを追加する（SG-4/TR-4）
- [ ] 🖐 **操作**: `tests/unit/test_onnx_refiner_metrics.py`へ絶対metricキー・K補正・公開PTの欠落制限・chunk共通z_ref・共有visibilityの5回帰ケースを追加する。
- [ ] 🔎 **確認**: `test_score_contains_absolute_metric_aj`、`test_intrinsics_match_official_256_scaling`、`test_public_checkpoint_only_omits_frozen_backbone`、`test_chunked_inference_keeps_global_z_ref`、`test_paired_visibility_uses_same_flow_mask`が小fixtureとmockを使い独立検証できる。
- [ ] 🧪 **テスト**: 現在の試作scoreでは最初の2ケースがmissing key/K不一致で失敗する。公開PTケースはbackboneのみ欠落を許し、tracker欠落/余計なkeysを拒否する挙動を先に失敗させる。最後の2ケースは既存の正しい共有挙動を保全する。初回結果を記録する。
- [ ] 🛠 **エラー時対処**: GPU/HFがテストのimport時に必須ならscore責務を独立した軽量moduleへ分離する設計に戻す。

### 手順 8: scoreを既存公式評価と同じ計算へ修正する（SG-4/TR-4）
- [ ] 🖐 **操作**: `scripts/eval_onnx_refiner.py`のscoreへ公式median/absoluteの両計算と256px補正を実装する。
- [ ] 🔎 **確認**: 2.5節のKeyErrorとK条件不一致が解消し、per-clipでAJとmetric-AJが両方得られる。
- [ ] 🧪 **テスト**: 手順7の2ケースが失敗→成功へ変わる。既存`eval_metric3d.py`と同じfixtureで結果を照合する。
- [ ] 🛠 **エラー時対処**: `compute_clip_metrics_absolute()`を省かない。GTは(F,N,3)、predictionは(N,F,3)、visibilityは(N,F)を確認する。

### 手順 9: 品質launcherのextra指定を整合させる（SG-5/TR-5）
- [ ] 🖐 **操作**: `scripts/check_training_quality.sh`の各`uv run`へ`--extra onnx`を指定する。
- [ ] 🔎 **確認**: `uv run --extra onnx`/`uv sync --extra onnx`で必要依存を維持でき、通常training依存への無条件追加を避ける。
- [ ] 🧪 **テスト**: extra指定なしで依存が落ちる問題を再現可能に記録し、採用コマンドでty/importが成功する。
- [ ] 🛠 **エラー時対処**: extraがなければ本書と同じ公開commitのpyproject/lockを取得できたか照合する。異なる依存方式を採用するなら設計を更新してから修正する。

### 手順 10: portableの演算テストを整える（SG-3/TR-2）
- [ ] 🖐 **操作**: `tests/unit/test_onnx_refiner.py`の独立参照照合・因果性・median・異常configのテストを実装または精査する。
- [ ] 🔎 **確認**: upstream recurrent式を期待値に使い、portable式自身をコピーした期待値にしない。
- [ ] 🧪 **テスト**: 公開済みの6成功ケースを保全し、足りない境界ケースは失敗→成功を記録する。
- [ ] 🛠 **エラー時対処**: 初期SSM状態、trapのshift、RoPEのinterleaved組、GQAの展開を分けて調べる。

### 手順 11: portable refinerを整備する（SG-3/TR-2/TR-3）
- [ ] 🖐 **操作**: `src/mamba3_tracker/model/onnx_refiner.py`を5.1節の式・schemaに合わせて実装または修正する。
- [ ] 🔎 **確認**: strict loadの全50 tracker tensorsが一致し、DINO backbone等を入力特徴で置換する境界が明示される。
- [ ] 🧪 **テスト**: 手順10の参照式・因果性テストが成功する。state_dict key/shape欠落は失敗にする。
- [ ] 🛠 **エラー時対処**: shape変換で重みを切り詰めない。per-frame-scale/pose等の未対応設定は明示エラーにする。

### 手順 12: exporterを整備する（SG-3/TR-3）
- [ ] 🖐 **操作**: `scripts/export_refiner_onnx.py`のCLIへF>=128の長系列検証を追加し、入出力・source metadata・checker・hash・Runtime比較を整備する。
- [ ] 🔎 **確認**: 実際の入力checkpoint stepを記録し、step200固定の誤表示や仮定をしない。external dataの扱いも明示する。
- [ ] 🧪 **テスト**: checker不合格、missing input、壊れた重みは成功扱いにしない。B/F/Nとdepth shapeを変えて比較する。
- [ ] 🛠 **エラー時対処**: custom domainだけを出して成功扱いにしない。直接Triton export失敗は既知証跡を再利用できる。

### 手順 13: 公開PTのnative読込を整備する（SG-3/SG-4/TR-1/TR-2/TR-4）
- [ ] 🖐 **操作**: `scripts/eval_onnx_refiner.py`のnative loaderへ公開PTと完全checkpointの明示的な読込分岐を追加する。
- [ ] 🔎 **確認**: 公開PTは同じ凍結HF backboneを復元し、tracker50 tensorsと前処理2bufferを全て照合する。完全checkpointはstrict loadを維持する。
- [ ] 🧪 **テスト**: 手順7の`test_public_checkpoint_only_omits_frozen_backbone`が成功。tracker欠落/shape不一致/unexpected keysは失敗する。
- [ ] 🛠 **エラー時対処**: HF revisionがcfgに固定されていない場合は元checkpoint/cacheのbackbone fingerprintと取得revisionを照合して記録する。同一性未確認のままpaired精度を確定しない。

### 手順 14: paired比較の共有入力を整備する（SG-4/TR-4）
- [ ] 🖐 **操作**: `scripts/eval_onnx_refiner.py`の共有DINO/flow/depth・全点z_ref・track chunk・誤差統計を整備する。
- [ ] 🔎 **確認**: clip内のfront-end出力と可視性が両モデルで同一。chunkごとにmedianを再計算しない。
- [ ] 🧪 **テスト**: 手順7で追加した`test_chunked_inference_keeps_global_z_ref`と`test_paired_visibility_uses_same_flow_mask`を成功させる。
- [ ] 🛠 **エラー時対処**: 公開PTのDINO missing keysは手順13の修正を照合する。2回別々にflowを生成しない。複数DINO呼出しやmarker不足は明示エラーにする。

### フェーズ4：検証

### 手順 15: 演算単体テストを実行する（SG-3/TR-2）
- [ ] 🖐 **操作**: `uv run --extra onnx pytest -q tests/unit/test_onnx_refiner.py tests/unit/test_onnx_refiner_metrics.py`を実行する。
- [ ] 🔎 **確認**: upstream式、因果性、median、metricキー、K補正が全て成功する。
- [ ] 🧪 **テスト**: 失敗数0。CPU試験でCUDAカーネル実行に依存しない。
- [ ] 🛠 **エラー時対処**: 同じformulaを両側で使って無理に一致させない。FP32/FP64、DT/ADT、角度の型を照合する。

### 手順 16: ONNXを再生成する（SG-3/TR-3）
- [ ] 🖐 **操作**: 4.2節のexportコマンドを実行する。
- [ ] 🔎 **確認**: `export_report.json`のsuccess、step200、重み一致、動的shape比較が成立する。
- [ ] 🧪 **テスト**: FP32同士の比較はrtol2e-4/atol5e-5以内。5.4節の追加長時間窓も確認する。
- [ ] 🛠 **エラー時対処**: opcode未対応は名称とopsetを保存する。型・shapeを固定して見かけだけ通す変更は仕様に明記する。

### 手順 17: artifactを独立ロードして検査する（SG-3/TR-3）
- [ ] 🖐 **操作**: 4.3節の独立checker/CPU Runtime検査コマンドを実行する。
- [ ] 🔎 **確認**: input/output名、dtype、domain、dynamic dimension、hash、外部ファイルの要否が確定する。
- [ ] 🧪 **テスト**: exporterプロセスを終えても推論できる。custom CUDAライブラリのロードに依存しない。
- [ ] 🛠 **エラー時対処**: `.onnx.data`が必要な方式なら同梱とhashを明示し、単一ファイルと誤記しない。

### 手順 18: 固定reference集合を照合する（SG-4/TR-4）
- [ ] 🖐 **操作**: `configs/v64_metric_reference_minival.json`の9clipの存在・所属と元のreference記録を照合する。
- [ ] 🔎 **確認**: ADT/DriveTrack/PStudio各3本、合計9本。trainと混ぜず、frameを短縮していない。
- [ ] 🧪 **テスト**: missing clipや重複は評価開始前に検出する。
- [ ] 🛠 **エラー時対処**: clipが不在ならmanifestを変えず、不足を記録する。人工データを9clip評価として代用しない。

### 手順 19: 同一9clipのpaired評価を実行する（SG-4/TR-4）
- [ ] 🖐 **操作**: 評価不具合修正後に4.2節のpairedコマンドを実行する。
- [ ] 🔎 **確認**: 各subset3本、9/9成功、failures0、comparison.jsonが生成される。
- [ ] 🧪 **テスト**: native/ONNX双方のAJ、metric-AJ、OA、メートル誤差とXYZ差の分布が得られる。
- [ ] 🛠 **エラー時対処**: `KeyError: metric_average_jaccard`は手順8の修正漏れ。OOMなら点のchunkを減らし、全点z_refは維持する。

### 手順 20: 精度差を採用基準に照合する（SG-4/TR-4）
- [ ] 🖐 **操作**: native再評価と既存baseline、ONNXとnativeの差を5.4節の基準で判定する。
- [ ] 🔎 **確認**: baselineとの差、変換による差、subset別の低下が区別される。
- [ ] 🧪 **テスト**: 暫定基準外を成功にしない。全体平均で悪いsubsetを隠さない。
- [ ] 🛠 **エラー時対処**: baseline再現に失敗したらK/可視性/flow.scale/iters/DINO/sourceを調査し、ONNXの精度差として扱わない。

### 手順 21: ruffを実行する（SG-5/TR-5）
- [ ] 🖐 **操作**: 新規ONNX関連module・scripts・testsへ`uv run --extra onnx ruff check`を実行する。
- [ ] 🔎 **確認**: 対象一覧を記録し、error0。
- [ ] 🧪 **テスト**: jaxtypingのshape文字列に必要なF722除外は当該moduleだけに限定する。
- [ ] 🛠 **エラー時対処**: repo全体の既存違反を混ぜず、今回の対象と既存品質対象を分けて照合する。

### 手順 22: tyを実行する（SG-5/TR-5）
- [ ] 🖐 **操作**: `uv run --extra onnx ty check`を実行する。
- [ ] 🔎 **確認**: 新コードをincludeした設定でdiagnostic0。
- [ ] 🧪 **テスト**: ONNX Runtimeのdense出力を型検査し、heterogeneous reportはTypedDict等で扱う。
- [ ] 🛠 **エラー時対処**: unresolved-importならextra同期を確認する。広いignoreやAnyで実際の型不一致を隠さない。

### 手順 23: 全unit testを実行する（SG-5/TR-5）
- [ ] 🖐 **操作**: `uv run --extra onnx pytest -q`を実行する。
- [ ] 🔎 **確認**: 元53ケース、新6ケース、今回追加ケースが成功。skipと理由も記録する。
- [ ] 🧪 **テスト**: fixed DAのsampler/cacheと既存profiling経路も回帰しない。
- [ ] 🛠 **エラー時対処**: 原因と再現入力を特定し、既存テストを消して成功にしない。

### 手順 24: lock整合性を確認する（SG-5/TR-5）
- [ ] 🖐 **操作**: `uv lock --check`を実行する。
- [ ] 🔎 **確認**: pyproject/lockが整合し、再同期でONNX依存を維持できる。
- [ ] 🧪 **テスト**: lockの不整合は完了扱いにしない。
- [ ] 🛠 **エラー時対処**: 依存方針に合わせてlockを再生成し、CUDA stack等の無関係なversion変更を点検する。

### フェーズ5：記録・commit・push

### 手順 25: 測定結果と再現手順をdocsへ記録する（SG-5/TR-1/TR-3/TR-4/TR-5）
- [ ] 🖐 **操作**: 実行済みexportとpaired比較の数値・hash・制約・入力準備・コマンドを結果文書へ保存する。
- [ ] 🔎 **確認**: best200の最終選定、DA保全、ONNXの評価範囲が明記され、途中結果と最終結果を区別できる。
- [ ] 🧪 **テスト**: JSONと表の数値を照合する。9/150結果を論文150clipの到達率として表示しない。
- [ ] 🛠 **エラー時対処**: 精度未達や未対応backendはそのまま記録し、未検証の性能値を記入しない。

### 手順 26: 検証済みの変更だけstageする（SG-5/TR-5）
- [ ] 🖐 **操作**: 実際の変更ファイルを列挙した`git add`でコード・tests・lock・docsをstageする。
- [ ] 🔎 **確認**: `git diff --cached --stat`にPT/ONNX/data/tempや他担当者の変更を含めない。秘密・個人情報の確認では値を出力せず、該当pathと件数だけを記録する。
- [ ] 🧪 **テスト**: `git diff --cached --check`が成功する。
- [ ] 🛠 **エラー時対処**: 余計なstageがあればそのpathのみunstageする。既存WIPを削除しない。

### 手順 27: 変更をcommitする（SG-5/TR-5）
- [ ] 🖐 **操作**: 検証範囲と最終挙動がわかるmessageでcommitする。
- [ ] 🔎 **確認**: commit内容と検証したtreeが一致する。
- [ ] 🧪 **テスト**: commit後に対象の差分と親commitを確認する。
- [ ] 🛠 **エラー時対処**: hook失敗は内容を修正して再確認する。無条件のhook無効化をしない。

### 手順 28: fork/mainへpushする（SG-5/TR-5）
- [ ] 🖐 **操作**: `git push fork main`を実行する。
- [ ] 🔎 **確認**: push先は`yuki-inaho/vmamba3-3Dpointtracker`。origin作者repoへはpushしない。
- [ ] 🧪 **テスト**: forceを使わず、親からのfast-forwardが成立する。
- [ ] 🛠 **エラー時対処**: CLI認証は過去に401、originは403だった。認証済みGitHub Appによる同じtreeのcommit/ref更新が利用可能なら明示的に使う。認証情報を出力しない。

### 手順 29: remoteと作業状態を照合する（SG-5/TR-1/TR-5）
- [ ] 🖐 **操作**: remote main、ローカルHEAD、未commit差分、保存したartifact hashを最終照合する。
- [ ] 🔎 **確認**: publishしたcommitが一致し、残WIPの有無と範囲を報告できる。DA jobは再開していない。
- [ ] 🧪 **テスト**: 未commitの必要ファイルがある場合は「push漏れなし」としない。
- [ ] 🛠 **エラー時対処**: 他担当者の変更は触らず、公開済み範囲と残作業を分けて報告する。

## 4. 作業に使用するコマンド参考情報

全てrepo rootから実行。以下のONNXコマンドはextra/各scriptを実装・修正した後に使う。

### 4.1 参照元と読み方（TR-2/TR-6）

| 参照 | 観測した内容と適用範囲 |
|---|---|
| [VMamba_onnx](https://github.com/HaoKun-Li/VMamba_onnx) / [run.sh](https://github.com/HaoKun-Li/VMamba_onnx/blob/main/run.sh) | CUDA scanをpure Torch化し、ONNXの小モデルで差分を見てから全体を評価する方法。対象はVMambaで、このMamba-3のcheckpointをそのままロードする実装ではない |
| [MambaVision issue110の作者側コメント](https://github.com/NVlabs/MambaVision/issues/110#issuecomment-4042007183) | custom CUDA/Tritonのscanが標準exportを妨げるという回答。元の演算をそのままexportする制約と、標準演算へ移植する可否を分けて考える |
| [ggml](https://github.com/ggml-org/ggml) / [ggml_ssm_scan API](https://github.com/ggml-org/ggml/blob/master/include/ggml.h) | SSM scan APIはあるがMamba-3のRoPE・trap・data-dependent AまでこのAPIだけで対応するとは確認していない。自作時は未対応演算を調査しpixiで依存を固定する |
| [mamba.py](https://github.com/alxndrTL/mamba.py) | pure Torch、parallel scan、ONNX例が参考になる。READMEの対象はMambaとMamba-2。Mamba-3固有式の代用品とはしない |

実際のMamba-3参照元は、local submodule revision
`e9594ce1c732d97440f0332fdc43170a2294dbfa`の以下。

- `third_party/visionMamba3/third_party/mamba-ssm/mamba_ssm/modules/mamba3.py`
- `third_party/visionMamba3/third_party/mamba-ssm/tests/ops/triton/test_mamba3_siso.py`
- 期待値に使う関数：`mamba3_siso_step_ref`、forwardの照合：`mamba3_siso_fwd_ref`。
- trackerの双方向・射影：`src/mamba3_tracker/model/official_mamba3.py`。
- 元のV35入出力：`src/mamba3_tracker/model/depth_refined_tracker.py`。
- 評価仕様：`scripts/eval_metric3d.py`、`src/mamba3_tracker/eval/tapvid3d_eval.py`。

### 4.2 実行コマンド（TR-1–TR-5）

```bash
date '+%Y-%m-%d %H:%M:%S %Z%z'
git status --short --branch
sha256sum result/v64_official_mamba3_cache_phase/best_200.pt

# extraが存在することを確認してから同期。通常のuv syncだけではextraは維持されない。
uv sync --extra onnx

# 既知の評価不具合を修正してから実行。
uv run --extra onnx pytest -q tests/unit/test_onnx_refiner.py tests/unit/test_onnx_refiner_metrics.py

uv run --extra onnx python scripts/export_refiner_onnx.py \
  --ckpt result/v64_official_mamba3_cache_phase/best_200.pt \
  --out-dir result/onnx_best200

# GPU比較とHF認証の環境を設定。トークン内容は表示しない。
bash -ic '. scripts/cudnn_env.sh && uv run --extra onnx python scripts/eval_onnx_refiner.py \
  --ckpt result/v64_official_mamba3_cache_phase/best_200.pt \
  --onnx result/onnx_best200/tracker_mamba3_daoff_best200.onnx \
  --manifest configs/v64_metric_reference_minival.json \
  --out-dir result/onnx_best200/paired_reference \
  --track-chunk 32 --threads 8'

uv run --extra onnx ruff check src/mamba3_tracker/model/onnx_refiner.py \
  scripts/export_refiner_onnx.py scripts/eval_onnx_refiner.py tests/unit/test_onnx_refiner.py \
  tests/unit/test_onnx_refiner_metrics.py
uv run --extra onnx ty check
uv run --extra onnx pytest -q
uv lock --check
git diff --check
```

`scripts/check_training_quality.sh`は手順9でextra方針を直してから使う。
本書作成時のoptional依存はonnx1.23.1 / onnxruntime1.30.0 / onnxscript0.7.2。
コマンド実行時のstdout/stderrを`temp/`に保存し、exit codeと該当Trace IDを7章へ記録する。
shell操作には意味のある依存順以外のコマンド連結を使わず、秘密情報を含む環境変数をdumpしない。

### 4.3 artifactの独立検査（手順17、TR-3）

exporterとtracker moduleをimportせず、別プロセスからchecker・schema・CPU推論を確認する。
この入力は動作確認用であり、精度の期待値ではない。ONNXのmetadataにあるstep/hashも元checkpointと照合する。

```bash
uv run --extra onnx python - <<'PY'
import hashlib
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort

path = Path('result/onnx_best200/tracker_mamba3_daoff_best200.onnx')
graph = onnx.load(path)
onnx.checker.check_model(graph, full_check=True)
assert {node.domain for node in graph.graph.node} == {''}
assert all(tensor.data_location != onnx.TensorProto.EXTERNAL for tensor in graph.graph.initializer)
session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
inputs = ['ray', 'z_raw', 'visibility', 'uv', 'depth_map', 'dino_features', 'intrinsics', 'z_ref']
outputs = ['xyz', 'uv_refined', 'vis_logits', 'delta_uv']
assert [item.name for item in session.get_inputs()] == inputs
assert [item.name for item in session.get_outputs()] == outputs
assert all(item.type == 'tensor(float)' for item in session.get_inputs() + session.get_outputs())
for item in session.get_inputs():
    if item.name != 'z_ref':
        assert isinstance(item.shape[0], str)
    if item.name not in ('intrinsics', 'z_ref'):
        assert isinstance(item.shape[1], str)
feed = {
    'ray': np.zeros((1, 1, 1, 2), dtype=np.float32),
    'z_raw': np.full((1, 1, 1), 3, dtype=np.float32),
    'visibility': np.ones((1, 1, 1), dtype=np.float32),
    'uv': np.full((1, 1, 1, 2), 448, dtype=np.float32),
    'depth_map': np.full((1, 1, 24, 32), 3, dtype=np.float32),
    'dino_features': np.zeros((1, 1, 384, 28, 28), dtype=np.float32),
    'intrinsics': np.array([[[896, 0, 448], [0, 896, 448], [0, 0, 1]]], dtype=np.float32),
    'z_ref': np.array(3 + 1e-6, dtype=np.float32),
}
observed = session.run(outputs, feed)
assert [value.shape for value in observed] == [(1, 1, 1, 3), (1, 1, 1, 2), (1, 1, 1), (1, 1, 1, 2)]
assert all(np.isfinite(value).all() for value in observed)
print('sha256:', hashlib.sha256(path.read_bytes()).hexdigest())
print('metadata:', {item.key: item.value for item in graph.metadata_props})
print('schema:', [(item.name, item.type, item.shape) for item in session.get_inputs()])
print('independent_cpu_runtime: success')
PY
```

## 5. 注意点・制約と変換仕様

### 5.1 演算を保つ要件（TR-2）

元mixerはmodel幅768、inner1536、state128、24 heads×64、GQA group1、RoPE fraction0.5。
tracker幅128からup/downで接続する。入力は公式順の
`[z,x,B,C,dd_dt,dd_A,trap,angle]`、幅は`[1536,1536,128,128,24,24,24,32]`。

最低限、次の計算を落とさない。

```text
DT = softplus(dd_dt + dt_bias)
A = -heavy_tail_activation(dd_A), clamp(max=-1e-4)
ADT = A * DT
Q = RMSNorm(C) + C_bias, K = RMSNorm(B) + B_bias
angle_state = cumsum(pi*tanh(angle)*DT) modulo 2pi
Q/Kを隣接2要素組でrotate。stateの後半64要素はrotateしない。
trap_gate = sigmoid(trap)
shifted_gamma[t] = DT[t+1]*(1-trap_gate[t+1]), 最後は0
scale[t] = DT[t]*trap_gate[t] + shifted_gamma[t]
causal_decay[t,s] = exp(sum(ADT[s+1:t+1])) for s<=t
y[t] = sum_s dot(Q_rot[t],K_rot[s]*scale[s])*causal_decay[t,s]*V[s]
y[t] += D*V[t] - dot(Q[t],K[t])*shifted_gamma[t]*V[t]
y[t] *= silu(Z[t])
out = out_proj(y)
```

双方向は同一重みのforwardとreverseの平均。元の残差、LayerNorm、深度patch、DINO射影、
Δuv制限、深度再sample、Δlog_z制限、KによるXYZ復元、vis_headを保つ。
FP32式の変更をtraining側へ反映しない。RMSNorm epsは1e-5。

### 5.2 入出力契約（TR-3）

全入力はFP32。B=batch、F=frames、N=tracks。image_sizeは896 pixel座標。

| input名 | shape | 意味 |
|---|---|---|
| ray | (B,F,N,2) | Kで正規化したpixel ray |
| z_raw | (B,F,N) | 元uvでsampleしたDA3 metric depth |
| visibility | (B,F,N) | 元WAFTのFB mask、0/1 |
| uv | (B,F,N,2) | 896px座標の元2D track |
| depth_map | (B,F,Hd,Wd) | 既存DA3 metric depth map |
| dino_features | (B,F,384,28,28) | DINOv3 ViT-S/16・448入力の凍結feature map。trainable feat_projより前 |
| intrinsics | (B,3,3) | 896画像へリサイズしたK |
| z_ref | scalar | **入力の全B/F/Nのz_rawのlower median＋1e-6**。chunk分割前に一度計算 |

出力は`xyz (B,F,N,3)`、`uv_refined (B,F,N,2)`、`vis_logits (B,F,N)`、`delta_uv (B,F,N,2)`。
ログのpos_3D/lossはメートル誤差でもAJでもない。現状評価のvisibilityはflow maskなので、
vis_headを学習済みの正解率モデルとして扱わない。

### 5.3 メモリ・backend・scope（TR-3/TR-6）

- このONNXはrefinerだけ。RGB動画から直接XYZを返す一体モデルではない。DINO/WAFT/DA3は別途必要。
- DINOv3のgated backboneをONNXへ同梱していない。別環境で利用するときも元の同意・取得条件を守る。
- 標準演算の参照式はO(F²)。元Triton recurrent/scanの線形メモリ特性や速度をそのまま主張しない。
- trackは独立なのでN方向のchunkは可能。frame方向を分割して状態をリセットすると別の推論になる。
- CPU Runtimeで確認したことと、ONNX Runtime CUDA、TensorRT、ggml対応を区別する。
- ggmlの`ggml_ssm_scan`があるという理由だけでMamba-3対応とは言わない。採用する場合は
  RoPE、trap、data-dependent A、norm、GridSample、GELU等の対応とweight形式を別途設計し、
  `pixi.toml`/lock/CPU・GPU build手順を確定する。既存uv training環境を暗黙に移行しない。
- INT8/BF16/F16量子化やmixer縮小を同時に行うと変換差の原因が分からなくなる。まず同じ重みのFP32で比較する。

### 5.4 精度検証と暫定採用基準（TR-4）

ユーザーは許容誤差を数値指定していない。以下を**実装担当の暫定基準**として開始前に固定し、
結果を見て緩めない。数値基準外でも出力を隠さず、理由・差分・未達事項を記録する。

1. 元checkpointから読んだtrainable tracker tensorsは全てkey/shape/value一致。strict loadを使う。
2. portable FP32とONNX FP32は有限値で、rtol2e-4 / atol5e-5以内。B=1/2、F=1/8/13だけでなく
   **F=128以上の長い系列**、異なるN、深度gridを追加検査する。N chunkの前後も比較する。
3. paired nativeは同じGPU・元Triton・元DINO/flow/depth/可視性で測る。ONNXは共有入力を使う。
   このnative値と既存9clip baselineのAJ/metric-AJが絶対値1e-6を超えて異なるなら、
   まず条件差を調査し、ONNXの劣化と混同しない。
4. 同じ9clipで、**全体と各subsetのAJ・絶対metric-AJの低下が0.1 percentage point以内**
   （0–1値でnative−ONNX<=0.001）を暫定採用条件にする。
5. flow可視性を共有するため、OAはabsolute差1e-10以内で一致すること。oracle visibilityは使わない。
6. 可視点XYZ差のmean/p50/p95/max、UV差、元とONNXのメートル誤差を保存する。
   遠距離DriveTrackと近距離ADT/PStudioを混ぜた一つの誤差数値だけで判断しない。
7. 9本全てのmembership・全frame・failures0が揃ってから結論を出す。
   この9/150監視集合と、論文の全150clipの25.6%を直接比較した到達率は出さない。

## 6. 完了の定義

**この作業書の作成・引き継ぎ完了**と、**次担当者のONNX作業完了**を分ける。

作業書に関する完了項目：

- docs以下に本書が存在し、write/reviewスキルのtemplate/rubric照合記録が7章にある。
- 試作7ファイル、既知の失敗、artifact hash、DA停止・最終best200の状態が明示され、個人情報・認証情報を除いた再現用コード・依存設定・小さな証跡がstageされる。
- 今回の公開treeとローカルstageが一致し、fork/mainへの反映を検証する。自身のcommit hashは本書へ循環して書き込まず、公開後の最終応答で報告する。

次担当者が満たすONNX作業の完了項目：

- [ ] TR-1：best200のsource hash/cfgと全tracker重みを照合。DAは停止のまま、コードとcheckpointは保全。
- [ ] TR-2：upstream Mamba-3参照式、causality、長系列、chunk共通z_refの回帰試験が成功。
- [ ] TR-3：ONNX checker、標準domain、入出力・dtype・dynamic shape、独立Runtimeロード、hashが記録済み。
- [ ] TR-4：同じ9clip・全frame・failures0で、native/ONNXのAJ、metric-AJ、OA、XYZ/UV差を出力。
- [ ] TR-4：baseline再現と5.4節の全体/subset暫定採用条件を満たす。未達なら「未達」と報告し完了扱いにしない。
- [ ] TR-5：必要なextraを維持したuv環境でruff/ty/全pytest/lockが成功。beartype/jaxtypingは入力境界で利用。
- [ ] TR-5：結果文書と再現手順がGitにあり、必要なコード・tests・lock・docsがcommit/push済み。バイナリ/dataは除外。
- [ ] TR-6：ユーザー提示の4参照先の適用範囲と選択理由を記録。ggmlを選んだ場合はpixi依存・追加演算・精度基準も整備。

## 7. 作業記録

**重要な注意事項：**

* 作業開始前に必ず`date "+%Y-%m-%d %H:%M:%S %Z%z"`で現在時刻を確認し、正確な日時を記録する。
* 各作業項目を開始する際と完了する際の両方で記録する。
* 作業内容は具体的なコマンドや操作手順を詳細に記載する。
* 結果・備考欄には成功／失敗、エラー内容、解決方法、重要な気づきを必ず記入する。
* 複数フェーズがある場合はフェーズごとに開始・完了の記録を取る。
* コード変更を行った場合は変更したファイル名と内容の概要を記録する。
* エラーが発生した場合はエラーメッセージと解決策を詳細に記録する。

| 日付 | 時刻 | 作業者 | 作業内容 | 結果・備考 |
|---|---|---|---|---|
| 2026-10-01 | 12:21:07 UTC | Codex | DA段階の停止、TR-1 | user指示でbest200最終採用。DA latest30/コード保全。停止記録user_stop.json |
| 2026-10-01 | 作成前の試行、ログ参照 | Codex | 直接Triton export、TR-2 | 失敗。FakeTensor data pointer/TorchExportError。direct_export_failure.txtに保存 |
| 2026-10-01 | 作成前の試行、ログ参照 | Codex | 標準ONNX export、TR-3 | 成功。32,669,266 bytes。checker・人工3shape Runtime一致。実データ精度は未確認 |
| 2026-10-01 | 作成前の試行、ログ参照 | Codex | 演算unit/ruff/ty、TR-2/TR-5 | 新6ケース、ruff、tyは成功。全pytest/lockは未検証 |
| 2026-10-01 | 12:41 UTCの現物確認 | Codex | paired評価の失敗、TR-4 | KeyError: metric_average_jaccard。scoreのabsolute計算とK補正が未実装。comparison.jsonなし |
| 2026-10-01 | 21:42:18 JST | Codex | 作業書作成フェーズ開始、TR-5 | write-workdoc-uvとreview-written-workdoc、およびtemplate/rubric読了。ユーザー指定docs保存 |
| 2026-10-01 | 21:57:39 JST | Codex | 作業書レビュー開始、TR-5 | review rubricと全章を照合。公開範囲、独立検査コマンド、別cloneの読込条件を調査 |
| 2026-10-01 | 追記指示受領時、秒未記録 | Codex | 公開範囲の更新、TR-5 | user指示で試作7ファイル・tests・依存lockも公開対象へ追加。docだけの公開という旧方針を更新 |
| 2026-10-01 | 21:57–22:01 JST、区間確認 | Codex | 公開前品質確認、TR-5 | `uv run --no-sync --extra onnx`で既存quality対象＋新ONNX対象のruff、ty、全pytestを実行。ruff/ty成功、59 passed in 3.38s。`uv lock --check`も成功。no-syncで既存環境を維持。quality launcher自体のextra問題は未修正 |
| 2026-10-01 | 22:04:01 JST | Codex | 再現用JSONの保存、TR-1/TR-3/TR-5 | docs/evidence/onnx_best200へ4 JSON。hash・依存・品質結果を記録、例外のANSI装飾だけ除去。作業用JSON copyの最初の`python`呼出しは不在で失敗し、`python3`で成功 |
| 2026-10-01 | 22:09:10 JST | Codex | レビュー修正・秘匿確認完了、TR-5 | 29手順の連番・4欄、相対リンク、独立検査コマンド構文、13件のhash照合が成功。公開12ファイルの認証値/URL埋込資格情報/メール/個人home pathパターン0件。pyprojectの既存メール欄を除去してhash更新、lock再確認成功 |
| 2026-10-01 | 22:11:18 JST | Codex | stage差分の修正、TR-5 | 最初の`git diff --cached --check`は本書先頭のMarkdown改行用行末空白で失敗。空白を除去し、段落改行に変更。再stage後に最終チェックする |

### 7.1 作業書レビュー

**Verdict:** PASS（初回REVISEからreview-and-fixで修正）。これは作業書の実行可能性の判定であり、ONNXの実データ精度が検証完了したという判定ではない。

**Mode:** review-and-fix。

**Findings（全て修正済み）**

- Major — 本書:129、2.3節：初稿は文書だけを公開する前提だった。追加指示に合わせ、試作7ファイルと再現JSONも公開対象へ変更し、別cloneで使えるコードと別途必要なdata/checkpointを区別した。
- Major — 本書:318、手順17：独立Runtime検査のコマンドが具体化されていなかった。4.3節へ実行可能なchecker/schema/CPU推論コマンドを追加した。ここでは構文確認までで、コマンドを新たに実行したとは記録しない。
- Major — 2.2/2.5節、手順13：公開PTは凍結DINO重みを含まず、paired nativeのstrict loadにそのまま渡せない。公開PTのkeys・52 tensorsを現物確認し、制限・読込実装手順・欠落制限の回帰ケースを追加した。
- Minor — 本書:268、手順9：依存設定と品質launcherの編集が一手順に混在していた。extraは公開済みpyproject/lockを使い、手順9の操作をlauncherだけの変更に限定した。
- Minor — 本書:256、手順7/14：chunk共通z_refと共有visibilityの回帰ケースをどこで追加するか曖昧だった。追加先・ケース名・初回結果の扱いを手順7へ明記した。

**Applied Changes**

- ゴール・公開範囲・別cloneの準備条件・現在の検証実績を更新。原データ/秘密設定/生ログは公開せず、小さな証跡とhash・取得/生成方法を保存。
- 公開PT用のloader手順を独立し、29手順へ連番化してTrace IDの対応を更新。
- F>=128の追加検証、独立Runtimeコマンド、具体的な回帰ケースとTDDを補強。
- 公開対象pyprojectの既存メール欄を除去。公開された著作者表記とライセンスは保持。

**Residual Findings**

- 作業書を開始できなくするBlocker/Majorはなし。試作の未完成事項は2.5節に明示し、次担当者の未チェック項目として残した。

**Coverage Notes**

- ゴール要求分析: adequate。
- サブゴールと作業要素の対応: adequate（SG-1–5、TR-1–6）。
- 完了の定義: adequate（文書公開と次担当者のONNX検証を別に判定）。
- チェックリスト原子性: adequate（29手順、各操作は一つの調査・ファイル編集・コマンド実行・照合に限定）。
- TDD/検証可能性: adequate（upstream参照、5回帰ケース、長系列、9clip、ruff/ty/pytest/lock）。
- エラー時対処: adequate（source不在・extra・metric key・K補正・DINO欠落・OOM・push認証）。
- トレーサビリティ: adequate（全Trace IDに成果物・hash・手順・DoDを対応）。

**Open Questions**

- ユーザー判断がないと文書を公開できない項目はなし。採用閾値は5.4節の暫定設計として明示。DINOの取得revisionとfingerprintは次担当者が手順13で現物から確定する。

**Recommended Patch Scope**

- 作業書の必須修正はなし。次担当者は3章の未チェック項目に従い、既知不具合と精度比較を進める。

### 7.2 公開時の照合方法

今回の公開対象は試作7ファイル、本書、証跡4 JSONの計12ファイル。
`git diff --cached --check`、stage内容とmanifestの一致、remote mainとローカルHEADの一致を公開時に確認する。
CLI認証に制約がある場合は、認証済みGitHub Appで同じblob/treeを作り、forceなしのref更新を使う。
最終commitのSHAとremote照合結果は最終応答に記載する。未確認の公開成功を本書で先に宣言しない。

## 8. 次担当者による実装記録（2026-10-01）

ONNX／C++ CPU実装、厳格な入力・重み検査、paired評価不具合の修正、回帰試験、uv軽量環境を追加しました。
詳細・コマンド・検証JSONは [best200 CPU実装記録](onnx_best200_cpu_20261001.md) を参照してください。
**TR-4の固定9動画・元CUDA/BF16との実測比較は未達であり、6章の全DoD完了とは扱いません。**
