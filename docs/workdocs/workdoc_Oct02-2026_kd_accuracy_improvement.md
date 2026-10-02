# KD精度改善の作業計画書 兼 記録書

**日付：** 2026年10月2日、20:00:06 JST+0900（11:00:06 UTC）。
**作業ディレクトリ・リポジトリ：** `/home/kasm-user/Desktop/vmamba3-3Dpointtracker`、yuki-inaho/vmamba3-3Dpointtracker。
**作業者：** Codex root、並列research/worker/audit agents。
**ブランチ：** `distill/refiner-kd-onnx-20261002`。未commitの既存成果は保持する。
**親作業書：** `temp/workdoc_Oct02-2026_refiner_kd_onnx.md`。本書は精度改善の追加実験計画であり、親D-1〜D-8・全150本受入判定を置き換えない。本書の完了だけで全体目標をcompleteにしない。

## 1. 作業目的

### 1.1 ゴール要求分析

ユーザーは、現在の学びを用いてKD／Vision Mamba／軽量Mamba／圧縮研究を詳細に調べ、実際の精度改善へつなげることを求めている。有効な根拠があれば現学習を停止し、構造・損失・カリキュラムを改善して仕上げる許可がある。

- 明示要求：一次論文の詳細調査、改善実装と比較訓練、ONNX GPU許可、32GB GPU使用可、write/review/start-with-workdocs。
- 成功条件：研究事実と適用仮説を分離した資料、GTと整合する新query、未使用projection除去の数値同等性、650k以下の時間局所＋大域student、段階的蒸留とGT-only対照、全F monitorでの採否、親の最終ONNX/150本AJ判定へ接続。
- 暗黙制約：uv、TDD、DRY/KISS/SOLID、明示的失敗、hashによる追跡。teacher・旧cache・旧checkpointは無変更。minivalでhyperparameterを選ばない。閾値を結果に合わせて緩和しない。資格情報は表示しない。
- 非ゴール：Release更新、commit/push/main統合、DINO/WAFT/DA3圧縮、量子化でFP32精度差を隠すこと。論文の分類・LLM性能をtracking性能の保証と扱わない。
- リスク：現teacherも学習query問題を継承している可能性があり、無条件KDは不適切。normalized poolやlocal branchの精度改善は未保証。GT query生成はtrain/monitor windowだけで、推論入力へGTを混入しない。

### 1.2 サブゴール構造

| ID | サブゴール | 成果物 | 検証 |
| :--- | :--- | :--- | :--- |
| ISG-1 | 一次研究と現学習の原因候補 | `docs/kd_accuracy_research_20261002.md` | 論文URL、式・ablation・限界、コード所見 |
| ISG-2 | 正しいanchorの新DA無し入力 | `result/refiner_kd_improved_20261002/` manifest/audit | 可視finite正depth・bounds、再投影誤差、track増加 |
| ISG-3 | 有効容量を時間表現へ再配分 | minimal/local-global modules/tests | pruning parity、parameter count、fusion、N permutation |
| ISG-4 | 段階的KDを同条件比較 | staged/GT-only pilots、gradient/gate logs | freeze、同teacher入力、finite、resume、全F monitor |
| ISG-5 | 親受入条件へ接続 | 選択記録・親workdoc更新 | minival未参照選択、最終GPU/CPU/150本DoD維持 |

### 1.3 トレーサビリティ方針

I-TR1=ISG-1/手順1–2/研究資料。I-TR2=ISG-2/手順3–4,9/anchor testsと入力manifest。
I-TR3=ISG-3/手順5–7,11/minimal parity・fusion・export。
I-TR4=ISG-4/手順8,10,12–13/訓練・勾配・全F report。
I-TR5=ISG-5/手順14–15/親手順24.6〜32。各完了記録に実時刻と成果物を記す。

## 2. 作業内容

### フェーズ1 調査・設計（ISG-1、I-TR1）

KD回帰のteacher-confidence/teacher-upper-bound、MOHAWK blockwise teacher forcing、gradient similarity、sequence-length warmup、VSSD/EfficientViM/TinyViM/MobileMamba、structured compression/reparamを一次資料から確認する。研究と実装を混ぜない。

現実測：teacher 8,117,988、student 620,502。全F monitor metric-AJ teacher .26813、A1 seed42 .22323、GT-only .22773。短lossの改善と全F AJは一致しない。
query時刻だけをwindow先頭へ変えるfallbackはxy/GT不整合。short1027本のtrack中央値9。projection未使用272,440（二層、CPU実測）、実効348,062。末端state poolingと非正規化sumの長さ依存は原因仮説。step190 KD勾配上界/GTは0.123%以下。

### フェーズ2 実装（ISG-2〜4、I-TR2〜4）

1. 新入力：明示opt-in `--reanchor-window`、window8/16/32を新rootへ計算、旧root禁止。GT finite/positive z/visible/boundsのある点だけを選び、最初の有効可視時刻のXYZをKで投影。固定窓に有効点がない場合だけ、同じclip内で決定的な順序で有効窓を探索し、該当clipを別clipへ置換しない。clip全体に有効点がない場合は明示失敗。window start/fallbackをpayload/manifestへ記録する。旧データ経路のdefaultは変更しない。query policy/frontend precision/window/config hashをpayload/manifestへ記録し再利用不一致拒否。
2. `model/compact_vssd.py`：C-only q/q2、B/V/Δ/A-only KVのminimal版。旧two-pool関数と重みを厳密移植し、全4出力parity。旧v1は保持。
3. `model/rep_temporal.py`：DW Conv1D kernel7/3/1とidentityの線形和。非線形は和の後。BN無し（小さな可変bucket対策）、copy-to-deployで単一kernel7へ融合。恒等branchを中心tapへ加算しbias合算。学習branchとdeploy paramsを別計数。
4. 新local-global student：128×2、4heads/state64のminimal poolにlocal temporal DWConv＋128→512→128 FFN、残差・LayerNormを追加。deploy600〜625k目安、必ず650k以下。normalized globalは終端累積decayを除き、`softmax(log(delta+1e-6)+delta*A, F)`、secondはpositive weights/sum。これはNC-SSDからの独自適用で旧関数と同一とは主張しない。local/global/FFNはそれぞれ残差の小さなlearned scaleで安定化。N方向には混ぜない。
5. architecture registryはv1と明示v2/minimalだけ許可しstrict save/load/exportを維持。wrapper/auxを公開artifactへ混入しない。fusedとunfusedは異なるarchitecture/keysを明示し、ロード時に暗黙fusionしない。v2 training publicbestはunfused形式（architecture末尾 `_train`）、optimizer resumeもunfusedを保持。exportは明示copy-to-deployでfusedarchitectureへ変換し、fusion数値reportを残す。未知tag・混在keys拒否。
6. 蒸留：teacher-forced block prealignment→GT重視finetune。前段のteacher hiddenを対応student blockへ入れる。teacher/common geometryは事前整合時freeze。projectorはtraining-only。end-to-endはteacherがGTよりstudentより良い位置だけKDを有効化、GTは常に残す。gateはdetach、confidence floor無しを明示。GT/KD normとcosineを測り、scale補正はbounded係数の別arm。負cosineのaux gateを別比較とする。

### フェーズ3 検証・入力生成（ISG-2〜4）

testsを実装前にREDにし、CPUparity→fusion→dynamic ONNX→GPUsmoke。新train/monitorは既存1027/15の固定split。新入力はDesktop別root150GB、空き100GB以上、旧workspace保全。新window16を最初に生成し、8/32は必要な対照・curriculumで追加する。旧16入力生成はbaselineとして完了まで保持可能だが、旧32以降の無用な作業はPID検証後の正常停止を記録してよい。

### フェーズ4 訓練・選択（ISG-4）

最初の比較はR0旧v1＋修正query＋GT-only、R1新v2＋修正query＋GT-only、R2新v2＋block事前整合＋selective KD。全て同seed42/common init/split/window/effective batch32/同finetune200steps、同安全microbatch。R2事前整合100stepsは追加費用として別記、total compute一致を偽らない。初期R0/R1の異種random mixerをbitwise同じと言わない。

安全microbatch1/4/8/16/32をprofileし2GBreserveを保つ。exact bucketが小さい場合は無理にpaddingしない。GPU使用率/track数/実microbatchを記録。32GBを使い切るためだけのtensor確保はしない。

採否は既存15本全F monitorのmacro metric-AJを第一、3D-AJ/subsetを併記して決める。選択前にminival予測metricを読まない。同点は小サイズ、少計算を優先。選択runでseed42/43/44の本訓練とGT-only対照、16→32長適応へ接続し、親の最終受入へ進む。追加64/fullF curriculumは必要時に入力容量/computeを見積もって本書へ手順追加する。

### フェーズ5 記録・受入への接続（ISG-5）

親protocol `result/refiner_kd_20261002/protocol.json` SHA e83a01247263c84c27273427578b8c8093b48c6a6de42b514dbe12d949941054は無変更。macro AJ低下≤.003、subset≤.005、10k sequence bootstrap95%上限、CPU/CUDA dynamic・長F900track・benchmark・全150本を保持。入力修正で旧lossと直接比較できないことを報告。

## 3. 作業チェックリスト

非同期入力生成と独立module/test作業はrootが担当範囲を明示して並行可。他agentによる同ファイル編集は禁止。チェックとその直後の作業記録更新はrootが一項目ずつ行う。

### フェーズ1 調査・設計

### 手順 1: 一次研究と現状を照合する

- [x] 🖐 **操作**: primary papersと現sourceを調査しresearch docへ分類する。I-TR1。
- [x] 🔎 **確認**: KD/vision/light/compression、実証・仮説・ONNX難度・不採用理由がある。
- [x] 🧪 **テスト**: 調査のRED不適用。URLと論文日時・出版区分を照合。
- [x] 🛠 **エラー時対処**: abstractしか読めない研究は限定範囲を記し、未公開コードの動作を断言しない。

### 手順 2: 本書をreviewする

- [x] 🖐 **操作**: review rubricで全文を点検し `temp/review_Oct02-2026_kd_accuracy_improvement.md` へ所見を残す。I-TR1/I-TR5。
- [x] 🔎 **確認**: final verdictにBlocker/Major未解決がない。
- [x] 🧪 **テスト**: 15手順×4checkbox、要求/DoD/実装境界・親との関係を照合。
- [x] 🛠 **エラー時対処**: schema/選択条件の曖昧さは実装前に修正する。

### フェーズ2 実装

### 手順 3: anchor整合性のREDテストを書く

- [x] 🖐 **操作**: `tests/unit/test_kd_improvement_data.py` へanchorなし・無可視・NaN/負depth・範囲外・query policy再利用拒否を追加。I-TR2。
- [x] 🔎 **確認**: 旧fallbackの不整合を再現し新strict opt-inの不足が失敗原因。
- [x] 🧪 **テスト**: `uv run --no-sync pytest tests/unit/test_kd_improvement_data.py -q` のREDを保存。
- [x] 🛠 **エラー時対処**: RGB実decodeに依存しない小fixtureを使い、失敗をimport環境問題と混同しない。

### 手順 4: strict reanchor準備を実装する

- [x] 🖐 **操作**: datasetの新strict opt-inとprepareの新namespace/provenanceを実装する。I-TR2。
- [x] 🔎 **確認**: xy/time双方正しく、visible/finite/positive/bounds、GT同frame、旧default不変。
- [x] 🧪 **テスト**: 手順3がGREEN、query0点で明示失敗。旧cache拒否を確認。
- [x] 🛠 **エラー時対処**: oldkeyやrawCRC不一致は新rootを使う。既存cacheを削除/上書きしない。

### 手順 5: minimal projectionを実装する

- [x] 🖐 **操作**: `model/compact_vssd.py` と `tests/unit/test_compact_vssd.py` をTDDで追加。I-TR3。
- [x] 🔎 **確認**: 永続unused272,440削減、旧function移植parity、zero/nonzero pool2 gate。
- [x] 🧪 **テスト**: 複数F/NのFP32 rtol1e-5/atol1e-6、grad使用行、teacher/sharedgeometry不変。全suiteのCPU thread設定でnear-zero parityが6件失敗したため20:45に再開し、許容誤差を変えず切り分ける。
- [x] 🛠 **エラー時対処**: packedLinear row rangesとBCNorm対応を確認し、parity toleranceを緩和しない。

### 手順 6: temporal reparamを実装する

- [x] 🖐 **操作**: `model/rep_temporal.py` と `tests/unit/test_rep_temporal.py` をTDDで追加。I-TR3。
- [x] 🔎 **確認**: 7/3/1/identity線形和→単一7、copy fusion、元学習model無変更。
- [x] 🧪 **テスト**: F1/8/31/600のfusion rtol1e-5/atol1e-6、非finite拒否、B/N独立。
- [x] 🛠 **エラー時対処**: branch内非線形/stride違いは拒否し、中心padding/biasを修正。

### 手順 7: local-global studentとstrict公開形式を実装する

- [x] 🖐 **操作**: v2studentとregistry/save/loadを追加する。I-TR3。
- [x] 🔎 **確認**: deploy≤650k、標準opのみ、8入力4出力、未使用bundle無し、teacher/aux無し。
- [x] 🧪 **テスト**: schema破損/未知architecture拒否、保存再load bitwise、F1/長F・N permutation。
- [x] 🛠 **エラー時対処**: train/deploykeys混在は明示versionを分け、wrapperを黙って落とさない。

### 手順 8: 段階的蒸留を実装する

- [x] 🖐 **操作**: teacher-forced block整合とselective KD/gradient診断をtrainへ追加。I-TR4。
- [x] 🔎 **確認**: 事前整合時common freeze、同hidden入力、GT常時、teacher-better detach gate、finite。
- [x] 🧪 **テスト**: `test_staged_distillation.py` のfreeze/teacher worse/no-mask/cos/ratio/resume同等性をRED→GREEN。
- [x] 🛠 **エラー時対処**: 勾配norm0ではbounded係数0、NaN失敗。無根拠なKD1000倍はしない。

### フェーズ3 検証・入力生成

### 手順 9: 新window16入力を生成する

- [x] 🖐 **操作**: 新config/明示reanchorで新rootへtrain1027/monitor15を生成する。固定窓内0 anchorなら同一clipの別窓を探索し、chosen window start/fallbackも保存する。I-TR2。
- [x] 🔎 **確認**: 全件anchor projection/validity audit、track中央値、policy/config/raw/depth hash、容量内。
- [x] 🧪 **テスト**: 先に2clip pilot、終了manifest complete、split exactset/重複0。固定窓が空のclipの同一clip-window fallback、従来有効窓不変、全clip窓で0 anchor時の失敗を単体試験する。
- [x] 🛠 **エラー時対処**: frontend認証/OOMは既存pinnedlocal重み/安全batchを確認。同clip以外resample禁止。

### 手順 10: staged smokeを実行する

- [x] 🖐 **操作**: 新入力でblock事前整合→20step finetuneを実行する。I-TR4。
- [x] 🔎 **確認**: blockloss減少、student更新、teacher不変、finite、GT/KD grad/gate統計。
- [x] 🧪 **テスト**: 10→20resumeと連続20のtrainable tensorsを比較、専用status/hash保存。
- [x] 🛠 **エラー時対処**: OOMはprofile-safe microbatchへ。failed stateから無検証resumeしない。

### 手順 11: v2のONNX互換性を試す

- [x] 🖐 **操作**: v2の未学習/試験済みcopy-fusedを標準opset18でexportする。I-TR3。
- [x] 🔎 **確認**: dynamic B/F/N/depthHW、F1/600/N900chunk、CPU/GPU主要float演算CUDA。
- [x] 🧪 **テスト**: fusion1e-5/1e-6、ORT2e-4/5e-5、chunk/permutation、aux外。
- [x] 🛠 **エラー時対処**: exporterのF固定化/非標準domainは拒否して演算を改める。

### フェーズ4 訓練・選択

### 手順 12: R0/R1/R2 pilotを比較する

- [ ] 🖐 **操作**: 同新入力・seed42・finetune200stepsでGT-only/新構造/stagedKDを実行する。I-TR4。
- [ ] 🔎 **確認**: shared初期化条件、microbatch、追加事前整合費用、全F monitor15、subset別AJを明記。
- [ ] 🧪 **テスト**: teacher-better gate通過率/cosine/梯度ratio、旧shortlossとの非直接比較を確認。
- [ ] 🛠 **エラー時対処**: 低下はstage/query/structureを個別切分け、minival探索禁止。

### 手順 13: monitor選択後の本訓練を進める

- [ ] 🖐 **操作**: monitor選択をhash記録し、選択モデルのseed42/43/44と対照・32適応を実行する。I-TR4。
- [ ] 🔎 **確認**: 学習済みpublicbest、全F性能、resume、処理clip数、GPUpeak、teacher不変。
- [ ] 🧪 **テスト**: selected config/manifest/init hashes、seedをtest結果でcherry-pickしない。
- [ ] 🛠 **エラー時対処**: 改善なしなら明示architecture候補C/assistantを別計画追加し、基準は維持。

- [ ] 🖐 **操作**: R0/R1/R2のwindow16 pilotと並行し、`configs/distill_refiner_local_global_w32.yaml` でstrict window32 train1027/validation15 cacheを独立rootへ生成する。既存legacy window32 cacheは再利用しない。
- [ ] 🔎 **確認**: window32 split exactset、anchor audit、同clip fallback/start、FP32 frontend provenance、容量を確認し、manifest completeを待つ。
- [ ] 🧪 **テスト**: window32 inputのSHA/policy/window/query identityがwindow16と独立していることを確認する。32 adaptation本体は全F monitor選択後に開始する。
- [ ] 🛠 **エラー時対処**: input budget/disk reserve/DINO/WAFT OOM/Authエラーはpartial manifestから同条件で再開し、window16/32や旧cacheを混同しない。

### フェーズ5 記録・接続

### 手順 14: docsと品質検査を仕上げる

- [ ] 🖐 **操作**: `uv run --no-sync pytest tests/unit -q` と対象ruff/tyで品質を検査する。I-TR5。
- [ ] 🔎 **確認**: 新旧regression成功、研究/実装/実測/未達の区別、docs索引リンク。
- [ ] 🧪 **テスト**: command exit0、全件数を記録。未実行GPUを成功と書かない。
- [ ] 🛠 **エラー時対処**: 対象新ファイルだけformatし、無関係dirty変更は保全。

### 手順 15: 親の最終受入へ接続する

- [ ] 🖐 **操作**: 親24.6以降に選択モデル/新入力/改善実験結果を追記する。I-TR5。
- [ ] 🔎 **確認**: 最終150本AJ/CI・FP32 ONNX CPU/CUDA・benchmarkの必須条件は不変。
- [ ] 🧪 **テスト**: protocol SHA不変、親D-1〜D-8を部分成果だけでチェックしていない。
- [ ] 🛠 **エラー時対処**: 精度未達は本書/親に理由を残す。goal completeを偽らない。

## 4. 作業に使用するコマンド参考情報

```sh
date '+%Y-%m-%d %H:%M:%S %Z%z'
git status --short --branch
uv run --no-sync pytest tests/unit/test_kd_improvement_data.py -q
uv run --no-sync pytest tests/unit/test_compact_vssd.py tests/unit/test_rep_temporal.py tests/unit/test_staged_distillation.py -q
uv run --no-sync pytest tests/unit -q
uv run --no-sync ty check
```

justfile無し、uv.lockあり。新CLI詳細は手順4/8の実装完了時に確定し、実行前にコピーペースト可能なcommandを本書へ追記する。環境再syncは不要。rootCPU ORTをGPU版へ上書きしない。GPUは既存 `result/onnx_gpu_20261002/.venv` を使う。

手順4/9で確定した新入力生成command（16-frame train完了後monitorを連続生成、150GB上限/空き100GB）：

```sh
source scripts/cudnn_env.sh
export HF_HUB_OFFLINE=1
for partition in train validation; do
  uv run --no-sync python scripts/prepare_refiner_kd.py --config configs/distill_refiner_local_global.yaml --mode cache --partition "$partition" --window 16 --reanchor-window --compute-frontend --frontend-precision fp32 --new-input-root /home/kasm-user/Desktop/vmamba3-3Dpointtracker/result/refiner_kd_improved_20261002/inputs --new-input-budget-gb 150 --out-dir "result/refiner_kd_improved_20261002/inputs_w16_${partition}" || exit "$?"
done
```

このsource commandはbashで実行する。2clip診断時は `--limit 2`、out-dir `input_pilot_w16`。既存oldrootと異なり、query-policyとFP32 frontendがmanifestに明示される。

window32 adaptation cacheはwindow16と同じstrict policy/FP32 frontendで独立したwindow identityへ作成する。16 pilotの実行と並行してよいが、32 adaptation本体は全F monitor選択後に開始する。

```bash
source scripts/cudnn_env.sh
export HF_HUB_OFFLINE=1
for partition in train validation; do
  uv run --no-sync python scripts/prepare_refiner_kd.py --config configs/distill_refiner_local_global_w32.yaml --mode cache --partition "$partition" --window 32 --reanchor-window --compute-frontend --frontend-precision fp32 --new-input-root /home/kasm-user/Desktop/vmamba3-3Dpointtracker/result/refiner_kd_improved_20261002/inputs --new-input-budget-gb 150 --out-dir "result/refiner_kd_improved_20261002/inputs_w32_${partition}" || exit "$?"
done
```

手順8/10で確定したGPU smoke command（hint10→finetune20、eff32）：

```sh
source scripts/cudnn_env.sh
HF_HUB_OFFLINE=1 uv run --no-sync python scripts/train_refiner_staged.py --mode smoke --method staged --architecture vssd_local_global_128x2_v2_train --config configs/distill_refiner_local_global.yaml --input-manifest result/refiner_kd_improved_20261002/input_pilot_w16/input_manifest.json --monitor-manifest result/refiner_kd_improved_20261002/inputs_w16_validation/input_manifest.json --out-dir result/refiner_kd_improved_20261002/smoke_staged_s42 --seed 42 --microbatch 4
```

再開試験は同じ引数で `--resume result/refiner_kd_improved_20261002/smoke_staged_s42/step10.pt`、別の未作成out-dirを使う。pilotではfull input manifestへ切り替え、`--mode pilot` とし、R0=`--architecture vssd_two_pool_128x2_v1 --method gt`、R1=`--architecture vssd_local_global_128x2_v2_train --method gt`、R2=`--architecture vssd_local_global_128x2_v2_train --method staged`。全F monitor15評価では旧selected configに固定済みのfrontend契約とprepared inputを共用し、新訓練configとは別のhashとして記録する。

`gradient_diagnostics.effective_ratio` は各microbatchでwarmupのrampを掛ける**前**のKD/GT norm比。実適用比はこれに同rowの`ramp`を掛ける。all_microbatch_weighted_meanの値はclip重み付き平均で、勾配を合算して計算したnorm/cosineではない。

teacher weights: `weights/mamba3-preview-20261002-best80/tracker_mamba3_daoff_best80.pt`、SHA c022511872c2d59a39bc1c39a3c62f44c142ee8ff9ca7dc7859881a78ae490b7。
split: `/workspace/vmamba3_data/result/v64_daoff_unused100gb_train_20261002/growing_pool.json`、SHA 1557b93e38cfd64825a67de8436b1e4322d37ad83c0422b149d9403ec2d0e5d5。
WAFT SHA9f4b24f48b3937eca690a12b73bc3190effde6d4d4c87db01998fe63d846397f、DINO ViT-S16 revision114c1379950215c8b35dfcd4e90a5c251dde0d32、DA3 snapshot4010e39f3634a45bc60553321fb49fb760bd594eは既存と同じ。

## 6. 完了の定義

2026-10-02 21:48 JSTのユーザー追記: 本日は残り約30分で区切り、必要な追加学習だけ開始し、early stoppingを有効にする。直近はR2 block100＋finetune最大200を優先し、`--early-stop`でmonitor_every10/patience5/min_delta0.001を適用する。R0/R1比較と本訓練は未実施のまま保持し、I-D4/親DoDの完了を主張しない。起動・同条件再開は `bash temp/run_kd_additional_r2.sh`。最良checkpointの全F monitor15評価を後続実行する。背景ジョブと記録は `docs/kd_additional_training_20261002.md` にまとめた。

- [ ] I-D1: 詳細primary research、実測所見・仮説・採否根拠がISG-1/I-TR1資料にありreview済み。
- [ ] I-D2: 新window16/32 inputsのtrain1027/monitor15でanchor整合・policy/manifest/provenance/旧保全を証明。ISG-2/I-TR2。
- [ ] I-D3: minimal parity、v2≤650k、fusion・dynamic ONNX CPU/CUDA成功。ISG-3/I-TR3。
- [ ] I-D4: R0/R1/R2と選択後本訓練/seed/32適応、全F monitor採否・梯度根拠。ISG-4/I-TR4。
- [ ] I-D5: uv tests/ruff/ty、docs/記録、親最終受入への接続が揃う。ISG-5/I-TR5。本書の完了は親の品質受入完了を意味しない。

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
| 2026-10-02 | 20:00:06 JST+0900 | root | フェーズ1開始・追加計画 | 正しいquery、unused削減、local/global、stagedKDを優先。学習本体なし、minival raw/depth/frontendと旧long16 prepは稼働中。旧成果は保持。 |
| 2026-10-02 | 20:04:55 JST+0900 | root | 手順 2 🖐 **操作**: revie | root review-and-fix PASS: train/fused architecture境界修正、15手順/65checkbox、親DoD維持。temp/review_Oct02-2026_kd_accuracy_improvement.md。研究doc作成と独立module作業の並列を明示。 |
| 2026-10-02 | 20:04:55 JST+0900 | root | 手順 2 🔎 **確認**: final | root review-and-fix PASS: train/fused architecture境界修正、15手順/65checkbox、親DoD維持。temp/review_Oct02-2026_kd_accuracy_improvement.md。研究doc作成と独立module作業の並列を明示。 |
| 2026-10-02 | 20:04:55 JST+0900 | root | 手順 2 🧪 **テスト**: 15手順 | root review-and-fix PASS: train/fused architecture境界修正、15手順/65checkbox、親DoD維持。temp/review_Oct02-2026_kd_accuracy_improvement.md。研究doc作成と独立module作業の並列を明示。 |
| 2026-10-02 | 20:04:55 JST+0900 | root | 手順 2 🛠 **エラー時対処**: s | root review-and-fix PASS: train/fused architecture境界修正、15手順/65checkbox、親DoD維持。temp/review_Oct02-2026_kd_accuracy_improvement.md。研究doc作成と独立module作業の並列を明示。 |
| 2026-10-02 | 20:04:55 JST+0900 | root | 手順 8 進捗 | staged training helper/test責務調査開始。native layer pre/post hookで同一teacher inputを各student mixerへ。GT常時・teacher-better gate・gradient診断をtraining-only実装。 |
| 2026-10-02 | 20:13:41 JST+0900 | root | 手順 9 進捗 | 旧window16 train1027/monitor15 complete保存。旧32train PID2819849+連鎖shell2756585をread-only照合後TERMし、不整合legacy queryの追加計算を停止。既存PT・cache削除なし。新FP32/reanchor生成へ切替準備。minival前処理はpartial終端を確認中。 |
| 2026-10-02 | 20:13:56 JST+0900 | root | 手順 5 🖐 **操作**: `mode | worker GREEN17/17 afterRED13未実装fail。minimal exact copied FP32 forward4outputs/F1–600 passes1e-5/1e-6、packedGEMM gradient比較はfloat64独立試験。134285/layer、実効refiner348062、unused272440削減。root再検証は全unit段階。 |
| 2026-10-02 | 20:13:56 JST+0900 | root | 手順 5 🔎 **確認**: 永続unu | worker GREEN17/17 afterRED13未実装fail。minimal exact copied FP32 forward4outputs/F1–600 passes1e-5/1e-6、packedGEMM gradient比較はfloat64独立試験。134285/layer、実効refiner348062、unused272440削減。root再検証は全unit段階。 |
| 2026-10-02 | 20:13:56 JST+0900 | root | 手順 5 🧪 **テスト**: 複数F/ | worker GREEN17/17 afterRED13未実装fail。minimal exact copied FP32 forward4outputs/F1–600 passes1e-5/1e-6、packedGEMM gradient比較はfloat64独立試験。134285/layer、実効refiner348062、unused272440削減。root再検証は全unit段階。 |
| 2026-10-02 | 20:13:56 JST+0900 | root | 手順 5 🛠 **エラー時対処**: p | worker GREEN17/17 afterRED13未実装fail。minimal exact copied FP32 forward4outputs/F1–600 passes1e-5/1e-6、packedGEMM gradient比較はfloat64独立試験。134285/layer、実効refiner348062、unused272440削減。root再検証は全unit段階。 |
| 2026-10-02 | 20:11:57 JST+0900 | root | 手順 8 進捗 | staged helper tests RED6ModuleNotFound→GREEN6passed2.67s。teacher-worseKDゼロ・GT常時・no confidence floor、negative cosine gate、bounded ratio、teacher/commonfreezeとhookcleanup、zero-mask/NaN。新trainer CLI/resume統合は未了。 |
| 2026-10-02 | 20:10:50 JST+0900 | root | 手順 3 🖐 **操作**: `test | worker初回RED18failed/1passed missing strictoptin/CLI/helpers/schema2。新23tests anchorなし/invalid/bounds/K/time/legacyreuse、nonzero start・schema2、旧経路維持。 |
| 2026-10-02 | 20:10:50 JST+0900 | root | 手順 3 🔎 **確認**: 旧fall | worker初回RED18failed/1passed missing strictoptin/CLI/helpers/schema2。新23tests anchorなし/invalid/bounds/K/time/legacyreuse、nonzero start・schema2、旧経路維持。 |
| 2026-10-02 | 20:10:50 JST+0900 | root | 手順 3 🧪 **テスト**: `uv  | worker初回RED18failed/1passed missing strictoptin/CLI/helpers/schema2。新23tests anchorなし/invalid/bounds/K/time/legacyreuse、nonzero start・schema2、旧経路維持。 |
| 2026-10-02 | 20:10:50 JST+0900 | root | 手順 3 🛠 **エラー時対処**: R | worker初回RED18failed/1passed missing strictoptin/CLI/helpers/schema2。新23tests anchorなし/invalid/bounds/K/time/legacyreuse、nonzero start・schema2、旧経路維持。 |
| 2026-10-02 | 20:16:40 JST+0900 | root | 手順 4 🖐 **操作**: datas | worker47passed（新23+既存24）、ownedruff/ty成功、実schema2configPASS。strict valid可視finitepositive/bounds・整数anchor・sameK再投影≤.001px、全0明示失敗、旧default不変。新policy/precision/seed/windowSHA reuse拒否。root2clipGPUpilot開始。 |
| 2026-10-02 | 20:16:40 JST+0900 | root | 手順 4 🔎 **確認**: xy/ti | worker47passed（新23+既存24）、ownedruff/ty成功、実schema2configPASS。strict valid可視finitepositive/bounds・整数anchor・sameK再投影≤.001px、全0明示失敗、旧default不変。新policy/precision/seed/windowSHA reuse拒否。root2clipGPUpilot開始。 |
| 2026-10-02 | 20:16:40 JST+0900 | root | 手順 4 🧪 **テスト**: 手順3が | worker47passed（新23+既存24）、ownedruff/ty成功、実schema2configPASS。strict valid可視finitepositive/bounds・整数anchor・sameK再投影≤.001px、全0明示失敗、旧default不変。新policy/precision/seed/windowSHA reuse拒否。root2clipGPUpilot開始。 |
| 2026-10-02 | 20:16:40 JST+0900 | root | 手順 4 🛠 **エラー時対処**: o | worker47passed（新23+既存24）、ownedruff/ty成功、実schema2configPASS。strict valid可視finitepositive/bounds・整数anchor・sameK再投影≤.001px、全0明示失敗、旧default不変。新policy/precision/seed/windowSHA reuse拒否。root2clipGPUpilot開始。 |
| 2026-10-02 | 20:15:14 JST+0900 | root | 手順 6 🖐 **操作**: `mode | worker7passed2.04s afterRED7modulemissing。kernel7/3/1+identity線形和copyfusion、F1/8/31/600 outputgradient、時間/独立batch、source不変、NaN/nonlinearbranch拒否。BN無し、非線形はfusion外。 |
| 2026-10-02 | 20:15:14 JST+0900 | root | 手順 6 🔎 **確認**: 7/3/1 | worker7passed2.04s afterRED7modulemissing。kernel7/3/1+identity線形和copyfusion、F1/8/31/600 outputgradient、時間/独立batch、source不変、NaN/nonlinearbranch拒否。BN無し、非線形はfusion外。 |
| 2026-10-02 | 20:15:14 JST+0900 | root | 手順 6 🧪 **テスト**: F1/8 | worker7passed2.04s afterRED7modulemissing。kernel7/3/1+identity線形和copyfusion、F1/8/31/600 outputgradient、時間/独立batch、source不変、NaN/nonlinearbranch拒否。BN無し、非線形はfusion外。 |
| 2026-10-02 | 20:15:14 JST+0900 | root | 手順 6 🛠 **エラー時対処**: b | worker7passed2.04s afterRED7modulemissing。kernel7/3/1+identity線形和copyfusion、F1/8/31/600 outputgradient、時間/独立batch、source不変、NaN/nonlinearbranch拒否。BN無し、非線形はfusion外。 |
| 2026-10-02 | 20:18:56 JST+0900 | root | 手順 9 進捗 | 新2clippilot complete。FP32DINO、window16/strictGTanchor、3.77s/3.17s入力生成。root全train→validation連鎖開始、旧cache再利用不可の別root150GB。全件前はcheckbox/DoD未完。 |
| 2026-10-02 | 20:27:31 JST+0900 | root | 手順 9 進捗 | 40行動refresh。新train140/1027時点進行中、2clip90/132tracks auditmax.000122px。新monitor15を独立プロセスで先行生成開始、roottrainchain後段はhash検証reuse。v2fusion/ONNX9CPUcases初版success、pool2極負値数値安定追加監査中。GT/KD stagedhelper6+schemaexport5testsPASS。minivalraw127/depth127、frontend111partial。 |
| 2026-10-02 | 20:31:34 JST+0900 | root | 手順 1 🖐 **操作**: prima | research29項目/515行、root全文split読みprimaryfacts/codechecks照合、nonexistentsourcepath修正。回帰KD/MOHAWK/gradient/lengthwarmup/Vim/VMamba/VSSD/EfficientViM/Tiny/Mobile/Local/Effi/LightViM/圧縮/reparam/PTQを採否と限界付き。source citations docs/kd_accuracy_research_20261002.md。改善実績未検証を維持。 |
| 2026-10-02 | 20:31:34 JST+0900 | root | 手順 1 🔎 **確認**: KD/vi | research29項目/515行、root全文split読みprimaryfacts/codechecks照合、nonexistentsourcepath修正。回帰KD/MOHAWK/gradient/lengthwarmup/Vim/VMamba/VSSD/EfficientViM/Tiny/Mobile/Local/Effi/LightViM/圧縮/reparam/PTQを採否と限界付き。source citations docs/kd_accuracy_research_20261002.md。改善実績未検証を維持。 |
| 2026-10-02 | 20:31:34 JST+0900 | root | 手順 1 🧪 **テスト**: 調査のR | research29項目/515行、root全文split読みprimaryfacts/codechecks照合、nonexistentsourcepath修正。回帰KD/MOHAWK/gradient/lengthwarmup/Vim/VMamba/VSSD/EfficientViM/Tiny/Mobile/Local/Effi/LightViM/圧縮/reparam/PTQを採否と限界付き。source citations docs/kd_accuracy_research_20261002.md。改善実績未検証を維持。 |
| 2026-10-02 | 20:31:34 JST+0900 | root | 手順 1 🛠 **エラー時対処**: a | research29項目/515行、root全文split読みprimaryfacts/codechecks照合、nonexistentsourcepath修正。回帰KD/MOHAWK/gradient/lengthwarmup/Vim/VMamba/VSSD/EfficientViM/Tiny/Mobile/Local/Effi/LightViM/圧縮/reparam/PTQを採否と限界付き。source citations docs/kd_accuracy_research_20261002.md。改善実績未検証を維持。 |
| 2026-10-02 | 20:43:31 JST+0900 | root | 手順 7 🖐 **操作**: v2stu | v2学習617374/推論615838。既存geometry共有・LocalGlobal2層・registeredtrain/deployschemaを分離。registry exacttype/modules/公開semanticflags(正規化・deploy・Convpadding・GELU・geometrybounds)/全keys/shapes/finiteFP32をsave/export前にfailclosed照合、factorytemplateはfork_rngCPUでRNG保存。追加RED12fail+RNG1pass→GREEN: newdeployment18+旧deployment13+trainer21=52tests成功、ty成功。loadbitwise、改ざん/aux/teacher拒否、dynamicF/N/perm/chunk成功。CPUexport stable9caseも成功(実GPUは手順11未実施)。 |
| 2026-10-02 | 20:43:31 JST+0900 | root | 手順 7 🔎 **確認**: deplo | v2学習617374/推論615838。既存geometry共有・LocalGlobal2層・registeredtrain/deployschemaを分離。registry exacttype/modules/公開semanticflags(正規化・deploy・Convpadding・GELU・geometrybounds)/全keys/shapes/finiteFP32をsave/export前にfailclosed照合、factorytemplateはfork_rngCPUでRNG保存。追加RED12fail+RNG1pass→GREEN: newdeployment18+旧deployment13+trainer21=52tests成功、ty成功。loadbitwise、改ざん/aux/teacher拒否、dynamicF/N/perm/chunk成功。CPUexport stable9caseも成功(実GPUは手順11未実施)。 |
| 2026-10-02 | 20:43:31 JST+0900 | root | 手順 7 🧪 **テスト**: sche | v2学習617374/推論615838。既存geometry共有・LocalGlobal2層・registeredtrain/deployschemaを分離。registry exacttype/modules/公開semanticflags(正規化・deploy・Convpadding・GELU・geometrybounds)/全keys/shapes/finiteFP32をsave/export前にfailclosed照合、factorytemplateはfork_rngCPUでRNG保存。追加RED12fail+RNG1pass→GREEN: newdeployment18+旧deployment13+trainer21=52tests成功、ty成功。loadbitwise、改ざん/aux/teacher拒否、dynamicF/N/perm/chunk成功。CPUexport stable9caseも成功(実GPUは手順11未実施)。 |
| 2026-10-02 | 20:43:31 JST+0900 | root | 手順 7 🛠 **エラー時対処**: t | v2学習617374/推論615838。既存geometry共有・LocalGlobal2層・registeredtrain/deployschemaを分離。registry exacttype/modules/公開semanticflags(正規化・deploy・Convpadding・GELU・geometrybounds)/全keys/shapes/finiteFP32をsave/export前にfailclosed照合、factorytemplateはfork_rngCPUでRNG保存。追加RED12fail+RNG1pass→GREEN: newdeployment18+旧deployment13+trainer21=52tests成功、ty成功。loadbitwise、改ざん/aux/teacher拒否、dynamicF/N/perm/chunk成功。CPUexport stable9caseも成功(実GPUは手順11未実施)。 |
| 2026-10-02 | 20:43:31 JST+0900 | root | 手順 8 🖐 **操作**: teach | stagedhelper6tests+trainer21tests成功、RED→GREEN。block teacher同入力/rawmixeroutput RMS loss・commonfreeze・hookfinallycleanup。selectiveはteacherbetterdetach/誤差confidencefloor無し・GT常時、cosine方向とtargetratio.1/max100のbounded係数。診断student.layers、allmicro重み平均とfirstmicro区別。phase別profile/VRAM/RNG/sampler/reset、strictconfig/split/raw/depth/query/sourcehash再開検証、CPUoptimizerbitwise再開成功。GPU学習/restartは手順10に残す。 |
| 2026-10-02 | 20:43:31 JST+0900 | root | 手順 8 🔎 **確認**: 事前整合時 | stagedhelper6tests+trainer21tests成功、RED→GREEN。block teacher同入力/rawmixeroutput RMS loss・commonfreeze・hookfinallycleanup。selectiveはteacherbetterdetach/誤差confidencefloor無し・GT常時、cosine方向とtargetratio.1/max100のbounded係数。診断student.layers、allmicro重み平均とfirstmicro区別。phase別profile/VRAM/RNG/sampler/reset、strictconfig/split/raw/depth/query/sourcehash再開検証、CPUoptimizerbitwise再開成功。GPU学習/restartは手順10に残す。 |
| 2026-10-02 | 20:43:31 JST+0900 | root | 手順 8 🧪 **テスト**: `tes | stagedhelper6tests+trainer21tests成功、RED→GREEN。block teacher同入力/rawmixeroutput RMS loss・commonfreeze・hookfinallycleanup。selectiveはteacherbetterdetach/誤差confidencefloor無し・GT常時、cosine方向とtargetratio.1/max100のbounded係数。診断student.layers、allmicro重み平均とfirstmicro区別。phase別profile/VRAM/RNG/sampler/reset、strictconfig/split/raw/depth/query/sourcehash再開検証、CPUoptimizerbitwise再開成功。GPU学習/restartは手順10に残す。 |
| 2026-10-02 | 20:43:31 JST+0900 | root | 手順 8 🛠 **エラー時対処**: 勾 | stagedhelper6tests+trainer21tests成功、RED→GREEN。block teacher同入力/rawmixeroutput RMS loss・commonfreeze・hookfinallycleanup。selectiveはteacherbetterdetach/誤差confidencefloor無し・GT常時、cosine方向とtargetratio.1/max100のbounded係数。診断student.layers、allmicro重み平均とfirstmicro区別。phase別profile/VRAM/RNG/sampler/reset、strictconfig/split/raw/depth/query/sourcehash再開検証、CPUoptimizerbitwise再開成功。GPU学習/restartは手順10に残す。 |
| 2026-10-02 | 20:44:03 JST+0900 | root | 手順 10 進捗 | 新query/FP32 frontend pilot2clip +監視15を使い GPU smokeを開始。staged: hint10→finetune20、eff32/profile micro4指定。publicsave semantic guard source固定。CLI session48151、実更新/teacher不変/resume検証は未完。 |
| 2026-10-02 | 20:46:12 JST+0900 | root | 手順 14 進捗 | 全unitは20:32jobで276pass/6fail。legacycompact forward parityのみF31/128/600×gate0/.37、単独35tests成功との差はthread/GEMM形状丸めをread-onlyworker調査。tolerance維持、手順5テストを再開。新v2source/RNG保存guardの52tests/tyは成功。staged GPU smoke hintloss.993135→.945075、FT8到達/finite/teacherbettergate .72-.77・cos.61-.76。学習続行・最終品質未達。 |
| 2026-10-02 | 20:50:58 JST+0900 | root | 手順 10 進捗 | 旧trainer source版 GPU smoke 10block+20FT complete、teacher/commonfreeze不変、fixedGT.228320→.154699/monitor.551610→.509668、phasepeak.270/.314GB。10→20再開98学習tensor/optimizer/CPU,CUDARNG/サンプラー/state counters bitwise一致 proof smoke_resume_proof_v1.json。公開status ft_processed=0のstale表示を発見(optimizercheckpoint640、processed_clips640正しい)。status_progress helperのRED2failedを追加、学習算術は変えずphase counters/badのログを同期修正。compactlegacy rounding対処と合わせ最終source freeze後もう一度GPU再開を確認予定。旧proofは保全。 |
| 2026-10-02 | 20:47:08 JST+0900 | root | 手順 11 進捗 | stable v2 未学習deployCPU9syntheticcase success。隔離ORTgpu1.30 env CUDA: synthetic9+実monitor3subset=12case成功、chunk1/permutationも成功。gpu_v2_stable_init/gpu_report.json、CUDA114436/CPU23485events。Conv/MatMul/FusedMatMul/Softmax/GridSample/LayerNorm/GELU全CUDA、CPUtensor全int64/主要float CPU0。訓練済みv2 export後の正式fusion/CPU/GPU確認はこれから。 |
| 2026-10-02 | 20:54:12 JST+0900 | root | 手順 5 🧪 **テスト**: 複数F/ | 旧FP32 parity fail原因はCPU threads32以上のpackedGEMM remainder: KV末尾Δ/A8行だけ丸め差。legacy normalized=False時のみ非parameter zero4rows追加の396row計算→392使用sliceで旧652row使用行bitwise一致。normalized=True新v2算術不変、learnedcount不変、alignment_rows primitiveをfactory固定しguard改ざん拒否。新RED14failure→compact33testsPASS、追加threads1/4/24/32/48/64×F1/8/31/128/600×gate0/.37=60conditions bitwise一致。tol1e-5/1e-6維持、FP64grad/移植/348062count成功。独立全unit341pass18.60s、ownedruff/typass。 |
| 2026-10-02 | 21:14:00 JST+0900 | root | 手順 9 進捗 | Fallback修正を作業開始。TDDでfixed window invalid時のみ同一clip内の可視valid anchor windowへ決定的に移し、初期窓有効時は不変、全clipで0なら失敗する。train query列固定、window_start/window_fallbackを新payload/manifestへ残す。 |
| 2026-10-02 | 21:17:44 JST+0900 | root | 手順 9 進捗 | strict cacheが681/1027、前回停止した実clipで同一clip内fallbackが成功。直近4.56s/件、manifestに実window_start/fallbackを記録する実装。unit386 passed。残346、cache prep継続中。 |
| 2026-10-02 | 21:19:50 JST+0900 | root | 手順 10 進捗 | 現行dataset same-clip fallbackと最終trainer guardの組合せで2clip smokeを開始。train2clip/monitor15 manifest completeを確認。 |
| 2026-10-02 | 21:22:50 JST+0900 | root | 手順 10 進捗 | 現行ソースsmoke完了: staged block10+FT20、GT短monitor0.55161→0.50967、KD/GT gate diagnostics finite、teacher不変、peak allocated313527808 bytes。step10から同条件resume比較を開始。 |
| 2026-10-02 | 21:25:02 JST+0900 | root | 手順 10 🖐 **操作**: 新入力でb | 現行ソースGPU smoke・resume証明success。98 trainable tensorsとoptimizer/RNG/counters bitwise一致、block loss .993135→.945075、固定GT probe .228320→.154699、short monitor .551610→.509668、teacher不変、peak allocated313527808B。scopeは2train clips/15 short-window monitorで精度評価ではない。 |
| 2026-10-02 | 21:25:02 JST+0900 | root | 手順 10 🔎 **確認**: block | 現行ソースGPU smoke・resume証明success。98 trainable tensorsとoptimizer/RNG/counters bitwise一致、block loss .993135→.945075、固定GT probe .228320→.154699、short monitor .551610→.509668、teacher不変、peak allocated313527808B。scopeは2train clips/15 short-window monitorで精度評価ではない。 |
| 2026-10-02 | 21:25:02 JST+0900 | root | 手順 10 🧪 **テスト**: 10→2 | 現行ソースGPU smoke・resume証明success。98 trainable tensorsとoptimizer/RNG/counters bitwise一致、block loss .993135→.945075、固定GT probe .228320→.154699、short monitor .551610→.509668、teacher不変、peak allocated313527808B。scopeは2train clips/15 short-window monitorで精度評価ではない。 |
| 2026-10-02 | 21:25:02 JST+0900 | root | 手順 10 🛠 **エラー時対処**: O | 現行ソースGPU smoke・resume証明success。98 trainable tensorsとoptimizer/RNG/counters bitwise一致、block loss .993135→.945075、固定GT probe .228320→.154699、short monitor .551610→.509668、teacher不変、peak allocated313527808B。scopeは2train clips/15 short-window monitorで精度評価ではない。 |
| 2026-10-02 | 21:25:02 JST+0900 | root | 手順 9 進捗 | 再開後入力cache進行中。fallback対象same clip start74、tracks10、valid_gt_count160で保存確認。cache件数は上記jq結果。 |
| 2026-10-02 | 21:29:28 JST+0900 | root | 手順 11 🖐 **操作**: v2の未学 | opset18 fused v2 copy-fused parity output maxima <1e-6–6.1e-5 across F1/8/31/600,N1/900; CPU tests/full suite 386 pass. GPU CUDAExecutionProvider 12/12 incl allF/N900 + real monitor subsets, output errors ≤1.23e-4, CPU profile nodes integer-only. export/graph correctness only, accuracy untested. |
| 2026-10-02 | 21:29:28 JST+0900 | root | 手順 11 🔎 **確認**: dynam | opset18 fused v2 copy-fused parity output maxima <1e-6–6.1e-5 across F1/8/31/600,N1/900; CPU tests/full suite 386 pass. GPU CUDAExecutionProvider 12/12 incl allF/N900 + real monitor subsets, output errors ≤1.23e-4, CPU profile nodes integer-only. export/graph correctness only, accuracy untested. |
| 2026-10-02 | 21:29:28 JST+0900 | root | 手順 11 🧪 **テスト**: fusi | opset18 fused v2 copy-fused parity output maxima <1e-6–6.1e-5 across F1/8/31/600,N1/900; CPU tests/full suite 386 pass. GPU CUDAExecutionProvider 12/12 incl allF/N900 + real monitor subsets, output errors ≤1.23e-4, CPU profile nodes integer-only. export/graph correctness only, accuracy untested. |
| 2026-10-02 | 21:29:28 JST+0900 | root | 手順 11 🛠 **エラー時対処**: e | opset18 fused v2 copy-fused parity output maxima <1e-6–6.1e-5 across F1/8/31/600,N1/900; CPU tests/full suite 386 pass. GPU CUDAExecutionProvider 12/12 incl allF/N900 + real monitor subsets, output errors ≤1.23e-4, CPU profile nodes integer-only. export/graph correctness only, accuracy untested. |
| 2026-10-02 | 21:32:37 JST+0900 | root | 手順 13 進捗 | strict window32 full-train/validation cache作成開始。dedicated w32 config validation済、Desktop free333GB、150GB budget、100GB reserveでstarts. Window16後半（1008/1027）と短時間並行。 |
| 2026-10-02 | 21:35:12 JST+0900 | root | 手順 12 進捗 | window16 1027/15 strict manifests complete. R0 GT-only full-cache smoke/profile candidate batch discovery starts to choose common safe microbatch before comparative 200-step pilots. |
| 2026-10-02 | 21:39:11 JST+0900 | root | 手順 9 🖐 **操作**: 新conf | window16 cache complete: train1027/1027 unique exact split count; validation15/15; every record positive anchor count and reprojection≤0.001px; 2 fallback clips logged (same clip starts74,113); median tracks206; 37GB/150GB cache budget; desktop free328GB (>100GB reserve). Full unit suite 386 passed; fallback tests27 passed. |
| 2026-10-02 | 21:39:11 JST+0900 | root | 手順 9 🔎 **確認**: 全件anc | window16 cache complete: train1027/1027 unique exact split count; validation15/15; every record positive anchor count and reprojection≤0.001px; 2 fallback clips logged (same clip starts74,113); median tracks206; 37GB/150GB cache budget; desktop free328GB (>100GB reserve). Full unit suite 386 passed; fallback tests27 passed. |
| 2026-10-02 | 21:39:11 JST+0900 | root | 手順 9 🧪 **テスト**: 先に2c | window16 cache complete: train1027/1027 unique exact split count; validation15/15; every record positive anchor count and reprojection≤0.001px; 2 fallback clips logged (same clip starts74,113); median tracks206; 37GB/150GB cache budget; desktop free328GB (>100GB reserve). Full unit suite 386 passed; fallback tests27 passed. |
| 2026-10-02 | 21:39:11 JST+0900 | root | 手順 9 🛠 **エラー時対処**: f | window16 cache complete: train1027/1027 unique exact split count; validation15/15; every record positive anchor count and reprojection≤0.001px; 2 fallback clips logged (same clip starts74,113); median tracks206; 37GB/150GB cache budget; desktop free328GB (>100GB reserve). Full unit suite 386 passed; fallback tests27 passed. |
| 2026-10-02 | 21:47:12 JST+0900 | root | 手順 12 進捗 | User timebox30min + necessary additional training only: R2 block100+FT200 pilot launched via nohup bash temp/run_kd_additional_r2.sh, effective batch32/microbatch32, strict train1027/monitor15/window16. Automatic full-frame monitor15 evaluation of best checkpoint follows. R0/R1 200-step pilots, final800×seeds and32adaptation remain pending; no unverified best-method/final-quality claim. Existing window32 cache and minival downloads continue. Resume uses same script and last.pt; no artifacts deleted. |
| 2026-10-02 | 21:50:00 JST+0900 | root | 手順 12 進捗 | Additional-training early stopping implemented after user request. RED9fail -> stagedCLI52PASS; all unit395PASS/31warnings/18.69s, ruff/ty and bash syntax PASS. First nohup launcher exited before training created artifacts; durable exec session90443 now runs launcher2866354/python2866360. At21:50:00 block19/100, FT0/200, identity.early_stopping_enabled=true. Stop rule: GT monitor every10, patience5, min_delta.001; block excluded; minimum-loss checkpoint retained and automatic full-frame monitor15 evaluation follows. source change is deliberate; old smoke resume proof remains historical and is not current-source proof. docs/kd_additional_training_20261002.md contains commands. Prior launch record timestamp21:47:12 is inaccurate: attempted nohup started21:46:53; successful early-stop launch21:48:29. |
| 2026-10-02 | 22:02:44 JST+0900 | root | 手順 14 進捗 | Action counter40 reminder/reset. New user request: add lightweight student column to demo; no new tests allowed. Plan independent non-overwriting student-mode output with SHA provenance/shared prepared frontend, existing tests + real render smoke. Current KD training/evaluation remains independent; not final DoD completion. |
| 2026-10-02 | 22:10:34 JST+0900 | root | 手順 14 進捗 | Student comparison demo narrow task complete under separate workdoc_Oct02-2026_student_comparison_demo.md (all boxes/DoD fulfilled). --mode with-student preserves3column/default and old4video SHA, independentDesktop KD output. RealDriveTrack31F4methods+2560x1080H264/fullDecode+snapshotPASS; existing11tests4.18s andruff/tyPASS; no newdemo tests. R2training200done/beststep180 shortGT.292478; fullmonitor15complete withmacro metricAJteacher.268133/student.196439 and3DAJteacher.148993/student.149651. These are known validation, notblind minival nor final quality parity. FormalR0/R1pilot andfullmulti-seed/long32 remain pending. |
| 2026-10-02 | 22:27:04 JST+0900 | root | 手順 14 進捗 | Followupcomplete:4column ALL3scenes481F Desktopcomparison_all fullDecodePASS; short-only completion corrected per user. Existingrelease gets9newexperimentalKDassets fused615838param/ONNX2.55MB+CPU/GPU12case reports,9redownloadSHA verified,old19assetSHA intact,notesadditive/teacherstillrecommended. PublishedsourcezipbecauseKDcode notinoldtag. docs/kd_metrics_release_20261002.md withprimaryPO2Dnotmeasured/TAP15knownscore differences. Separatedemoworkdocsteps1–10/DoD allcomplete; overallKDparent acceptance remains incomplete (formalR0/R1/fullmulti-seed/32adapt/minival150). No newdemo tests, no staging/commit/push. |
