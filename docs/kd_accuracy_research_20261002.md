# refiner KD精度改善：研究調査と現行学習の診断

調査日：2026-10-02。初稿の実績確認時刻：20:10:00 JST+0900（11:10:00 UTC）。
対象ブランチ：`distill/refiner-kd-onnx-20261002`。
対象：学習済み8,117,988パラメータのMamba-3 refinerから、650,000以下のONNX推論可能なrefinerへの圧縮。
関連計画：[初期KD設計](knowledge_distillation_plan_20261002.md)、ローカルの親作業書 `temp/workdoc_Oct02-2026_refiner_kd_onnx.md`、追加計画 `temp/workdoc_Oct02-2026_kd_accuracy_improvement.md`。

この文書は22件の一次研究と現行のコード・実験を照合した調査資料である。論文で確かめられたこと、現リポジトリで確かめたこと、これから比較する仮説を分ける。新query・local/global student・段階的KDによる精度改善は、この初稿時点では未検証。最終150本評価と親D-1〜D-8の完了を主張しない。

## 1. 今回の判断

優先するのは、queryとGTの対応修正、使用されないprojectionの整理、時間局所表現と大域表現への容量再配分、teacher-forced block事前整合、GTを保護した選択的KDである。単純にKDの損失項や係数を増やす前に、訓練信号と有効容量を整える。

| 優先度 | 改善対象 | 現在の根拠 | 検証する仮説 |
| :--- | :--- | :--- | :--- |
| 1 | window内queryの正しい再anchor | 時刻だけ変更するfallbackにxy/GT不整合、短windowのtrackが少ない | 正しいqueryと監督密度の回復が改善する |
| 2 | unused projectionを厳密に除去 | 現経路で272,440パラメータが出力に寄与しない | 同出力を348,062で保持し、余った予算を時間表現へ回せる |
| 3 | local temporal＋normalized global＋FFN | 現studentの全F性能低下、軽量Vision SSMは局所/大域/FFNを組み合わせる | 同程度のdeployサイズで長Fに必要な表現が増える |
| 4 | 段階的蒸留 | joint feature追加は悪化、回帰KDとMOHAWKにstage分離の実証 | random mixerからのend-to-end模倣よりblock事前整合がよい |
| 5 | teacher-better／gradient gate | GT-only対照が短loss・seed42全F metric-AJでA1よりよい | 教師の誤りとGTに逆らうKDを抑制できる |
| 6 | 8→16→32の混合curriculum | 訓練8Fと評価全Fが異なる | 長系列へ滑らかに移行し、短系列への過適応を減らせる |
| 条件付き | teacher assistant／段階的state圧縮 | 13倍の容量差と異種mixer | 上記の小さな改善で不足したとき、追加訓練費を使う価値がある |

「unusedの削除」は精度を変えない等価な整理が目標。「normalized global」「local branch」「curriculum」「KD gate」はモデル・学習を変える比較実験であり、同一関数の変換ではない。

## 2. 現行実績と原因の切り分け

### 2.1 短lossが良くても、全Fのtracking精度は一致していない

以下は同じmonitor15本、同じ全F入力とflow visibilityによる実測。macroは3 subsetを等重みで平均する。公式minival150本の最終受入結果ではない。

| run | best短window GT loss | 全F 3D-AJ | 全F metric-AJ |
| :--- | ---: | ---: | ---: |
| teacher | student短lossとの直接比較不可 | 0.1489932590 | 0.2681328770 |
| A1 seed42 | 0.1156595416 | 0.1023926706 | 0.2232333612 |
| A1 seed43 | 0.1147727948 | 0.1058393695 | 0.2437281765 |
| A1 seed44 | 0.1097330624 | 0.1063759005 | 0.2411640121 |
| A0 GT-only seed42 | 0.1138488412 | 0.1010767302 | 0.2277285263 |

根拠はローカルの `result/refiner_kd_20261002/full_{s42,s43,s44,control_s42}/training_status.json` と `monitor_full_{s42,s43,s44,control_s42}/accuracy_report.json`。初稿時にJSONの値、`status=complete`、`expected=15`、`scope=known_sequence_overlap_not_blind`を確認した。seed42 A1は240 steps/7,680 clips、GT-onlyは300 steps/9,600 clipsでearly stop。最大予算・停止規則は同じだが実stepsが違うため、完全に同じ計算量の比較とは呼ばない。

A0〜A4 pilotのbest GT lossは順に0.1212941284／0.1156595416／0.1245791333／0.1237190531／0.1288348157。A1のpilot改善は、その後のGT-only対照に対する最終KD効果や、全F精度維持を証明しない。A2はfeature、A3はtemporal、A4はauxを追加したjoint訓練である。

短lossのランキングと全F AJが異なる以上、改善案の採否は15本の全F monitorを第一にする。seed43の良い結果を見て最終seedを付け替えない。既存の固定seed42基準とseeds43/44の再現性報告を保つ。

### 2.2 queryの時刻と座標を一緒に変える必要がある

監査対象の準備経路は `scripts/prepare_refiner_kd.py` からdatasetへ `reanchor_window=False` を渡す。`src/mamba3_tracker/data/dataset.py` の旧fallbackは、crop内に元のquery anchorが一つもない場合に時刻だけをcrop開始時刻へ変更する。xyは元anchorの座標のままなので、同じ時刻のGTとの対応が崩れる。この問題はKDの理論ではなく、現実装の入力整合性の問題である。

既存window8のtrain manifest1,027本を監査した結果：N中央値9、N≤8が481本。subset別ではADT585本で中央値6、DriveTrack357本で37、PStudio85本で12。全N=256の15本ではanchorsが全て0であり、fallback対象が含まれる。

CPUでprepared inputのKと同anchor時刻のGT XYZを再投影して `uv[anchor]` と比べたところ、7本で中央値27〜249pxの差があった。単位は896へresize後のpixel。例は `adt/Apartment_release_work_seq139_2.npz`、crop開始278、条件を満たすGT26点、中央値248.623px。prepared payloadはローカル `/workspace/vmamba3_eval/refiner_inputs/window8/train/adt__Apartment_release_work_seq139_2.pt`。delta_uvの上限は2pxなので、この大きさのquery不整合はrefinerのUV headだけでは吸収できない。

このCPU検算は監査エージェントの実行stdoutで確認され、初稿時に独立JSONへ保存されていない。再現可能な新入力監査をpayload/manifestへ残すことを追加計画の必須項目にした。7本の検算を全1,027本の誤差分布とみなさない。

新policyでは、crop内でGTが可視・finite・正depth・画像範囲内の時刻を探し、その時刻のGT XYZをKで投影してxyとtを両方作る。訓練queryをGTで生成する操作は、推論時にGTをrefinerへ渡すことを意味しない。新policy・precision・windowを新namespaceに固定し、旧cacheを流用しない。teacherも旧訓練経路の問題を継承した可能性があるため、teacherに無条件で従う改善は避ける。

### 2.3 620,502のうち272,440は現経路で使用されていない

監査した契約は `collapse`、`two_pool=true`、dim128、state64、4 heads、2 layers。`third_party/visionMamba3/src/visionmamba3/cross_attention.py` と同ディレクトリの `projections.py` のpacked projectionは652 outputsをまとめて作るが、q/q2はC側だけ、KVはB/V/Δ/A側だけを参照する。

| 1 layerの内訳 | parameters |
| :--- | ---: |
| packed bundle3個＋out＋pool2 gateの登録総数 | 270,505 |
| q/q2の使用されないweight行 | 50,688＋50,688 |
| KVの使用されないweight行 | 33,280 |
| 使用されないnorm/bias等 | 1,564 |
| 現forwardへ寄与しない合計 | 136,220 |
| 実効合計 | 134,285 |

二層で272,440削減、共通geometry/head79,492を加えた実効合計は348,062。監査エージェントがCPU forward/backward、pool2 gate=0.1で未使用packed行のstrict zero gradientと残りのno-gradを確認した。これは現契約での構造的不要部分であり、別のattention経路・flagsまで不要と一般化しない。

等価なminimal版の完了条件は、使用行・BCNorm・biasの正しい移植、gate zero/nonzero、F1〜600、全4出力のparity、track permutation不変。登録パラメータだけを数えた「620kの容量」と、現forwardの実効容量を分けて考える必要がある。約272kの余白をFFNやlocal temporalへ回す新v2は、出力同等性のあるpruningとは別の訓練実験になる。

### 2.4 今の出力KDは、少なくともこのstepでは非常に小さい

`result/refiner_kd_20261002/full_s42/metrics.jsonl` のstep190：GT loss0.1493862923、residual KD0.0003714471、XYZ KD0.0004765576。係数0.5／0.25を掛けた追加lossは0.000304863。

ログ上のterm gradient normsはGT8.714673、residual0.0123924、XYZ0.0180642。三角不等式による合成KD gradient normの上界は

```text
0.5 * 0.0123924 + 0.25 * 0.0180642 = 0.01071226
0.01071226 / 8.714673 = 0.00122922 ≈ 0.122922%
```

方向のcosineはこのログにない。したがって「KDがGTを支配して精度を壊した」とは言えず、「このstepでKDの有効信号は極めて弱い」という診断になる。norm差だけから固定係数を数百倍にする根拠もない。教師の正誤、マスク通過率、方向、代表stepでの寄与を計測してからboundedなscale調整を比較する。

### 2.5 長さ依存と独立性は別の問題

pool1の末端への累積decay、pool2の非正規化sumはFに依存する。window8から全Fへ適用すると出力scaleや参照範囲が変わる可能性がある。ただし「これがAJ低下の原因」という反実仮想実験は未実施。入力queryの問題、実効容量、長さ依存を個別に変える対照が必要である。

既存splitのclip重複は0だが、保守的sequence groupingではmonitor14/15、minival141/150がtrainと同群。教師の全過去学習履歴も完全には証明されていない。根拠は `result/refiner_kd_20261002/preflight/inventory.json` と親作業書の監査記録。これらを完全未見testと呼ばない。公式全150本のpaired比較は必須のまま保持し、minivalを新hyperparameterの選択には使わない。

## 3. 一次研究の分類と採否

以下の「事実」は一次論文または著者公式コードからの短い要約。「適用」は今回の設計判断・仮説。ONNX難度は本リポジトリに移植する場合の見積であり、各論文の公式ONNX検証結果を意味しない。直接引用は使用しない。

### R01. 回帰KD：Attentive Imitation／Hint Training

**一次資料・区分：** [Distilling Knowledge From a Deep Pose Regressor Network](https://arxiv.org/html/1908.00858v1)、ICCV 2019、§4–5、Table 1。

**事実：** Visual Odometry回帰では、無条件teacher模倣がGT-onlyより悪化する比較がある。teacherのGT誤差で模倣とhintを重み付けするAIL/AHT、hint→outputの分離訓練を評価し、jointより分離が良いと報告。最大92.95%パラメータ削減は同タスクでの結果。

**適用：** `L_GT + λ w_T L_KD` を使い、GT側は残す。比較対象のteacher-upper-boundはstudentのGT誤差がteacher以下ならKDを0にする。現confidence floor0.05を必須とせず、teacher-better gateをdetachして試す。

**限界・ONNX：** 論文の範囲ベース信頼度をそのまま移植せず、3D尺度・可視マスクに合わせる。loss/gateはtraining-only、ONNX追加op無し。**採否：優先採用して比較。**

### R02. MOHAWK：異種SSMのblock事前整合

**一次資料・区分：** [Transformers to SSMs: Distilling Quadratic Knowledge to Subquadratic Models](https://arxiv.org/html/2408.10189v1)、NeurIPS 2024、§4.2、Table 3。

**事実：** mixer行列→teacher入力を共有したblock出力→end-to-endの3段階。固定5B tokensでPhi-Mambaの平均accuracyはoutput stageのみ54.5、hidden＋output62.3、全stage62.7。共通重み移植も用いる。

**適用：** `Σ_l ||T_l(u_T[l-1]) - S_l(u_T[l-1])||²` で128次元入出力の2 blocksを事前整合。common geometry/headsをfreezeし、その後student自身のhiddenでfinetuneする。joint A2と異なる初期化実験である。

**限界・ONNX：** LLMの結果は13倍圧縮のtracking保証ではない。Mamba3をMamba2のscalar SSDと同一視せず、matrix-stageは保留。block-stageはtraining-only。**採否：優先。**

### R03. FitNets：hintで初期化してから最終訓練

**一次資料・区分：** [FitNets: Hints for Thin Deep Nets](https://arxiv.org/abs/1412.6550)、ICLR 2015、stage-wise hint training。

**事実：** 薄く深いstudentのguided layerをteacher hintへregressorで合わせ、その初期化から全体を蒸留する。hint trainingと最終KDを段階化する。

**適用：** 最終headsのGT学習とhidden整合を最初から競合させない設計根拠。今回は2層同士なので、深さを増やす根拠としては使わない。

**限界・ONNX：** 画像分類での証拠。training-only regressorは公開artifactから除く。**採否：R02の段階設計へ反映、別の巨大hint moduleは追加しない。**

### R04. Teacher Assistant：capacity gapの橋渡し

**一次資料・区分：** [Improved Knowledge Distillation via Teacher Assistant](https://arxiv.org/abs/1902.03393)、AAAI 2020。

**事実：** 大きなteacherを小さなstudentへ直接蒸留すると性能が落ちる場合をCNN/ResNet分類で評価し、中間容量のassistantによる多段KDを提案。

**適用：** 8.118M→同系assistant約1.5〜2.5M→最終620kを候補にする。このサイズ範囲は我々の候補であり、論文が今回向けに指定した値ではない。

**限界・ONNX：** assistant訓練費が増え、teacherが不正確なら中間モデルにも伝わる。deploymentは最終studentだけなのでassistantのONNXは不要。**採否：query修正・R02で不足した場合の第2候補。**

### R05. GT／auxのgradient similarity

**一次資料・区分：** [Adapting Auxiliary Losses Using Gradient Similarity](https://arxiv.org/pdf/1812.02224)、arXiv 2018／改訂2020、式3–6、Algorithm 1。

**事実：** main taskを助けるときだけauxを使う設定。共有gradientを `g_main + max(0,cos(g_main,g_aux))*g_aux` とし、binary gateも比較。小さなstep等の条件下でmainの収束を論じるが、精度改善の一般保証はない。

**適用：** GTを主目的、KD/feature/occlusionをauxとする今回に合う。まずnormとcosineを計測し、負方向のauxを止めるarmを比較。aux専用projectorの更新と共有mixersのgateを分ける。

**限界・ONNX：** minibatch cosineはノイズを持つ。AdamW＋clipへ理論保証を外挿しない。training-only。**採否：診断は必須、gateは独立比較。**

### R06. GradNorm：勾配の大きさを適応調整

**一次資料・区分：** [GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks](https://proceedings.mlr.press/v80/chen18a.html)、ICML 2018。

**事実：** lossの重みを、共有層のgradient magnitudeと各taskの相対学習速度に基づき調整するmulti-task手法。回帰・分類の組合せを評価。

**適用：** 0.123%というKD上界を見て、重みを固定のまま議論しない診断根拠。今回はGTだけが主目的なので、全taskを等しく強める設計はそのまま採用しない。bounded ratio補正は独自の比較armと明記する。

**限界・ONNX：** normを揃えても逆方向gradientは残る。training-onlyだが追加backwardの費用がある。**採否：計測を採用、全面的な自動等重み化は保留。**

### R07. PCGrad：衝突gradientの射影

**一次資料・区分：** [Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782)、NeurIPS 2020。

**事実：** task gradientsの内積が負なら、衝突相手に直交する平面へgradientを射影し、multi-task supervised/RLを評価する。

**適用：** cosineが負の場合の候補。GTを変更せずKD側だけを射影する設計は、対称PCGradそのものとは区別する。

**限界・ONNX：** full param gradientsの保存・複数backwardが必要。GT主目的にはR05の非対称gateが先に適する。推論graphへの影響無し。**採否：R05で不十分な場合の対照。**

### R08. Sequence Length Warmup

**一次資料・区分：** [The Stability-Efficiency Dilemma: Investigating Sequence Length Warmup for Training GPT Models](https://arxiv.org/abs/2108.06084)、NeurIPS 2022。初期arXiv版の題はCurriculum Learning…。

**事実：** 長系列・大batch・大LRとAdamのgradient variance外れ値の関係をGPTで調査し、sequence length warmupを評価。短長を急に切り替える対照と滑らかな移行は同じではない。

**適用：** 8/16→16/24→16/32の混合を候補にし、最後まで一部短windowを残す。新長さでLRとeffective batchを同時に急増させない。

**限界・ONNX：** GPTの安定性がvision trackingのAJ向上を保証しない。cropごとにanchor・z_ref・teacherを再整合する。training-only。**採否：32-frame入力後に比較。**

### R09. Feature projectorとnormalization

**一次資料・区分：** [Understanding the Role of the Projector in Knowledge Distillation](https://arxiv.org/pdf/2303.11098)、arXiv 2023、Table 1、§projector analysis。

**事実：** feature次元が同じでもprojectorが役割を持ち、normalizationがprojectorの特異値・情報損失・student性能へ影響することを分類/検出で分析。大capacity gapでは完全feature一致が難しい。

**適用：** A2の失敗をfeature KD一般の否定にしない。直接block出力回帰とtraining-only projector付き回帰を分け、direction/scale誤差とprojector collapseを観測する。

**限界・ONNX：** 論文で良かったBatchNormを、小さな可変bucket/可視maskへ無検証で移植しない。projectorは公開artifactから除く。**採否：R02で直接回帰を先に試し、必要時比較。**

### R10. TOR：annotation outlierへの回帰KD

**一次資料・区分：** [An Efficient Method of Training Small Models for Regression Problems with Knowledge Distillation](https://arxiv.org/pdf/2002.12597)、MIPR 2020、§4.1。

**事実：** teacherとannotationの差からlabel outlierを推定し、GT回帰とteacher回帰の2 outputsを使う。MPIIGaze/Multi-PIE等のnoisy regressionで評価。

**適用：** 回帰KDの参考にはなるが、teacherとGTの不一致をGT誤りとする証拠が現在ない。bad teacher outputを除くR01とは方向が違う。

**限界・ONNX：** GTをteacherに合わせて捨てると受入の目的を変える恐れがある。追加headsはtraining-onlyにできるが、採用根拠がない。**採否：現時点不採用。**

### R11. CoTracker3：実動画pseudo-labelとheadの扱い

**一次資料・区分：** [CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos](https://openaccess.thecvf.com/content/ICCV2025/papers/Karaev_CoTracker3_Simpler_and_Better_Point_Tracking_by_Pseudo-Labelling_Real_Videos_ICCV_2025_paper.pdf)、ICCV 2025、§3.1。

**事実：** 複数frozen trackersをbatchごとに選び、real videosのpseudo tracksで訓練。pseudo-label段階ではconfidence/visibilityを監督しない方が安定し、対応linear headsをfreezeする。

**適用：** 未監督teacher visibilityをKD対象にしない現在の判断を支持する参考。複数教師の強みを使う場合も3D coordinate/depth尺度の整合が必要。

**限界・ONNX：** 2D trackersのpseudo labelsを3D GTへ直接置換しない。現在の評価visibilityは共通flowなので、aux occlusion精度の改善はAJへ直接入らない。teacher ensembleは訓練費が増える。**採否：headの制約を採用、外部tracker ensembleは保留。**

### R12. Vision Mamba（Vim）：双方向SSM

**一次資料・区分：** [Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model](https://arxiv.org/abs/2401.09417)、ICML 2024。

**事実：** positional情報とbidirectional SSMを組み合わせてvisual representationを作り、分類・検出・segmentation等を評価する。

**適用：** offline trackingでは未来側contextも使えるが、現在のtwo-pool globalが順序の局所的変化をどれだけ保持するかとは別問題。時間の前後処理を試す根拠の一つ。

**限界・ONNX：** 画像flatten scanをpointの順序へ移植するとN permutation契約を壊す。公式scan kernelは標準ONNX化に追加実装が要る。**採否：概念参考、full Vim backboneは不採用。**

### R13. VMamba：SS2Dと画像局所性

**一次資料・区分：** [VMamba: Visual State Space Model](https://arxiv.org/abs/2401.10166)、NeurIPS 2024。

**事実：** 2D selective scan（SS2D）の複数方向処理で、画像の空間構造と大域contextを扱うVSS blocksを設計する。

**適用：** scanの形式は入力の構造に合わせる必要がある。今回の圧縮対象はDINO後の時間×track refinerであり、DINO画像backboneをVMambaへ置換するタスクではない。

**限界・ONNX：** trackに空間raster順を要求しない。全2D scan導入はサイズ・依存・export費が増える。**採否：設計背景として参照、backbone導入は範囲外。**

### R14. VSSD：non-causal SSD＋local processing

**一次資料・区分：** [VSSD: Vision Mamba with Non-Causal State Space Duality](https://openaccess.thecvf.com/content/ICCV2025/papers/Shi_VSSD_Vision_Mamba_with_Non-Causal_State_Space_Duality_ICCV_2025_paper.pdf)、ICCV 2025。

**事実：** causal SSDをnon-causal visual処理へ変えるため、tokenとhiddenの相互作用の絶対量を取り除き相対重みを保持する。モデルはglobalなNC-SSDだけでなくlocal processing/FFNも含む。

**適用：** 現コードの名称がVSSDでも、二つのpoolを持つrefinerは論文backbone全体と同じではない。時間local branchとFFNを有効予算内で足す設計へ反映する。

**限界・ONNX：** 我々のnormalized temporal poolは独自適用であり、論文の等価再現と呼ばない。MatMul/Softmax/Reduce/Convなら標準opで表せるが動的F実測が必要。**採否：local＋globalの構成原理を優先。**

### R15. EfficientViM：compressed hiddenでchannel mixing

**一次資料・区分：** [EfficientViM論文](https://arxiv.org/abs/2411.15241)、CVPR 2025。[著者公式モデルコード](https://github.com/mlvlab/EfficientViM/blob/main/classification/models/EfficientViM.py)。

**事実：** 入力tokenごとのprojectionがSSDのruntimeを支配することから、圧縮hidden内でchannel mixingを行うHSM-SSDを設計。公式blockはtoken軸の重み正規化、DWConv、FFNも用いる。

**適用：** 未使用projectionを減らした上で、128次元のlocal temporal＋global hidden mixingへ容量を回す。正規化による長さ依存の変化を独立比較する。

**限界・ONNX：** ImageNetの速度/accuracyをRTX5090のrefinerへ数値外挿しない。hidden内channel mixingと今回のFFN追加は同じ操作ではない。標準op設計の難度は低〜中。**採否：最優先の構造参考。**

### R16. TinyViM：低周波globalと高周波localの分離

**一次資料・区分：** [TinyViM: Frequency Decoupling for Tiny Hybrid Vision Mamba](https://openaccess.thecvf.com/content/ICCV2025/papers/Ma_TinyViM_Frequency_Decoupling_for_Tiny_Hybrid_Vision_Mamba_ICCV_2025_paper.pdf)、ICCV 2025。

**事実：** hybrid Conv-MambaでMambaが主に低周波情報を扱う分析から、Laplace mixerで低/高周波を分け、低周波をMamba、細部をmobile-friendly convolutionへ送る。層ごとのfrequency配分も設計する。

**適用：** global poolだけでなく時間local convolutionで短い動き・残差変化を保持する仮説を支持する。まず周波数変換無しのlocal/global対照を行う。

**限界・ONNX：** 画像周波数と3D trajectoryの時間周波数は同一ではない。強い平滑化は急な運動や遮蔽境界を損なう。dynamic時間のfrequency splitは追加検証が必要。**採否：構成原理を採用、Laplace split自体は保留。**

### R17. MobileMamba：複数受容野と実latency

**一次資料・区分：** [MobileMamba: Lightweight Multi-Receptive Visual Mamba Network](https://openaccess.thecvf.com/content/CVPR2025/papers/He_MobileMamba_Lightweight_Multi-Receptive_Visual_Mamba_Network_CVPR_2025_paper.pdf)、CVPR 2025。

**事実：** global WTE-Mamba、multi-kernel local convolution、identityを組み合わせるMRFFIを提案。小FLOPsだけでは高throughputにならない問題を扱い、KDや長期trainingを含む設定で評価する。

**適用：** 全channel・全layerへ重いSSMを入れる代わりに、local/global/identityの容量配分を考える。訓練budgetの違いを構造の純粋な効果と混同しない。

**限界・ONNX：** wavelet/scan等の公式network全体導入はexport工数が大きい。まず標準Conv1D＋global reductionへ限定する。**採否：受容野配分と実速度重視を採用、full backboneは不採用。**

### R18. LocalMamba：windowed selective scan

**一次資料・区分：** [LocalMamba: Visual State Space Model with Windowed Selective Scan](https://arxiv.org/abs/2403.09338)、arXiv 2024。今回確認した資料では会議区分は確定していない。

**事実：** 画像flattenによって近傍tokenが離れる問題へwindow内local scanとglobal処理を提案し、層ごとのscan選択も扱う。

**適用：** temporal neighborhoodを明示的に残す必要性の参考。今回のF順序は時間として既に意味があるので、画像のscan direction searchを追加する必要はない。

**限界・ONNX：** dynamic Fのwindow boundary/paddingは仕様を固定する。Nを跨ぐwindowはchunk/permutation契約に反する。**採否：local設計の補助根拠、scan searchは不採用。**

### R19. MambaVision：hybrid Mamba／attention

**一次資料・区分：** [MambaVision: A Hybrid Mamba-Transformer Vision Backbone](https://openaccess.thecvf.com/content/CVPR2025/html/Hatamizadeh_MambaVision_A_Hybrid_Mamba-Transformer_Vision_Backbone_CVPR_2025_paper.html)、CVPR 2025。[著者repo](https://github.com/NVlabs/MambaVision)。

**事実：** Mambaとself-attentionを組み合わせるvisual backboneで、長距離空間関係を含むhybrid設計を評価する。

**適用：** pure poolに必要な表現が不足する場合のlimited-window attention／latent mixing候補。ただしこれは論文backboneの直接移植ではない。

**限界・ONNX：** full F600のattentionはF²を増やす。standard MatMulでexport可能でもruntime memory要件を満たすとは限らない。**採否：現local/globalで不足した場合にlatent/window限定で再設計。**

### R20. RepVGG：訓練branchを単一Convへ畳み込む

**一次資料・区分：** [RepVGG: Making VGG-Style ConvNets Great Again](https://openaccess.thecvf.com/content/CVPR2021/html/Ding_RepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.html)、CVPR 2021。

**事実：** training時のmulti-branch構造をstructural reparameterizationでplain Convへ変換し、training topologyとinference topologyを分ける。

**適用：** 1D depthwise kernel7/3/1とidentityの線形和を、padしたkernel7とbias合算へcopy-fuseする。非線形は枝の和の後に置く。

**限界・ONNX：** LayerNorm／SiLU／input-dependent gateはkernelへ融合しない。branchが異なるstride/group/channelなら単純和は不可。Conv1D標準opのexport難度は低。**採否：temporal branchの再パラメータ化へ採用。**

### R21. MobileOne：mobile latencyを重視するreparameterization

**一次資料・区分：** [MobileOne: An Improved One millisecond Mobile Backbone](https://arxiv.org/abs/2206.04040)、CVPR 2023。[公式fusionコード](https://github.com/apple/ml-mobileone/blob/main/mobileone.py)。

**事実：** Conv＋BN等の複数training branchesをinference時に融合し、parameter/FLOPs以外の実device latencyも評価する。

**適用：** 今回は小さな可変bucketなのでBN無しの同shape線形branch和をまず使う。train paramsとdeploy paramsを別集計し、fusion前後の数値を証明する。

**限界・ONNX：** iPhoneのlatencyをCUDA EPへ持ち込まない。BN無し版は我々の適用でありMobileOneそのものではない。**採否：copy fusionと実benchmarkの方法へ反映。**

### R22. CompreSSM：大きく始めて訓練中にstateを縮める

**一次資料・区分：** [The Curious Case of In-Training Compression of State Space Models](https://arxiv.org/abs/2510.02823)、arXiv初出2025-10-03／v4 2026-02-24。[著者公式repoのICLR 2026表記](https://github.com/camail-official/compressm)。

**事実：** Hankel singular valuesに基づくbalanced truncationを訓練中に使い、影響の低いstateを削除する。大きく開始して縮小する方法を、初めから小さいstateで訓練する対照と比較。公開main repoはLRU、Mamba実験は別repoに分かれる。

**適用：** state128→64やassistant→smallへの段階圧縮候補。容量を使いながら訓練し、最終deployだけ650k以下にする方針と整合する。

**限界・ONNX：** LTIのbalanced truncation保証をselective・input-dependent・Mamba3複素回転へ直接適用しない。VSSD二poolのstateも同一のLRU状態ではない。削除後の形状・重み変換・再訓練を明示的schemaで扱う。**採否：第2段階の研究候補、直ちに導入しない。**

## 4. 改善モデルと訓練方法の詳細案

以下は追加作業書で管理する比較案。まだ精度向上の成果ではない。

### 4.1 minimal版と新v2を分ける

minimal版は使用するprojection行だけを持ち、旧v1の出力をそのまま再現する。旧checkpointから使われる行を明示コピーし、暗黙のreshape・drop・fusionをしない。全4出力、zero/nonzero second-pool gate、F1/8/31/600、N順序変更を検証してから「等価な削減」とする。

v2はminimal版で空いた約272kを、各blockのlocal temporal DWConv1D、128→512→128 FFN、residual/LayerNormへ割り当てる。global poolはF方向のみ、N方向を混ぜない。learned residual scaleを小さく初期化して新branchの急な出力変化を抑える。deploy600k〜625kを目安に、実集計650k以下を必須とする。

長さに依存する絶対量を抑えるglobal正規化の候補は、累積終端decayを使わず `softmax(log(delta+epsilon)+delta*A, F)`、second poolはpositive weights／weight sumで正規化する方式。これはVSSD/EfficientViMから着想した独自のtemporal適用。旧v1の関数と等価ではなく、順序情報をlocal branchへ任せることの限界も全Fで測る。

```text
各trackの同じ128-dim hidden
  → local temporal（7/3/1/identity training branches、deploy kernel7）
  → normalized global hidden mixing（Fのみ）
  → FFN（128→512→128）
  → 既存geometry/head → 8入力4出力の公開student
```

### 4.2 block事前整合

teacherのcommon入力encoderとreadoutをfreezeし、teacherの各block入力 `u_T[l-1]` をteacherと対応student blockへ同じように渡す。2 blocksを独立に合わせ、下流studentの誤差が上流alignmentを曖昧にしないようにする。

開始候補は100 stepsの事前整合。raw output MSEとscale/normalized差を記録し、単にfeature projectorだけがteacherを近似していないか確認する。両blockは128次元なので直接出力回帰を最初のarmにする。projectorを使う場合はtraining-onlyとして費用を別記し、deploy hiddenが直接teacher-spaceを再現することとは区別する。

事前整合後はoptimizerを新phaseとしてresetし、通常student自身のhiddenでfinetuneする。common/headsのfreeze解除時刻、LR、GT/KDの係数をconfigへ固定する。teacher tensorsとcommon freeze対象が変わっていないことをbefore/after hashとnamed testで検証する。

### 4.3 selective KDとgradient診断

有効可視位置でGT尺度正規化したteacher/student誤差を `e_T,e_S` とし、初期比較は `gate=detach(e_T < e_S)`。教師がGTから大きく外れる点のconfidenceをゼロまで下げる。GT lossは常時残し、visibility teacher logitsは使わない。

GT/KDのnorm、cosine、gate通過率、subset別通過率、dlog/duv clamp率、actual microbatchとNを記録する。norm0ならratio補正0、NaNは失敗にする。normを合わせる場合の係数は上限付きの別armで、gradient similarity gateも独立armにする。複数改善を一度に入れた結果だけから原因を断定しない。

事前整合時の教師入力hiddenには可視/不可視両contextが含まれる。end-to-endのGT可視mask、finite query mask、padding maskと、hidden整合に使うmaskを別に記録する。遮蔽contextを全て捨てる設計と、誤った教師の遮蔽予測を強制する設計の双方を避けるため、mask policyをtestsとconfigで固定する。

### 4.4 長window curriculum

新policyでwindow16を先に生成し、必要な比較の8/32へ広げる。32から短cropを作る場合はquery anchorがcrop外へ出ないよう再生成し、camera K・query indices・GT尺度・z_refをcrop単位で再計算する。32-frame teacher labelsを8/16へ切り出して使わない。

混合例は、8/16中心→16/24混合→16/32中心。一部短windowを最後まで残す。初期8のみと、急な16/32切替と、混合の3つを無条件に全gridで走らせず、最初は16固定のR0/R1/R2から構造/KDの効果を判定する。64/fullF適応が必要なら計算費と入力容量を見積もり、追加作業書へ手順を加える。

### 4.5 32GB GPUの使い方

現在の短訓練は最大VRAM約5.48GBだった。これにはN中央値9という入力密度が関係する可能性がある。正しいreanchorでtrackが増える新入力では、改めてmicrobatch1/4/8/16/32と最大Nの実backwardをprofileする。

32GB使用許可は容量を無理に埋める要件ではない。2GB程度の余裕を維持し、安全で速いmicrobatchを採る。exact F/N bucketの制約と実batchを報告し、paddingやbatch変更でz_refが変わる場合はteacher/studentへ同じscalarを渡す。effective batchと学習率を同時に変えて精度効果を混同しない。

## 5. 比較実験と採否の規則

| arm | query | 構造 | 方法 | 何を切り分けるか |
| :--- | :--- | :--- | :--- | :--- |
| 旧baseline | 旧policy | v1 | A0/A1既存成果 | 過去結果の記録。新lossとの直接比較はしない |
| R0 | 修正policy | 旧v1 | GT-only | data修正の影響 |
| R1 | 修正policy | local/global v2 | GT-only | 有効容量の再配分・時間表現 |
| R2 | 修正policy | local/global v2 | block事前整合＋selective KD | staged KDの追加効果 |
| 条件付きR3 | 修正policy | 採用構造 | gradient gate／bounded ratio | KDの強さ・方向 |
| 条件付きR4 | 修正policy | 採用構造 | 16→32／mixed curriculum | 系列長の分布差 |

最初のR0/R1/R2は同seed42、固定train1,027／monitor15、同window16、effective batch32、同profile-safe microbatch、同finetune200-step予算。common初期値は一致させるが、異種random mixerまでbitwise同じとは主張しない。R2の事前整合100 stepsは追加費用として報告する。total compute一致の効果を検証するなら、R1にも追加100-step GT訓練の対照を別に用意する。

採否は全F monitor15のmacro metric-AJを第一とし、3D-AJと各subsetも併記する。選択config/checkpointはhashで固定。最良short GT lossだけで長Fの優位を判断しない。minivalの予測metricを選択前に読まない。選択後にseeds42/43/44・GT-only対照・32適応・最終ONNXへ接続する。

新queryで旧結果と前処理分布が変わることを明示する。旧teacherのGT監督条件も同じ問題を持った可能性があり、最終teacher超えやteacher同等が得られるとは現時点で言えない。

## 6. ONNX・精度・速度の完成条件

親protocol `result/refiner_kd_20261002/protocol.json` のSHA256は `e83a01247263c84c27273427578b8c8093b48c6a6de42b514dbe12d949941054`。以下は変更しない。

- deploy parameters≤650,000。teacher／aux／optimizerをpublic PT/ONNXへ含めない。
- 8入力4出力、FP32、opset18、標準domainのみ。B/F/N/depth H/W dynamic。
- copy fusionの数値一致はrtol1e-5／atol1e-6。PyTorch→ORTはrtol2e-4／atol5e-5。
- CPU/CUDAの全4出力、F1/8/31/128/257/300/600、B1/2、N1/7/32、F600 N900 chunk、全不可視、track permutationを確認。
- CUDA provider名だけで合格とせず、主要float演算の実CUDA profileを確認。CPUは整数shape処理まで。
- 公式minival全150本をteacher/student同入力でpaired評価。metric-AJと3D-AJのteacherからの低下macro≤.003、subset≤.005、paired sequence bootstrap10,000（seed42）の片側95%上限も同margin。
- 速度は精度と別判定。threads4、warmup20、trials100、p50/p95、同B/F/N/chunk、GPU同期、転送込み／model-only、VRAM、ONNX bytesを記録。model-only≤teacher0.7倍は目標で、parameter減少から達成を推測しない。

既存short studentのONNX/CUDA一致が成功していても、新v2のexportや精度の証明にはならない。新v2のGPU smoke／訓練／全F／最終150本を実測してから親DoDを更新する。

## 7. 調査で保留したものと次の判断

量子化はFP32精度差を解消する方法ではない。非構造sparsityは標準dense ONNXの実速度を自動的に上げない。任意SVDでMamba3 weightsをVSSDへreshapeすると、異なるstate dynamicsを保つ保証がない。これらは今回の最初の改善に含めない。

R0で大幅に改善すれば、data修正が主因だった可能性が高い。R1がR0を上回れば時間表現への容量配分を支持する。R2がR1を上回り、teacher-better gate／gradientの証跡が整えばKDの追加効果を支持する。全てを同時に変えた1 runだけで、論文由来の改善効果を断定しない。

local/global＋staged KDでも不足する場合は、teacher assistant、official Mamba3幅縮小、CompreSSMに着想した段階圧縮を別計画にする。新候補も650k上限・dynamic ONNX・最終AJ marginを維持する。未達時は基準を緩めず、何が未達かを記録する。

この初稿の担当はresearch資料のみ。モデル・訓練・GPU・ジョブの停止／起動は行っていない。新実装・入力生成・改善訓練の成果は、追加作業書の完了記録と実artifactで追記する。

## 8. 追加調査：軽量Vision Mambaと圧縮の境界

2026-10-02追記。上の20:10 snapshotを保持し、初稿22件に7件を補足する（R01〜R29、合計29件）。Vim／VMambaはR12／R13を再利用し、同じ論文を重複した研究件数に数えない。今回の追加調査もモデル・GPU・訓練jobを操作していない。

### R23. EfficientVMamba：atrous selective scanとlocal／global融合

**一次資料・区分：** [EfficientVMamba: Atrous Selective Scan for Light Weight Visual Mamba](https://ojs.aaai.org/index.php/AAAI/article/download/32690/34845)、AAAI 2025、§Efficient 2D Scanning／EVSS、Table 6。[arXiv初稿](https://arxiv.org/abs/2403.09977)は2024年。

**事実：** ES2Dは空間位置をskip samplingで分けてscanし、元の配置へmergeする。EVSSはglobalなES2Dとlocal convolutionを別々にSEで重み付けし、加算する。Table 6のtiny対照ではES2Dのみ73.6%、conv fusion追加75.1%、inverted residual構成追加76.5%のImageNet accuracyを報告する。構造追加に伴いparametersも5Mから6Mへ変わる。

**適用：** local temporal＋global mixerを独立に持つv2の参考。SSMだけを減らす設計と、local経路で細かい変化を補う設計を分けて評価する。EfficientViM（R15）のhidden-state channel mixingとは別論文・別機構である。

**限界・ONNX：** 時間stepやGT targetsを単純に間引く手法へ読み替えない。原論文の空間skip/mergeを採るなら全時刻出力・padding・遮蔽境界を再設計する必要がある。scan kernelを含む公式backboneのexportは今回のConv1D／Reduce設計より難しい。**採否：local／globalの構成原理を採用、ES2Dの直接移植は保留。**

### R24. LightViM：publisher abstractが示すlocal／global設計

**一次資料・区分：** [A lightweight visual mamba network for image recognition under resource-limited environments](https://doi.org/10.1016/j.asoc.2024.112294)、Applied Soft Computing 167 Part A、2024年12月、112294。[publisher掲載ページ](https://www.sciencedirect.com/science/article/pii/S1568494624010688)。今回確認できた根拠はpublisher abstract／highlightsのみで、本文の実験条件・著者公式実装は未確認。

**事実：** abstractはMamba sub-blockで低周波global情報を取り、grouped convolutionのlocal情報と融合するLGF-Mambaを述べる。小さいresource環境での画像認識を目的とする。

**適用：** temporal global poolとlocal convolutionの併用に整合する独立の設計参考。ただしこの記述だけでfusionの正確な式、deploy parameters、KDの有無、訓練量を指定しない。

**限界・ONNX：** 実装未確認なので標準ONNX対応やRTX5090での速さを主張できない。低周波画像認識の結論は高速3D運動・遮蔽の保存を保証しない。**採否：概念参考のみ。公式方式の再現、数値比較、ONNX成功の根拠には使わない。**

### R25. MambaLiteSR：回帰KDと低rank、ただし精度維持の読みに注意

**一次資料・区分：** [MambaLiteSR: Image Super-Resolution with Low-Rank Mamba using Knowledge Distillation](https://arxiv.org/html/2502.14090v1)、arXiv 2025、§IV-D/E、Table II/III。確認版は2025-02-19のv1で、会議採択は今回の根拠として主張しない。

**事実：** 幅／深さ縮小、低rankなlinear、`α L1(S,T) + (1−α) L1(S,GT)`を組み合わせる。本文はstudentをONNX→TensorRTでJetson Orin Nanoへ展開したと報告。一方Table IIはDVMSR 424k／500,000 iterations／PSNR32.19に対し、MambaLiteSR 370k／2,500 iterations／28.28で、同等精度の実測表ではない。論文は15%縮小と表記するが、記載値からbaseline比を計算すると約12.7%減である。

**適用：** regressorでもGTとteacherのL1を独立に重み付けでき、低rankは標準MatMulへ落とせる。ただし低rankの訓練費・保存parameters・runtimeを別集計する。

**限界・ONNX：** 約13倍圧縮でAJ差≤.003を達成する根拠にはならない。PSNR差3.91dBと訓練量差を隠してabstractの類似精度だけを引用しない。JetsonのTensorRT成功はdynamic F600／ORT CUDAの成功証拠ではない。**採否：回帰KD／低rankの参考、精度維持の強い実証としては不採用。**

### R26. Vision Mamba PTQ：token outlierとhidden stateの量子化

**一次資料・区分：** [Post-Training Quantization for Vision Mamba with k-Scaled Quantization and Reparameterization](https://arxiv.org/html/2501.16738v2)、arXiv 2025、§III/IV、Table I/III。今回の分類は確認したpreprintに限定する。

**事実：** activationの大きなrangeとtoken/channel outlierを分け、similarity-based scaleとk-scaled量子化を組み合わせる。SSM hiddenの誤差伝播には再パラメータ化を用いる。Table IのViM-T、linear/conv W8A8対照はMinMax62.2%、組合せ75.5%、FP76.1%。SSMまで含む最終実験ではImageNet低下0.8〜1.2ポイントを報告する。

**適用：** 将来INT8を検討する際のSSM activation診断の参考。長F・最大N・全不可視・遮蔽復帰を含むcalibrationが必要という今回の要求を強める。

**限界・ONNX：** 論文はfixed image token位置のoutlierを利用する。可変F/Nのtrackingへ固定token scaleを直接使えない。量子化のscale再配置とR20/R21のlinear branch fusionは別操作。CUDA EPの実integer実行・dynamic scaleの対応・AJ劣化を別検証する必要がある。**採否：FP32精度を先に満たした後の別計画。現在のFP32 gapを治す方法としては不採用。**

### R27. Mamba-Shedder：構造感度はMamba世代によって異なる

**一次資料・区分：** [Mamba-Shedder: Post-Transformer Compression for Efficient Selective Structured State Space Models](https://aclanthology.org/2025.naacl-long.195/)、NAACL 2025、[本文PDF](https://aclanthology.org/2025.naacl-long.195.pdf)、§3、Table 2。[著者コード](https://github.com/IntelLabs/Hardware-Aware-Automated-Machine-Learning/tree/main/Mamba-Shedder)。

**事実：** calibration集合のmetricでblock／SSM／hybrid部品の除去感度を測り、影響の小さい構造から反復削除する。Table 2では64個中16 SSMのtraining-free削除で、Mamba-2.8Bの平均accuracyは59.9→49.8、Mamba2-2.7Bは60.2→59.8。SSM内部のparameter数は小さく、主な効果は計算削減と明記する。

**適用：** 「Mambaを刈れば平気」を一括で仮定しない。将来teacherを同系幅縮小する場合、state／head／projection groupを個別に感度評価し、削除後の短い再訓練と全F評価を行う。

**限界・ONNX：** 本件teacherは2 blocksしかなく、64層LLMのdepth冗長性は期待できない。Mamba-3の複素回転、BCNorm、state/head対応を維持するschemaが必要。今回のunused行削除は感度に基づく近似pruningではなく、現forwardに寄与しない部分の等価削除。**採否：同系teacher圧縮の第2候補、2-layerの無条件block削除は不採用。**

### R28. Minitron：構造化pruning後のKDと対照実験

**一次資料・区分：** [Compact Language Models via Pruning and Knowledge Distillation](https://arxiv.org/html/2407.14679v2)、§2.3、Table 10/11、Best Practices 5–9。[著者NVIDIA掲載ページ](https://research.nvidia.com/labs/lpr/publication/minitron2024/)でNeurIPS 2024を確認。

**事実：** depth／embedding／head／MLP幅を刈り、対応weight matricesを実際にtrimしてKD再訓練する。初期pruning直後と再訓練後で候補順位が変わる。Table 11のiso-compute比較はrandom初期化、pruned GT再訓練、pruned KDを分離し、15B→8B→4Bの段階圧縮も比較する。depthを大幅に減らさない場合、追加intermediate-state lossに改善がなくlogit KDのみを選ぶ例もある。

**適用：** teacher同系構造から重要度を測って幅／stateを縮めるbackup案と、scratch v2を分ける。A2 feature追加が悪化した現在も「KDならfeatureを必ず増やす」とはしない。pruning後の短い回復訓練を経て候補を選ぶ。

**限界・ONNX：** LLMのlogit KLD-only成功を3D回帰のGT削除へ読み替えない。幅縮小はgeometry／LayerNorm／readout／Mamba stateの対応を同時に維持する必要がある。共通encoder/headの128-dim契約を保つbottleneckは独自設計である。**採否：現local/global v2で不足した場合のstructured pruning＋KD候補。**

### R29. Deep Compression：parameters、bit幅、保存bytesの区別

**一次資料・区分：** [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/pdf/1510.00149)、ICLR 2016、§2–4、Figure 1。

**事実：** 重要connectionを残すpruning、weight sharingを伴うtrained quantization、Huffman符号化の三段階で保存容量を削減する。AlexNet／VGGの結果であり、保存形式・疎表現・実行方式を含む圧縮である。

**適用：** 650kのdeploy parameters上限と、ONNX file bytes、tensor dtype、nonzero数、runtimeを分けて報告する。重みを0にしただけのdense tensorや小bit化を、tensor dimensionsを縮めたparameter削減と呼ばない。

**限界・ONNX：** Huffman圧縮PTのダウンロードbytes減はONNXのMatMulサイズやCUDA latencyを直接減らさない。コードブック／sparse演算には対応runtimeが要る。**採否：集計の定義へ反映。今回のFP32標準ONNXへ疎形式・符号化runtimeは追加しない。**

### 8.1 Vim／VMamba／EfficientVMamba／LightViMの位置付け

以下はアーキテクチャの対応表であり、異なる訓練条件・model scaleのImageNet accuracyを順位付けする表ではない。

| 系統 | 一次資料が示す中心設計 | 今回参考にする部分 | 今回そのまま採らない部分 |
| :--- | :--- | :--- | :--- |
| Vim（Vision Mamba） | positional情報＋bidirectional SSM：[R12論文](https://arxiv.org/abs/2401.09417) | offlineの前後context | 画像patch/CLS設計、Nの順序依存scan |
| VMamba | SS2Dで画像2D近傍／大域を扱う：[R13論文](https://arxiv.org/abs/2401.10166) | 入力構造に合わせる局所性 | DINO画像backbone置換、trackのraster化 |
| EfficientVMamba | ES2D＋conv/SEのdual-path：[R23論文](https://ojs.aaai.org/index.php/AAAI/article/download/32690/34845) | local＋globalの予算配分 | 時間targetsの間引き、直接空間scan移植 |
| LightViM | low-frequency Mamba＋grouped local conv：[R24 publisher](https://doi.org/10.1016/j.asoc.2024.112294) | local/global融合の概念 | 実装未確認の再現式・deploy速度主張 |
| EfficientViM（別系統） | compressed hidden内channel mixing：[R15論文](https://arxiv.org/abs/2411.15241) | token projection費、正規化、local/FFN | EfficientVMambaと同一視、GPU速度の外挿 |

Mamba系という名前だけで、causal/selective recurrence、non-causal SSD、normalized pool、2D scanを互換なものと扱わない。本件はDINO後のper-track temporal refinerなので、時間Fには順序があるがtrack Nには並び順の意味を与えない。このcontractを守ることをarchitecture名より優先する。

### 8.2 今回の圧縮形式の整理

構造化／非構造化pruningの区別は[Mamba-Shedder §2.3](https://aclanthology.org/2025.naacl-long.195.pdf)、実matrix trimmingは[Minitron §2.3](https://arxiv.org/html/2407.14679v2)、bit幅・weight sharing・符号化は[Deep Compression](https://arxiv.org/pdf/1510.00149)を参照。下表のONNX判断は本リポジトリ向けの推論であり、各論文のORT検証結果ではない。

| 形式 | 何を減らすか | 今回の650k集計／標準ONNXとの関係 |
| :--- | :--- | :--- |
| unused projectionの等価削除 | 現forwardへ寄与しないtensor行 | dense shapeを実際に縮める。旧出力parityが必要。今回は先に実施する整理 |
| 構造化pruning | 活動している幅／state／head／block | 関連tensorを整合してtrimし、再訓練とAJ回復が必要。dense標準opでも形状削減を利用できる |
| 非構造化pruning | 個別weightを0へmask | dense shapeが同じなら登録parametersや標準dense計算は減らない。専用sparse支援無しの速度向上を主張しない |
| low-rank factorization | 大matrixを2個の小matrixへ近似 | `r(m+n) < mn`なら重み数減。2 MatMulのlaunch／memory費まで実測し、元denseへ戻すfusionで削減が消えないか確認 |
| quantization | 1 weight／activationあたりのbits | 原則parameter個数を減らさない。FP32受入を先に満たし、量子化後AJ／calibration／EP支援を別評価 |
| branch reparameterization | 訓練branch数をdeployで融合 | 線形和を単一kernelへ変換。訓練paramsとdeploy paramsを分け、fusion parityを示す |
| 符号化／圧縮保存 | 配布file bytes | decode後のshapeやdtypeは別。圧縮archiveサイズをONNX推論時サイズと混同しない |

現在の優先順は「query整合→unused等価削除→local/globalへ容量再配分→段階KD→必要時structured圧縮」で維持する。量子化・疎符号化でサイズだけ小さくしてFP32の全F AJ不一致を隠す方向へ変更しない。

## 9. 初稿後の実装差分メモ

2026-10-02 20:21:19 JST+0900（11:21:19 UTC）の統括エージェント提供の実装確認：新v2はtrain617,374／copy-fused deploy615,838 parameters、CPU module/fusion関連9 tests成功。strict queryの2-clip試験は90／132 tracks、anchor再投影最大誤差0.0001221px。これは20:10初稿の未実装状態からの限定的な進展として記録し、旧snapshotの診断値を書き換えない。

この実装確認は改善訓練済みcheckpointの証拠ではない。train/deploy parameter上限と2-clip整合・CPU testsだけで、全1,027本の新入力完成、CUDA実行、ONNX全contract、全F AJ、公式150本margin、最終DoDを合格にしない。後続の正式結果は `temp/workdoc_Oct02-2026_kd_accuracy_improvement.md` と対応artifactへ時刻付きで追記する。
