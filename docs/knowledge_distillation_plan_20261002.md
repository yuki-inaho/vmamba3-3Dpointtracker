# refinerを約62万パラメータへ蒸留する設計方針

2026年10月2日調査・実行中。専用ブランチは `distill/refiner-kd-onnx-20261002`、起点はmainの `6223405f44c72487dfd3d6669e490282a29b0821` です。以下の設計各章は初回計画の説明を残し、実行状況は次の節と作業書の証跡で区別します。

## 最新状況（2026-10-02 22:47 JST）

改善R2追加学習はblock100＋finetune200で終了しbest180を選択。同じ15動画の全系列評価、3シーン481Fの4列動画、copy-fused ONNX CPU/CUDA検証、既存Releaseへのexperimental軽量9assets追加と実再download監査を完了。単体テスト395件、全体ty、変更Python ruffはPASS。絶対metric-AJは教師0.26813→student0.19644で、精度維持・独立test/公式150件・長系列最終訓練の受入は未達。[最新スコア](kd_metrics_release_20261002.md)、[次回の課題](diary/2026-10-02_refiner_kd_remaining_issues.md)を参照。

以下は19:09時点の履歴。temp作業書の公開時点snapshotは [docs/workdocs](workdocs/README.md) に保存。実測R2 deployは615,838パラメータ、PT約2.49MB、ONNX約2.55MBで、初期計画の未測定記載を更新するもの。

## 初期実行状況（2026-10-02 19:09 JST、履歴）

VSSD student、専用KD損失・学習専用aux、strict checkpoint、ONNX export、全F評価を実装。194 unit tests成功後にGPU境界fixtureの追加testも成功。最終品質監査は未完です。[短window訓練と全F monitorの証跡](evidence/refiner_kd_20261002/short_training_summary.json)

- A0〜A4を同じ200-step上限で比較し、monitor GT loss最良のA1（residual/XYZ蒸留）を選択。feature/temporal/auxの追加は今回のpilotでは改善しませんでした。
- 同予算fullではA1 seed42/43/44のbest GT lossは0.11566/0.11477/0.10973。GT-only seed42は0.11385。実終了は240/270/270/300 stepsのearly stopで、上限まで常に学習したという意味ではありません。
- 全15本・全F/Nのmonitor metric-AJは教師0.26813、A1 seed42/43/44が0.22323/0.24373/0.24116、GT-onlyが0.22773。短windowだけでは教師精度に届かず、蒸留の最終優位性も未証明です。seed42を基準とする事前方針は維持します。
- 学習済みshort seed42は620,502 paramsでFP32 ONNX化とCPU一致に成功。長系列・全不可視・track chunkを含む検証です。最終採用モデルではありません。実動画/CUDAの追加検証を進めています。
- window16/32の前処理を別namespaceで再生成中。公式minival150本の取得・depth・共通入力準備も進行中で、部分集合を合格扱いしません。
- 既知データをsequence単位で監査するとtrainとmonitor/minivalに重複があります。clip名が違うだけで完全未見とは呼ばず、今回の評価は既知sequence重複ありの比較です。

最終checkpoint選択、long適応、全150本のAJ/信頼区間、最終CPU/CUDA・速度gateは未達です。Releaseの差替え、commit/pushは行っていません。

## 初回設計の要約

推奨は、公開済みbest80のrefinerを固定teacherにして、まずフォーク元の **VSSD two-pool構造、620,502パラメータ** をstudentとして学習することです。GTによる監督を残し、補正量、中間特徴、時間方向の関係を順に蒸留します。第二候補は、学習時の畳み込み分岐を推論時に統合する約62万パラメータのtemporal CNNです。どちらも先にFP32 ONNXを検証します。

これは約13倍の圧縮を目指す実験計画であり、同等精度の達成を保証するものではありません。パラメータ数、数値一致、追跡精度、実速度は別々に判定します。実行用のチェックリストと記録欄はローカルの `temp/workdoc_Oct02-2026_refiner_kd_onnx.md` にあります。

## 圧縮する範囲と基準値

refiner単体には、DINO特徴の射影、入力埋め込み、temporal mixer、正規化、depth・UV・visibilityの出力headを含めます。凍結DINO backbone、WAFT、Depth Anything 3は含めません。学習専用headは推論モデルから除去しますが、学習時の追加パラメータ数も別途開示します。

| 対象 | 現行best80 teacher | フォーク元v64相当student A |
| :--- | ---: | ---: |
| refiner単体 | 8,117,988 | 620,502 |
| temporal mixer部分 | 8,038,496 | 541,010 |
| mixer以外 | 79,492 | 79,492 |
| 凍結DINOv3 ViT-S/16 | 21,596,544 | 21,596,544 |
| DINOとrefinerの合計 | 29,714,532 | 22,217,046 |
| refinerのFP32 tensor容量の理論値 | 32,471,952 bytes | 2,482,008 bytes |

約92.36%減、約1/13.08になるのはrefiner部分です。DINOとの合計では約25.23%減であり、前処理を含むシステム全体が13倍小さくなるわけではありません。現行公開PTは32,490,639 bytesで、ファイルのメタデータ等によりtensor容量とは異なります。小型studentのPT/ONNXファイルサイズはまだ実測できません。

teacherは `weights/mamba3-preview-20261002-best80/tracker_mamba3_daoff_best80.pt`、SHA256は `c022511872c2d59a39bc1c39a3c62f44c142ee8ff9ca7dc7859881a78ae490b7` です。`weights_only=True`で再読込し、50 tracker tensorsの要素数8,117,988とstep80を確認しました。DINO前処理の2 buffersはパラメータに数えません。[公開と訓練の証跡](evidence/daoff_best80_20261002/training_summary_20261002.json)

上流の比較対象は [MasahiroOgawaのv64設定](https://github.com/MasahiroOgawa/vmamba3-3Dpointtracker/blob/4d4047145648fab5bb1c70b117585935e601319c/configs/v64.yaml) と、同じ `visionMamba3` revision `637c30b3fa904893204163d94bc9a5fe0a7fece1` のVSSD経路です。dim128、state_dim64、4 heads、2 layers、two_pool=trueとして数えた構造値であり、上流の配布済み学習モデルファイルを測った値ではありません。

「同等の軽量度」は、推論時refinerの目安を600,000〜625,000、上限650,000パラメータとする提案として具体化します。精度不足を理由に無断で100万以上へ緩和しません。まず予算内の失敗も含めて結果を示します。

## 論文調査と採用する要素

以下は一次資料を2026年10月2日に確認したものです。分類・言語モデルの論文結果を、そのまま3D追跡の精度や速度保証に置き換えません。

| 資料 | 論文が扱う内容 | この計画への適用と限界 |
| :--- | :--- | :--- |
| [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531), 2015 | teacherの出力による小型モデルへの知識転移 | 出力蒸留の基本。ただしUV・XYZは連続値なので分類logitのsoftmax KLを適用しない |
| [FitNets](https://arxiv.org/abs/1412.6550), ICLR 2015 | 中間表現をhintとして、studentからteacher表現への学習可能な写像を使う | 128次元のadapter後特徴を2段階で蒸留。射影は学習専用で推論から除去する |
| [Relational Knowledge Distillation](https://arxiv.org/abs/1904.05068), CVPR 2019 | サンプル間の距離・角度などの関係を転移 | 時間差分の蒸留を着想。今回の隣接時刻の補正差分は独自の適用であり、論文のRKDをそのまま再現するものではない |
| [RepVGG](https://arxiv.org/abs/2101.03697), CVPR 2021 | 学習時の畳み込み・identity分岐を代数的に単一kernelへ統合 | 候補Bの時間方向Conv1dに適用。非線形なMamba全体を等価なConvに変換できるという意味ではない |
| [MobileOne](https://arxiv.org/abs/2206.04040), CVPR 2023 | depthwise/pointwise構造での学習時分岐と推論時統合、実機latencyの検証 | 少パラメータでも実機速度を測る。画像分類の1msを本trackerへ外挿しない |
| [Transformers to SSMs / MOHAWK](https://arxiv.org/abs/2408.10189), 2024提出・2025改訂 | mixing matrix、block hidden、最終予測を段階的に整合する異種構造の蒸留 | 段階的hint学習の根拠。teacher/studentで異なるstateの逐要素一致やF×F行列全体の蒸留は初期計画では行わない |
| [CoTracker3](https://arxiv.org/abs/2410.11831), 2024 | 実動画にteacherのpseudo-labelを与えるpoint trackingの訓練 | trackerでもteacher supervisionを使う先例。今回の3D refiner・前処理固定・62万という条件で同等精度が得られる証明ではない |
| [Mamba-3](https://arxiv.org/abs/2603.15569), 2026年3月提出 | 離散化・複素state・MIMOを使うSSMとstate trackingを検証 | 幅とstateを小さくする候補Cの根拠。現teacherはSISOなので、MIMOの論文結果をteacherの効果と取り違えない |
| [DepGraph](https://openaccess.thecvf.com/content/CVPR2023/html/Fang_DepGraph_Towards_Any_Structural_Pruning_CVPR_2023_paper.html), CVPR 2023 | 層間の依存を考慮して構造単位を除去するpruning | 構造pruningを予備候補とする。Mamba-3のhead/state/rotaryの依存を個別に検証する必要がある |

## 現行コードから分かった設計上の制約

`src/mamba3_tracker/model/depth_refined_tracker.py` の `Mamba3V35Refiner` は、ray2、depth1、flow visibility1、5×5 depth patch25、射影DINO64の計93次元を128次元へ埋め込みます。各trackを独立した時間列 `(B*N,F,128)` に変形して処理します。現行teacherの2 mixerは128→768→128のadapter付きofficial Mamba-3です。順方向と逆方向は重みを共有します。

headは `delta_uv=2*tanh(...)`、`dlog=clamp(...,-2,2)` を出し、`uv'=uv+delta_uv`、`z'=GridSample(depth,uv')*exp(dlog)` としてintrinsicsからcamera-frame XYZを再構成します。この幾何経路を保持し、XYZを自由なMLPで直接出す方式には変えません。

現行 `TrackingLossV35` は可視GTへの深度スケール正規化L1 XYZとUV補正のL2正則化です。ログの `pos_2D` はUV正則化であり、2D追跡誤差ではありません。**visibility headへの教師信号はありません。** best80のvisibility logitは信頼度として未校正なので、visibility KL蒸留とteacher logitによるKD重み付けは採用しません。[現行設定](evidence/daoff_best80_20261002/run_config_20261002.json)

teacherはbidirectionalであり、未来フレームも使います。この計画は既存と同じoffline trackerです。causalなonline trackerへ変えることは別課題です。また `z_ref` は全B/F/Nのlower median+1e-6という1個のscalarです。track chunkごとに再計算したり、推論でFを分割して状態をリセットしたりしません。

## student Aは元のVSSD two-pool構造

第一候補は、現行コードに既にある `temporal_mixer=vssd_cross`、`two_pool=true` の2 layersです。共通encoderとheadはteacherからshape一致するtensorをコピーできます。mixerは構造が異なるため、768次元のMamba weightを強制的にreshapeして移植せず、既存VSSD初期化を使います。全refinerを学習対象とし、DINO等は凍結します。

`third_party/visionMamba3/src/visionmamba3/cross_attention.py` の標準経路は、キーを小さなstateへ集約する `collapse` です。固定state_dimではF方向の計算量は線形で、`CumSum`・`Exp`・`Softplus`・正規化・行列積によるpure PyTorch実装です。二乗経路 `OnlyForCodingCorrectnessTestTokenLevelPathOrderTSquare` とF×F attention可視化は推論に使いません。これは現行ソースからの判断で、studentのONNX export成功を確認した結果ではありません。

ONNXではeinsumを必要に応じて標準MatMulに分解します。two-poolの別query射影、第二pool weight、zero-init gateを保持し、使われていないprojection出力を削る最適化は等価性を確認した後に限定します。最初は620,502という比較基準の構造をそのまま再現します。

VSSDはteacherのselective scanそのものではなく、圧縮されたglobal stateを読む近似能力に限界があります。同一パラメータのGT-only studentを必ず対照として置き、蒸留の効果と単なる構造差を区別します。

## student Bは再パラメータ化するtemporal CNN

候補Aの精度か実機latencyが不足する場合の第二候補です。128 channels、4 blocks、gated pointwise MLPの中間幅304、dilation `[1,2,4,8]` を初期案とします。各blockの処理案は次の通りです。

1. 外側のLayerNorm後に、同じ入力へdepthwise Conv1dのkernel7、kernel3、kernel1、identityを並列適用し、加算後に1個のSiLUを置く。
2. 各trackのF方向だけで可視特徴のmasked meanを取り、128→128のprojectionでglobal contextを加える。mask分母はclamp_min(1)。全不可視のtrackにはzero contextを与える。
3. 残差加算後、もう1個のLayerNormとgated MLP `128→608(split304+304)→128` を使い、残差を加える。NやBをまたぐpoolingはしない。

局所Convの受容野は91 framesですが、global meanは全Fを参照します。global contextは順序を表現できないので、長距離の時系列関係でteacherと同じ能力を保証しません。短いwindowのみで採用を決めない理由です。可視maskとpadding maskは別に扱い、padding frameはcontextや損失へ混ぜません。

分岐はすべて線形、同じstride・groups・dilationと整合するpaddingで構成します。BNは初期案で使わず、Batch間の依存とtrain/eval差を避けます。学習後は `K7 + pad(K3) + pad(K1) + identity_center` とbiasの和で1個のdepthwise kernel7に統合します。LayerNorm、SiLU、gating、入力依存contextは統合しません。この1Dへの適用は本計画の設計案です。[RepVGGの統合原理](https://arxiv.org/abs/2101.03697)

推論時パラメータの見積は `78,468 + 4*(1,024 + 117,472 + 16,512 + 512) = 620,548` です。78,468は共通部分79,492から元2 layersのpre/post norm計1,024を除いた値で、各blockはDW Conv、gated MLP、context projection、2 normsです。学習時分岐と補助headはこの数に含めません。これは未実装の算術見積であり、実モデルの `named_parameters()` とONNX initializer数で再計測します。

## 候補Cは幅を狭くするofficial Mamba-3

Aのglobal stateで時間関係を移し切れない場合に、teacherと同じSISO再帰・複素state・双方向処理を残す追加候補です。部品のCPU計数では、`d_model=160,d_state=128,expand=2,headdim=64,ngroups=1,rope_fraction=.5` のmixerと128→160→128 adapter/RMSNormは1層244,746、共通79,492と2層の合計は **568,984** になります。幅192/state64では684,572となり上限650kを超えるので、そのまま採用しません。これはcore実体の計数とadapterの算術合算で、組み上げたstudentや学習結果の実測ではありません。

768幅のteacher weightを160幅へそのままstrict copyできないため、共通128次元部分のみ移植し、小型mixerは別初期化して出力・hintから学習します。160幅は5 headsとなるので、native Tritonのforward/backwardを別smokeで確認する必要があります。[Mamba-3のstate-space設計](https://arxiv.org/abs/2603.15569) は小型3D trackerでの精度維持を証明していません。

現行 `PortableMamba3SISO/Adapter` は幅768・inner1536・24 headsに固定されています。Cにはこれをshape-drivenに一般化する追加実装と旧teacherのbitwise回帰検証が必要です。denseな標準ONNX展開を使う限りF² workspaceは残り、A/Bの線形構造とは速度上の利点が違います。Fを固定したONNXやcustom opへ変更するfallbackはしません。Cは必須工程ではなく、採用する場合は追加作業書をwrite/reviewしてから着手します。

## 補助headと蒸留する特徴

学習専用機能を `DistillationWrapper` に隔離し、student本体がteacherや補助headを所有しない設計にします。推論 `forward` は従来の4出力のままです。hint取得は明示的な `forward_train` / feature tapsで行い、グローバルhookやstate-dict keyの恒常的な変更を避けます。

| 学習専用部品 | 入力と教師信号 | 推論時の扱い |
| :--- | :--- | :--- |
| feature projector×2 | studentの128次元hidden→teacher adapter後の128次元hidden、Linear128→128 | 完全除去 |
| 中間補正head×2 | 中間hidden→bounded dlogとdelta_uv、teacher補正を信頼度重み付きで監督 | 完全除去 |
| occlusion head | 最終hidden→GT visibilityのBCE。finiteなGT/queryだけを使用 | 完全除去。評価visibilityは従来のflow maskを保持 |

候補Aは2 layers同士を対応させ、候補Bは第2・第4blockをteacherの第1・第2layerへ対応させます。768次元の内部SSM stateを強制一致させず、adapter後の残差表現を使います。特徴の比較はFP32の非affine LayerNorm後MSEで行います。[FitNetsのhintと写像](https://arxiv.org/abs/1412.6550)

teacherは `eval()` / `requires_grad_(False)` / `no_grad()` に固定し、teacherやteacher featureにgradientが流れないテストを用意します。コピー初期化のheadだけでもゼロ補正からは離れるので、初期step0のGT loss・補正分布・clamp率を必ず記録します。

## 専用損失関数の初期案

`RefinerDistillationLoss` を新設し、各項の分母を個別に計算します。位置の基本maskは `M=GT_visible*query_valid*padding_valid`、時間差分は隣接2 frameの両方が有効なmaskです。全mask=0の場合は有限な0とzero gradientを返します。モデル出力のNaN/Infは無視せずそのstepを失敗として停止します。

GT損失は現行V35と同じ重み0.9090909の位置L1と0.0909091のUV L2を基準にします。追加項はSmoothL1/Huber(delta=0.1、正規化後の単位)を初期案とし、各次元を平均します。以降の係数は実測済みの最適値ではなく、ablation用の開始値です。

```text
L = L_V35_GT
  + ramp * (0.50 L_residual + 0.25 L_xyzKD + 0.05 L_feature
            + 0.05 L_temporal + 0.10 L_aux)
L_residual = weighted_huber((dlog_S-dlog_T)/2)
           + weighted_huber((duv_S-duv_T)/2 pixels)
L_xyzKD    = weighted_huber((XYZ_S-XYZ_T)/s_GT)
L_feature  = masked_MSE(LN(P(hidden_S)), stopgrad(LN(hidden_T)))
L_temporal = weighted_huber(diff((XYZ_S-XYZ_raw)/s_GT)
                           - diff((XYZ_T-XYZ_raw)/s_GT))
L_aux      = mean(intermediate residual KD losses) + 0.1 BCE(occlusion,GT)
```

`s_GT` は現行と同じclipの有効query anchor深度尺度で、clamp_min(1e-3)を用います。`XYZ_raw` は補正前の同じflow/depthから再構成したcamera-frame座標です。時間損失はteacherの補正差分を合わせるもので、移動する対象や移動カメラの軌跡を「静止」「低加速度」に押しつけるsmoothness損失ではありません。3D scale正規化だけではmetric-AJを保証しないため、必要時は固定1m尺度のGT Huber項0.1を別ablationとして試します。

`weighted_huber` は各点の次元平均に `M*w` を掛け、分母は **M.sum().clamp_min(1)** とします。wの和で割ると全点が低信頼のとき弱める効果が相殺されるためです。時間項のwは隣接2点のmin。occlusion BCEはquery/padding有効な可視・不可視両方のGTを使い、位置項の可視maskで不可視を消しません。欠損GTをmaskし、inactiveなGTはloss計算前に有限値へ置換しますが、student/teacherの非有限出力は停止対象です。

位置蒸留の信頼重みは、GTがあるtrain点だけでteacherのGT誤差から固定計算する `w=clip(exp(-norm(XYZ_T-XYZ_GT)/s_GT/0.1),0.05,1)` を開始案とします。teacher logitやstudentの自己申告uncertaintyは使いません。固定teacherの不正確な補正を弱め、GT lossはwで弱めない設計です。温度やfloorの変更はtrain/monitorだけで判断し、testで調整しません。GT無しデータの追加は今回の主計画に含めません。

最初はGT-only、次にresidual/XYZ KD、その後feature、temporal、auxを追加します。各項の値、gradient norm、clamp率を記録し、係数の支配やgradient衝突を確認します。分類用temperature KLはこの回帰設計には不要です。特に未学習のteacher visibility headからのKLは避けます。

## データ分割とteacher cache

既存rawは `/workspace/vmamba3_data` の1,042 clips、train1,027・monitor15です。splitは `/workspace/vmamba3_data/result/v64_daoff_unused100gb_train_20261002/growing_pool.json` を固定します。DA無しはphotometric augmentation無しという意味で、DA3 metric depthは使用します。DINOv3はViT-S/16、revision `114c1379950215c8b35dfcd4e90a5c251dde0d32` を維持し、ユーザーが以前提示したViT-Bへ自動変更しません。[データと訓練の継続記録](session_memory_20261002.md)

monitor15本はteacher checkpoint選択にも使われました。既存の3比較動画はその一部です。どちらも完全な独立testとは呼びません。公式minival150は今回のtrainから除外されていますが、過去の開発で参照された可能性、同一撮影sequenceの別clip混入、全rawのローカル可用性を先に監査します。完全未見の主張が必要ならteacherの全train/monitorとsequence単位で重複しない追加holdoutが必要であり、取得が必要な場合は容量と出所を報告してから進めます。

teacher labelsは**実際にstudentへ渡す同じwindow・同じtrack・同じ前処理入力**から生成します。全Fのteacher出力を切り出して8-frame studentの正解とすることは、未来情報の違う別実験になるため行いません。window8の学習ではteacherもwindow8、long-window段階では同じlong-windowです。z_refの範囲も一致させます。

最初はnative BF16 teacherの出力をFP32にcastして使い、teacher実行環境と精度を記録します。portable FP32へ変更する場合は別cache namespaceとし、実動画差を確認します。nativeとportableの不完全な一致は [ONNX変換記録](onnx_export_design_20261002.md) に既知の制約があります。

`z_ref` はwindowだけでなくbatch内の全clipのdepthに依存します。B=1で生成したラベルを、異なるB=32の中央値を使うstudentへ流用することは不一致です。**初期のsmoke/pilot/fullはonline teacherを既定**とし、凍結前処理cacheを共用した同一microbatchからz_refを一度だけ算出してteacher/studentの両方へ渡します。gradient accumulationの各microbatchでもこの規則を維持します。batch1と32の学習を同一入力条件と扱わず、A0〜A4のmicrobatch条件を揃えます。

label cacheは任意の高速化です。keyにはteacher SHA、DINO/flow/depthのrevisionと設定・入力hash、clip/sequence/split、window開始と長さ、track IDs/query anchors、座標変換、**そのmicrobatchのfull B/F/N z_refの正確なFP32値**、mask、precision、schema versionを含めます。全batchの構成・順序もmanifestに記録します。batch構成が変わりz_refが変わればmissとし、再計算または明示的なonline modeを使います。cacheにはdlog、delta_uv、XYZ、2段のhiddenを保存し、GTは元のtrain partitionから読みます。teacher visibility logitは学習信号にしません。

cacheを採用する場合は `/workspace/vmamba3_data/tapvid3d_distillation_cache/` を新規namespaceとし、既存の凍結前処理cacheを上書きしません。10 windowsのpilotで実容量・生成時間とmicrobatch変更によるmiss率を計測し、`result/refiner_kd_20261002/cache_budget.json` に総必要量を書きます。追加cache上限20GB、空き容量10GB以上を保つ開始案で、効果が低ければonlineを継続します。hiddenの保存を間引く場合は対応するfeature loss maskも明記します。無断で旧cacheを削除したり総workspace上限を拡張したりしません。

## 訓練とablationの順序

1. parameter集計、freeze、損失mask、batch依存cacheキー、旧checkpoint読込、ONNX構造のunit testを先に作る。teacher共通headとencoderはshape一致で初期化し、mixerと学習専用headは別初期化する。A0〜A4で初期studentのtensor hashを同じにする。
2. 少量の固定入力で20 stepsのsmokeと32 stepsのoverfitを実施する。GT-onlyとKDの双方でfinite・teacher unchanged・studentの更新を確認する。
3. 同じtrain1,027/monitor15、window8、seed42を使い、A0=GT-only、A1=output KD、A2=+feature、A3=+temporal、A4=+auxを逐次比較する。まず短い200-step pilotで効果を調べる。最終KDモデルにも同予算のGT-only full/long対照を用意し、200対800-stepの学習量差をKD効果と呼ばない。
4. monitorで選んだ候補のみfull runへ進める。初期上限は既存の800 steps/20,000 processed clips、warmup100、monitor10 steps、early stop patience5・min_delta0.001を継承する。これは20,000 optimizer stepsという意味ではない。
5. full runの候補はseed42/43/44で再現性を確認する。候補BはAの評価後に同じ予算で対照とKDを比較する。実行時間はpilotで見積もり、全組合せを無条件で走らせない。
6. window16/32、必要時64のlong-context適応を別phaseとして行う。入力前処理cacheとteacher labelsを対応するwindowで作り直し、短いcacheを流用しない。long phaseの初期上限200 steps、lrをshort phaseの0.3倍として別configに固定する。

optimizer開始案はAdamW lr3e-4、weight_decay0.01、grad_clip1.0です。teacherで使ったAMUSEとの差を隠さず、GT-onlyとKDを同じoptimizer条件にします。BF16はstudent内部の候補で、geometry/loss/reductionはFP32です。始めはbatch1、4、8、16、32を測り、同じwindowの最大安全batchを選びます。32GB GPUでpeak使用量は概ね28〜30GBまで、少なくとも2GBの余裕を残す方針で、パラメータの少ないモデルを無理にVRAMで満たしません。gradient accumulationはeffective batch32の開始案です。

checkpointは `last`・`best_monitor`・`training_status.json` を保存し、optimizer、schedule、RNG、split/cache hashをresumeでstrict確認します。通常のcheckpoint選択はGT-only monitor lossで行い、GT/KDの合計lossを異なるablation間で直接比較しません。long-context候補の比較はmonitor15本の全F AJで行い、testは使いません。OOMでwindowを勝手に短くせずbatchを下げ、比較runの条件も揃え直します。early stopも正常な終端条件として扱い、bestを選びます。

## ONNXと圧縮ファイルの形式

最初の配布形式は `student_refiner_fp32.onnx` と `student_refiner_deploy.pt`、`deployment_manifest.json` です。公開PTは `weights_only=True` 対応のtensor/config主体、schema versionとstudent architecture IDを必須にします。学習状態はローカルの `student_training.pt` に分け、teacher、optimizer、RNG、認証情報、絶対host pathを公開PTへ入れません。学習用wrapperのweightsを推論用へ黙って `strict=False` でロードしません。

候補Bは分岐統合と補助head除去後にパラメータ数を測り、その後exportします。候補Aは補助head除去後にexportします。現行 `deployment/checkpoint.py:load_portable()` はofficial Mamba専用なのでstudent用loader/factoryを追加する必要があり、現在のexportスクリプトをそのまま実行してもstudentに対応しません。既存teacherのSHA・schema・bitwise loader検証は維持します。

ONNXの契約は既存の8入力 `ray,z_raw,visibility,uv,depth_map,dino_features,intrinsics,z_ref`、4出力 `xyz,uv_refined,vis_logits,delta_uv`、FP32です。B/F/Nとdepth grid H/Wはdynamic、DINO特徴は384×28×28で固定です。opset18、標準domain、単一ファイル、custom CUDA op無しを開始仕様とします。GridSampleのbilinear/border/align_corners=falseを保持します。[ONNX GridSample仕様](https://onnx.ai/onnx/operators/onnx__GridSample.html#gridsample-16)

検証は「学習構造PyTorch→deploy PyTorch→ONNX CPU→ONNX CUDA」の順です。再パラメータ化の前後はFP32でrtol1e-5/atol1e-6を開始gateとし、ORTは従来と同じrtol2e-4/atol5e-5を4出力へ適用します。精度gateとこれらの数値gateを混同しません。B=1/2、F=1/8/31/128/257/300/600、N=1/7/32、異なるdepth H/W、全不可視を含めます。F=600,N=900相当はtrack chunkで実測します。

現行 `runtime.run_chunked` のF² workspace estimateはteacher用です。studentへ流用すると不要な拒否が起きるため、architecture IDに応じた明示的なmemory estimatorを用意します。推論で分割するのはNだけ、z_refと全Fを保持します。track permutation・chunk1/7/32/allの一致をテストし、Nをまたぐcontextが混入していないことも検査します。

CPUとCUDAは既存の別環境を維持します。CUDA provider名だけで合格にせず、profileでMatMul/Conv/GridSample等の主要float演算のCUDA割当を確認します。整数shape処理のCPU割当は別集計、float処理のCPU fallbackは失敗として記録します。[現行GPU検証の方式](onnx_gpu_video_20261002.md)

FP16やINT8はパラメータ数を減らす手段ではなく、主に保存byte数と演算精度の選択です。FP32で追跡精度とdeploy gateを通した後に別artifactとして検討します。INT8はtrain-only calibrationでConv/Linear等の対象演算を限定し、depth/geometry/Exp/GridSampleはFP32を基本とします。QDQを作っただけではCUDAExecutionProviderで加速するとは言えません。公式のGPU量子化説明はTensorRT EP経路であり、CUDA EPの検証とは別です。[ONNX Runtime量子化](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#quantization-on-gpu)

## 精度と速度の受入条件案

公式TAPVid-3D evaluatorのmetric-AJとmedian scalingによる3D-AJ、subset別AJ、位置誤差、visibility関連指標を、teacher/studentに同一入力・同一query・同一flow visibilityで計測します。最終checkpoint選択はmonitor15だけで行い、minivalは選択後のreport用に固定します。主比較はnative teacher対deploy FP32 student、student内のtorch/ORT差は別表です。

「同等精度」の開始案は、各AJが0〜1尺度でteacherとの差について、3 subset等重みmacroで低下0.003以内、各subsetで0.005以内です。scene/sequence単位のpaired bootstrap 10,000回、seed42による片側95%上限も同じmargin以内なら非劣性gate合格とします。境界を満たさない場合は「精度維持未達」と報告し、marginを後から広げません。150本のsubset母数やgroup依存によってCIが広くなる可能性は残ります。

runtime benchmarkはCPU threads4とRTX5090のCUDA EPで、同じB/F/N・track chunk・warmup20回・計測100回を使います。model-onlyのp50/p95、H2D/D2H込み時間、RSS/VRAM peak、ONNX byte数、train/deploy paramsを別々に報告します。目標は同条件の現行FP32 ONNXに対しmodel-only p50が0.7倍以下ですが、これは未測定の目標です。前処理込みend-to-end時間を13倍高速と表現しません。

採用は、サイズ上限、ONNX CPU/CUDA一致、実CUDA実行、精度gateをすべて満たしたcandidateから行います。速度目標が未達なら精度合格と区別して報告します。新Releaseの更新、main統合、commit/pushは今回の計画作成に含めません。

## 実装予定ファイルと成果物

以下は新規作成する予定のパスで、現在利用可能なCLIとは区別します。

| 予定パス | 責務 |
| :--- | :--- |
| `src/mamba3_tracker/model/student_refiner.py` | A/Bのstudent、共通geometryとfeature taps |
| `src/mamba3_tracker/model/rep_temporal.py` | Bのlinear temporal branchesとstrict deploy fusion |
| `src/mamba3_tracker/train/distillation.py` | frozen teacher wrapper、損失とauxiliary heads |
| `src/mamba3_tracker/data/distillation_cache.py` | window固有key、atomic write、容量・hash監査 |
| `src/mamba3_tracker/deployment/student_checkpoint.py` | schema、architecture-aware loading、teacher/publicの分離 |
| `scripts/prepare_refiner_kd.py` | split/group監査、入力・teacher pilot/cache生成 |
| `scripts/train_refiner_kd.py` | smoke/pilot/full/long、resumeとstatus |
| `scripts/export_student_refiner_onnx.py` | aux除去、fusion、FP32 exportと一致検証 |
| `scripts/evaluate_student_refiner.py` | paired evaluator、bootstrapとlatency report |
| `configs/distill_refiner_vssd.yaml` / `configs/distill_refiner_rep_tcn.yaml` | サイズ・損失・split・実験条件を明示 |
| `tests/unit/test_refiner_distillation.py` / `test_rep_temporal.py` / `test_distillation_cache.py` / `test_student_deployment.py` | named RED→GREEN cases |
| `result/refiner_kd_20261002/` | ローカルweights、cache manifest、run/checkpoint/metrics/profile |
| `docs/evidence/refiner_kd_20261002/` | 小型の集計・再現証跡のみ。現在は未作成 |

優先順はAの未学習ONNX smoke、同サイズbaselineと出力KD、feature/aux、実機gateとlong-contextです。その後、精度面が課題ならC、latency面が課題ならBを比較候補とします。量子化と構造pruningはさらに後で、初期実装の必須工程にしません。
