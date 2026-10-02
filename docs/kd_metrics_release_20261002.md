# 軽量refinerの現時点スコアと論文指標の対応

対象は段階的蒸留R2、teacher best80、student best180。window16でblock整合100＋finetune200の追加訓練完了、最良loss0.2924778565。教師を同じstrict16F validation15で測ると0.1962026500。このlossは以前のRelease学習lossとは比較しない。

## 同一15動画・全フレーム・全点の評価

既知validation15動画（ADT/DriveTrack/PStudio各5）で各clipを平均し、各subsetを等重みにした値。studentは学習時のunfused PyTorch FP32、教師のMamba内部はnative BF16、前段FP32を共有。独立testではなく公式minival150でもない。

| 指標 | 教師best80 | 軽量best180 |
| :--- | ---: | ---: |
| Median-scaled 3D-AJ ↑ | 0.14899326 | 0.14965128 |
| Median-scaled APD3D ↑ | 0.21947441 | 0.22579581 |
| Absolute metric-AJ ↑ | 0.26813288 | 0.19643878 |
| Absolute metric-APD3D ↑ | 0.39121931 | 0.30581274 |
| Occlusion accuracy ↑ | 0.81118275 | 0.81118275 |
| GT可視点3D距離のclip平均 [m] ↓ | 1.28484698 | 1.50404550 |
| 各clipのGT可視点3D距離中央値を平均 [m] ↓ | 0.70915979 | 0.98990281 |

AJ/APD/OAは0–1表記で、百分率は100倍。最後の行は全点をpoolした中央値ではなく、PointOdysseyのMTEでもない。スケール補正後の構造は近い一方、絶対metric-AJは軽量モデルが約26.7%低く、精度維持完了とはいえない。Occlusion accuracyが同値なのは全方法で同じflow可視性を使うためであり、軽量visibility headの改善証明ではない。特にPStudio絶対metric-AJは教師0.36037736→軽量0.16687923と低下している。

証跡: `result/refiner_kd_improved_20261002/monitor_full_pilot_R2_s42/accuracy_report.json`。

## PointOdysseyとの違い

[PointOdyssey原論文 §5.2](https://arxiv.org/html/2307.15055v1#S5.SS2)では2Dピクセル追跡のδ_avg、MTE、Survivalを報告する。δ_avgは256×256に正規化した座標で1/2/4/8/16px以内にある割合の平均、MTEは軌跡誤差の中央値、Survivalは50px超えの追跡失敗までの長さを動画長で割った割合。今回はPointOdysseyデータ自体とこれら2D指標を評価していないので、数値は未評価とする。

[元tracker論文 §4.1–4.2](https://arxiv.org/html/2609.34035v1#S4.SS2)の主ベンチマークはTAPVid-3D minival150。median-scaled 3D-AJと、中央値スケール補正なし・固定1/4/16/64/256cm閾値のmetric-AJを分けて報告する。こちらが今回の3D指標に対応する。ただし論文minival150と今回validation15は集合が違うため、論文の0.256に今回0.19644を並べて優劣を断定しない。

## 公開方針

[既存Release](https://github.com/yuki-inaho/vmamba3-3Dpointtracker/releases/tag/mamba3-preview-20261001-step200)へ `refiner_kd_v2_best180_20261002` prefixの9assetsを追加済み。既存19assetsはSHAを保全し、Release本文は軽量モデルのexperimental節だけ追加した。9assetsの実再download SHA一致を確認。証跡は [公開監査JSON](evidence/refiner_kd_20261002/public_release_kd_best180_audit.json)。

配布ONNXはcopy-fusion＋CPU dynamic9条件と、ORT CUDA12条件（3subsetの実monitor入力、F600/N900、点chunk1、点permutation）で数値検証済み。主要float演算はCUDA、CPU演算はint64のshape/indexのみ。GPUの数値正しさと上記タスク精度は別の証跡。

モデルは615,838パラメータのcopy-fused deploy形式、ONNXは約2.55MB、PTは約2.49MB。学習型617,374パラメータのbest180も4列demo用に添付。教師best80の推奨は保全し、軽量モデルはexperimental追加assetとする。CPU/GPUの数値一致とタスク精度は別の証跡にする。実配布とdownload監査の結果は作業書に追記する。
