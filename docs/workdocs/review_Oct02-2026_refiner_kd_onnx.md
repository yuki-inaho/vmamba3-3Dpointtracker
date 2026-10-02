# refiner知識蒸留作業書のレビュー

## 再レビュー：2026-10-02 18:40 JST

**Verdict:** PASS_WITH_NOTES

**Mode:** review-and-fix（同一Codexによる自己レビュー）

**Findings / Applied Changes**

- Major：第2/4章の「新規ファイルはまだ存在しない」は後続実行記録と矛盾。実装済み部分と未実装modeを区別する記述へ修正した。
- Major：benchmark例に必須の `--fixture-dir` が無い。既存fixture保存先を追加し、同一ONNX hashの検証を前提とした。
- Major：selected_configのmetadata追加とstrict schema、long入力window8制限、全F評価未実装の依存が不明瞭。第4章に対応手順と実行前ゲートを明記した。実装そのものは今回変更していない。
- Minor：初回レビューの57手順は履歴であり、取得/depth追加後は59手順。初回記録は残し、最新の件数と区別した。

**Residual Findings**

- long入力・選択設定schema・全F評価は後続実装の未完事項。完成済みCLIとして実行しない。
- データに既知sequence重複があり、完全未見評価という主張はできない。D-7の精度合格も未判定。

**Coverage Notes**

ゴール分析、SG/TR対応、DoD、原子性、TDD、エラー処理、証跡の各項目はadequate。計画の実行可能性の判定であって、訓練・ONNX・AJの最終合格ではない。

**Open Questions / Recommended Patch Scope**

文書作成を止める質問は無し。後続実装時に上記ゲートを満たし、実測結果でチェックリストを更新する。今回の追加作業は文書のみで、commit/push無し。

---

日付：2026年10月2日。対象：`temp/workdoc_Oct02-2026_refiner_kd_onnx.md`。
方式：`review-written-workdoc` のrubricによるreview-and-fix。同一Codexの自己レビューであり、独立した別エージェントの監査ではない。
今回の判定対象は計画の実行可能性。訓練・ONNX・AJの実行合格を判定したものではない。

**Initial Verdict:** REVISE

**Final Verdict:** PASS_WITH_NOTES

**Mode:** review-and-fix

## Findings

- Major / 初稿の手順4・9・20：window cacheのz_refが全batchに依存する点の実行方針が不足していた。B=1ラベルをB=32へ流用するとteacher/studentの入力が違う。現行84行と手順9に、online teacher既定・同一microbatchのz_ref共有・cache invalidationを明記した。
- Major / 初稿の手順15・23・24・31：CLI4個、複数ablation、seeds/long、品質コマンドがまとめられていた。独立の4欄付き手順へ分割し、計57手順とした。
- Major / 初稿の候補選択：KD込みtotalをablation間で比較すると追加項の大小を性能差と誤認できる。GT-only monitorで選択し、long候補はmonitor全F AJで固定。testは選択に使わない。
- Major / 初稿の手順12–13：BがA pilot後の条件付きなのに、順次読みではpilot前の実装を要求し得た。初回保留と23.5からの明示的な戻り順序、Bのfactory/config/再検証を記述した。
- Major / 初稿の損失：低信頼wの分母とocclusionのmaskが十分具体的ではなかった。有効点数で正規化して信頼度の効果を相殺しないようにし、不可視GTもocclusion BCEへ含めるnamed testsを追加した。
- Minor / 初稿の手順10：config依存testのGREENがconfig作成より先になっていた。そのtestだけ手順11後とし、関数内importで段階的なRED→GREENを可能にした。
- Minor / 初稿のMarkdown：手順見出し直後の空行が無かった。全見出しをCommonMarkに合わせ、4欄を機械検査した。

## Applied Changes

- 作業書にCLI実装、ablation個別run、full GT-only対照、seeds、long対照、checkpoint選択、lint/format/ty/diffを追加。新CLIは未実装と明記し、予定仕様と現在使える確認コマンドを分離した。
- `selected_config.yaml` / `selected_checkpoint.json` の選択条件、strict resume、long時だけの明示的window変更を記載した。
- 現CPU ORTを維持するisolated GPU環境のuv指定を検証した。pythonの実pathはGPU `.venv`、ORT1.30、CUDA EPがavailable。これはprovider availabilityの確認で、新studentのGPU動作証明ではない。
- 調査設計文書にも同じcache・損失・公平比較方針を反映。2026年Mamba-3論文と小幅SISOの追加候補Cを追記し、Cの追加実装は別write/review計画が必要と区別した。
- VSSDモデルをオフラインCPU構築して620,502 trainable/541,010 mixer/21,596,544 frozen DINOを再確認。teacher safe PT readは8,117,988 elements、B620,548は算術見積、C568,984はcore計数とadapterの合算。小型studentの学習結果ではない。

## Residual Findings

- Minor / サイズ・AJ margins：600k〜625k目安、650k上限、macro .003/subset .005は事前固定する提案値である。ユーザーによる明示確定値や実績として書かない。
- Minor / データ独立性：minival全150本の可用性、過去の参照、teacherのtrain/monitorとのsequence重複は後続の監査対象。完了済みと誤記しないため、手順2/18とD-2/D-7を未チェックにした。
- Minor / 任意候補：B/CやINT8の精度・ONNX・GPU実行は未検証。Aを優先し、Cは追加作業書で扱う。未検証の部品計数をdeploy成功と呼ばない。

## Coverage Notes

- ゴール要求分析：adequate。現在の計画作成と将来の訓練完了を分離。
- サブゴールと作業要素の対応：adequate。SG-1〜SG-6、TR-1〜TR-7、P/DのDoDへ対応。
- 完了の定義：adequate。パラメータ数、数値一致、AJ/CI、実CUDA、速度を別gateにした。
- チェックリスト原子性：adequate。57手順、各操作/確認/テスト/エラー4欄。条件付きBの戻り順を明示。
- TDD/検証可能性：adequate。named RED→GREEN、既存teacher回帰、strict failure cases、未学習exportと精度評価の分離。
- エラー時対処：adequate。auth/NaN/OOM/cache miss/overlap/CUDA fallbackに対して明示的な停止・再調査を定義。
- トレーサビリティ：adequate。固定SHA、run/split/cache/protocol hash、開始・終了記録、小型evidence先を記載。

## Open Questions

- 後続開始時にサイズ・精度margin・実験予算の提案値を固定する。今回の文書作成を止める未回答事項ではない。
- 完全未見評価の主張には、teacherを含むgroup overlap監査が必要。欠落時はscopeを狭めた明示reportか、追加holdout取得の判断を求める。
- Aのpilotで精度/latencyが不足した場合にB/Cのどちらを追加するかは、実測根拠で決定する。

## Recommended Patch Scope

現在のMajor/Blocker残留は無し。後続のデータ監査、予定CLI実装、実験結果に応じて作業書を更新し、計画・実績を混在させない。commit/push、Release差替えは今回の対象外。

## 検証範囲

2026-10-02 17:38:42 JST+0900の追加review-and-fix：チェック済み手順1と矛盾する「全て未実行」の記述を修正。window32の学習率をshort基準9e-5と明示し、段階ごとの累積減衰との曖昧さを解消した。最終判定はPASS_WITH_NOTESを維持。以下は初回計画作成時の検証記録であり、その後の進捗は作業書第7章を参照する。

全作業書を再読し、57手順の4欄、空行、末尾空白、将来の操作が未チェックであることを機械検査した。Markdownの保存とリンクを確認するが、アプリの表示previewや新モデル訓練・exportは実施していない。
