---
name: vmamba3-training
description: Prepare, configure with Hydra, start, resume, or inspect vmamba3-3Dpointtracker training while preserving existing model and data settings. Use for this repository's training workflow, not unrelated JAX workloads or generic GPU tuning.
---

# vmamba3 training

Locate the repository and its external data directory from the current workspace. Inspect their README, continuation records and current processes rather than assuming saved status or PIDs remain valid. Paths to external workflow helpers below are relative to that data directory.

Preserve the user's experiment unless a change is requested: model architecture, public checkpoint identity, DINO/DA3 revisions, flow settings, image/depth resolutions, seed, augmentation, train/heldout split, optimizer, schedule, batch/bucket semantics, cache namespace and budgets. When migrating YAML to Hydra, compare fully resolved sections semantically, including CLI overrides and output directories. Record the resolved configuration and provenance for each launch; configuration management must not silently change training math.

Run preflight against the exported configuration actually passed to the trainer, not a fixed old YAML while launching different overrides. New output directories do not automatically inherit heldout membership: preserve the desired `growing_pool.json` using `data.split.validation_state_from`. Same seed alone does not preserve heldout when the ready pool grows. Yesterday's partition cannot be reconstructed from absent artifacts; identify the real preservation baseline explicitly.

On the current workspace, `hydra/launch.py` manages model/data/train/flow/loss/frozen_cache groups and exports the upstream YAML without double-normalizing loss weights. `train.sh launch.dry_run=true` records resolved settings and provenance without running GPU training. `train.sh --cfg job --resolve` only displays the composed configuration. Ordinary Hydra overrides follow the script argument, for example `train.seed=42`. Actual runs invoke the memory-safe wrapper and preflight; a checkpoint-bearing output requires compatible saved configuration. Use a separate output for a new experiment, not a reused smoke directory.

## Host-specific execution

- Source `environment.sh` and invoke `VMAMBA3_PYTHON` directly. The locked root environment currently selects CUDA13, incompatible with driver R570; the separate CUDA12.8 environment is GPU-tested. Recheck these facts on other hosts rather than replacing their dependencies automatically.
- Public best80 omits frozen DINO tensors and optimizer/RNG. A new experiment is a weights-only warm start. Require that all public tensors match and only the expected frozen DINO backbone is missing. Resume an existing experiment from its optimizer/RNG checkpoint, never substitute the public release weights and call it exact resume.
- WAFT prewarm can retain unused PyTorch allocator blocks and prevent Triton autotuning from allocating GPU memory. The external `run_training.py` releases only unused blocks before model forward; reuse that launch path when this condition applies. Do not reduce resolution or change the model just to make a smoke pass.

## Small-model distillation

Identify the actual architecture before reporting a run as lightweight: `train_best80` trains the approximately8.118M-parameter teacher refiner; R2 has617374 training-form/615838 deployment parameters. These counts exclude the frozen frontend pipeline. R2 fine-tuning uses the unfused `_train.pt` artifact and a frozen native teacher, not the fused deployment state or the ordinary teacher trainer.

The staged KD route is `scripts/train_refiner_staged.py`; its legacy protocol assumes train1027/validation15. A different ready-pool experiment requires explicitly pinned `data.partition_counts`, exact split/manifest membership and preserved heldout, not silently removing the guard. `--warm-start` permits SHA-pinned student initialization outside long mode; that resets optimizer, whereas `--resume` restores optimizer/RNG and requires unchanged configuration/source/input identity. `--stop-after-updates` is an operational checkpoint pause, not a training-budget or identity change.

Reuse trained R2 mixers when fine-tuning existing weights; fresh teacher-common copying is not itself mixer knowledge transfer. Strict reanchored FP32/window16 KD inputs cannot reuse the existing BF16/window8 frozen-cache namespace. Keep new input artifacts isolated and account them in the same global data budget. Selective teacher-better KD and bounded gradient balancing are available, but their effectiveness is an experiment, not a quality guarantee; the gradient gate covers mixer parameters, not every head. Distinguish short-window loss/metric selection from full-sequence absolute-metric evaluation, and retain the original student as a candidate so a worse fine-tuned model is not silently promoted.

## Readiness and running jobs

Use preflight, an isolated real-data smoke with finite updates, and a checkpoint resume check before claiming runnable training. Artificial GPU probes do not prove full-resolution real-data readiness. Distinguish the configured maximum batch from actual bucket batch sizes.

Raw dataset selection is not downloaded inventory: inspect completed raw files and matching depth markers. Verify official split membership, no minival contamination, fixed heldout membership, NPZ CRC/hashes and depth frame correspondence. Approximate100GB raw is distinct from the larger raw+depth+frontend-cache budget.

For zero-update prewarm, preserve the compatible prewarm output and use steps0 with step0 validation disabled. This trainer snapshots the current ready pool before checkpoint restore and still prewarms on a step0 resume. Use the external `verify_prewarm.py` to record a model digest before the final pass, then compare it with `--baseline-report`; `--require-current-pool` is appropriate after the ready pool is stable. Check step0, empty history, zero consumed clips, finite tensors, checkpoint/exported configuration agreement and fixed heldout. The ordinary update-history verifier is not a valid gate for an intentionally empty prewarm history. A historical partial prewarm remains valid when new ready clips arrive; report its coverage gap separately.

Prewarm covers one configured fixed window per train clip, not every video frame or heldout clip. Distinguish a completed whole-pool pass from simultaneous cache residency: LRU eviction or global disk pressure can prevent persistence without failing the pass. Cache limits are maxima, not required allocation sizes. Assess actual usage and expected content-key coverage before claiming all features remain cached.

Background download/depth processes should continue independently of chat turns. Confirm `/proc` command/start identity, actual files and logs. Never duplicate or restart a live job merely because an observation timed out. OOM recovery must retry the same work with bounded adaptive batches; permanent errors should be recorded, not skipped silently.

For this dataset's gzip tar shards, restarting the gzip decoder after every dropped connection repeatedly re-downloads the prefix. The external `range_stream.py` keeps the same decoder alive and resumes compressed bytes via HTTP Range, requiring matching strong ETag, total length and exact Content-Range start. Test reconnection with fault injection and a small real-server Range comparison before using another host's server; do not assume it supports safe range continuation. Process termination still loses decoder state, but completed NPZs remain reusable.

Use training authorization already given in the conversation; do not ask for it again for routine steps within that scope. Preparation alone does not authorize a long experiment, and a later instruction to do only research or planning excludes new training. Once authorized, start after the requested readiness checks, use a distinct experiment output, retain checkpoints and persistent logs, and verify actual process and update progress. Do not mark the larger preparation goal complete while required downloads or depth generation remain unfinished.

After early stopping, distinguish the stopper's reference from the saved best model. `EarlyStopping.best_step` advances only for improvements exceeding `min_delta`; it need not identify the lowest raw validation score. Select the saved best via `training_status.best_checkpoint` or `checkpoints.json`, verify its checkpoint history, and report the stopping reference separately. Preserve stopped state; do not automatically reset the stopper or launch another experiment.

## Metric-driven improvement diagnosis

Report improvement only against the same checkpoint, split, input/query contract, sequence length and metric. Record short-window loss, short-window AJ and full-sequence absolute AJ separately, including per-subset changes. A lower training/monitor loss does not demonstrate better tracking. List the checkpoints actually evaluated at full length; evaluating the original, short-AJ winner and loss winner does not establish the result for every saved checkpoint. Known monitor clips remain non-blind even when the current fine-tune has no sequence overlap.

The current GT loss preserves absolute coordinate targets while dividing errors by a detached GT scene scale. This changes scene weighting; it is not invariant to scaling the prediction. Likewise, median-scaled and absolute AJ use different threshold definitions, so their difference alone does not prove that every failure comes from scale.

Before another improvement run, inspect matched-input behavior across lengths, distance ranges and subsets. `reference_depth` computes one lower median across B/F/N; mixed microbatches and different windows can therefore change normalization for the same clip. Isolate this effect from time-context changes by retaining identical frontend tensors and queries. Any correction must preserve or explicitly version the eight-input deployment/cache contract and the shared reference used for all track chunks.

Inspect the criterion used by the actual training method: balanced/selective training currently uses teacher-better residual and XYZ KD, while the generic criterion also offers feature and temporal terms. Availability in code does not mean a run enabled them. Check actual warmup progress and effective KD gradients before recommending larger coefficients or longer training. Already trained R2 mixers and existing block alignment should be assessed before proposing random reinitialization or new pruning machinery.

Make the next experiment's candidate, metric, resource limit and stopping condition concrete. Reuse existing inputs when their identities match, preserve the original student as a candidate, and stop to report a failed gate rather than silently extending the experiment. Record timestamps with explicit UTC/JST labels; distinguish launcher time, optimizer time, evaluation time and background data work. Billing cannot be inferred from elapsed wall-clock time alone.

Improve this skill only from demonstrated workflow failures or verified behavior. Keep run-specific progress in the external work records, not in these reusable instructions.
