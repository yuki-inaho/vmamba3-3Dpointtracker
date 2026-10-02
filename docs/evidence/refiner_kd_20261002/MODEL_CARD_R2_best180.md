# Experimental lightweight refiner — KD v2 best180 (2026-10-02)

This is an additional experimental asset, not a replacement for the recommended teacher best80.
Copy-fused deployment: **615,838 parameters**; train form: 617,374.
FP32 deployment PT: 2,492,693 bytes; ONNX: 2,553,222 bytes.
Selected best finetune step180 after block-alignment100 + finetune200, seed42, DA off.
Early stopping enabled (validation every10, patience5, min_delta0.001); the200-step ceiling was reached.

## Artifacts

- `refiner_kd_v2_best180_20261002.onnx`: standard-op opset18, single-file FP32 refiner, fused temporal convolution; portable preferred inference format.
- `refiner_kd_v2_best180_20261002.pt`: strict deployment checkpoint, no optimizer, teacher, gated DINO backbone or training-only auxiliary heads.
- `refiner_kd_v2_best180_20261002_train.pt`: original public best180 checkpoint for the four-column demo, NOT an optimizer/resume checkpoint.
- `refiner_kd_v2_best180_20261002_source.zip`: explicit Python/source snapshot needed to load this new architecture; the historical Release tag does not contain this KD implementation. No pretrained frontend weights, raw data, local tokens or caches are bundled.
- Companion export/GPU reports, metrics JSON and SHA256SUMS.

## Measured results and limitations

Evaluation uses **15 known validation clips** (5 each ADT/DriveTrack/PStudio), all frames/queries, shared FP32 frozen frontend inputs and the same flow visibility. This is NOT independent-test or official150-clip minival evaluation.
Numbers are proportions in [0,1]; multiply by100 for percentage presentation.

| Metric | Teacher best80 | Lightweight best180 |
| :--- | ---: | ---: |
| Median-scaled 3D-AJ | 0.14899326 | 0.14965128 |
| Median-scaled APD3D | 0.21947441 | 0.22579581 |
| Absolute metric-AJ | 0.26813288 | 0.19643878 |
| Absolute metric-APD3D | 0.39121931 | 0.30581274 |
| Occlusion accuracy | 0.81118275 | 0.81118275 |

Scale-normalized tracking structure is close, **but absolute metric accuracy is worse**; accuracy-preserving compression is NOT established. OA is identical because the flow visibility is shared, not because the student visibility head improved. Teacher mixer uses native BF16; student uses FP32. Results above are the UNFUSED PyTorch student; no full-minival deployed-model task accuracy is claimed.
Student loss0.2924778565 vs teacher0.1962026500 uses the same strict16-frame monitor loss; it is not comparable to earlier Release training losses.

## ONNX numerical checks

Copy-fusion parity and dynamic CPU ONNX cases passed, including temporal lengths1/8/31/128/257/300/600 and track chunking up to900 tracks. CUDAExecutionProvider checks passed on 12 cases including3 real monitor clips, permutation and chunk1. Major floating computation was verified on CUDA; CPU shape/index nodes contain only integer tensors. These prove numerical/graph correctness, NOT accuracy parity with the teacher or end-to-end RGB ONNX.

## Inference contract

Inputs in order: ray, z_raw, visibility, uv, depth_map, dino_features, intrinsics, z_ref.
Outputs: xyz, uv_refined, vis_logits, delta_uv. Keep the full temporal axis; chunk only tracks.
Compute z_ref as the full B/F/N lower median z_raw plus1e-6 BEFORE chunking; do not recompute per chunk.
DINOv3 ViT-S/16 FP32 features are [B,F,384,28,28]; image coordinates use896x896. WAFT, DA3 metric-large and authorized DINOv3 weights remain EXTERNAL frozen prerequisites. This is not an RGB-to-tracks ONNX graph.

For standalone ONNX use onnxruntime or onnxruntime-gpu with these named FP32 inputs. The included `student_runtime.py` preserves track independence and full temporal context. GPU execution uses CUDAExecutionProvider (use_tf32=0); do not allow silent provider fallback.
For PyTorch, use the included source snapshot and `load_student(path, expected_sha256)`; do not attempt to load these weights with the old teacher architecture.

## Benchmark scope

PointOdyssey itself and its2D delta_avg/MTE/Survival have NOT been evaluated. Our current APD3D/AJ are TAPVid-3D measures, not PointOdyssey pixel metrics. The original tracker paper uses official TAPVid-3D minival150, so its published0.256 metric-AJ is not directly comparable to this known-validation15 experiment.
Primary sources: https://arxiv.org/html/2307.15055v1#S5.SS2 and https://arxiv.org/html/2609.34035v1#S4.SS2.
