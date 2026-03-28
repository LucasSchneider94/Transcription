# Piano Transcription Rewrite Plan (Single Handoff Document)

## 1. Goal

Rewrite model, training, and inference so onset, frame, and duration learning is stable and meaningful.
This document is intended as a complete handoff for a fresh agent to execute the rewrite.

Primary objective:
- Reliable onset learning (non-zero precision/recall/F1 on validation)
- Frame predictions that are not broad noisy activation bands
- Duration predictions that correlate with note lengths

Secondary objective:
- Keep architecture factorized (spectral reasoning first, temporal reasoning second)
- Keep run artifacts and eval workflow simple and reproducible


## 2. Architecture Review Findings (Current Code)

### Critical issues

1. Frequency stage collapses pitch structure too aggressively
- Evidence: src_new/model.py:188
- Current frequency encoder mean-pools all frequency tokens into one vector per frame.
- This destroys explicit pitch-local information before heads predict 88 keys.
- Likely effect: weak chord/note separability and blurry frame outputs.

2. Extreme class imbalance is not handled despite config claiming it is
- Evidence: src_new/config.py:44, src_new/train.py:157, src_new/train.py:158
- Config enables focal loss, but loss implementation always uses plain BCE for onset/frame.
- Likely effect: model converges to mostly-negative onset outputs.

3. Computed class weights are never used
- Evidence: src_new/train.py:673, src_new/train.py:674
- Initial onset/frame pos_weight values are printed only and not injected into the loss.
- Likely effect: imbalance remains unaddressed.

4. Onset bias initialization can further delay onset emergence when imbalance handling is missing
- Evidence: src_new/model.py:296
- Bias set to -4.6 (about 0.01 prior) is reasonable only if positive examples are emphasized elsewhere.
- Likely effect: near-zero onset predictions for many epochs.

### High priority issues

5. Visualization path uses global config, not run config
- Evidence: src_new/utils.py:107
- In training, visualization calls inference with CONFIG, not saved run config.
- Likely effect: config mismatch between training and visualization, confusing diagnostics.

6. Expensive full visualization every epoch slows sanity runs and hides core failures
- Evidence: src_new/train.py:862
- Likely effect: training loop spends significant time in plotting/inference and is easy to interrupt.

7. Loss weighting likely overemphasizes duration relative to onset recovery
- Evidence: src_new/config.py:32, src_new/config.py:33
- Onset is the sparse key signal but duration has 5x weight by default.


## 3. End-State Design (What To Build)

Use a factorized architecture that preserves pitch-specific information through spectral stage output.

Pipeline:
1. Audio preprocessing -> log-mel or log-STFT magnitude (choose one and keep consistent)
2. Spectral encoder per frame (frequency attention)
3. Key query projection (88 key tokens per frame)
4. Temporal encoder over time for each key token sequence
5. Heads for onset, frame, duration

Important change from current model:
- Do not reduce a full frame spectrum to one global pooled vector before key prediction.
- Produce key-conditioned embeddings explicitly before temporal modeling.


## 4. Data and Representation Specification

### 4.1 Input representation

Default rewrite target:
- Keep log-mel representation first for compatibility with existing processed dataset.
- Shape per snippet: (freq_bins, time_frames) = (352, 300) with current config.

Optional later variant:
- Add raw STFT magnitude mode as a separate preprocessing option.

### 4.2 Normalization

Required:
- Normalize spectrograms consistently.
- Recommended: per-file or global mean/std normalization after log scaling.
- Store normalization stats in run folder and use same stats in inference.

### 4.3 Label semantics

- onset: binary at frame and key
- frame: binary sustain activity at frame and key
- duration:
	- regression mode: log duration in seconds
	- classification mode: duration bins


## 5. Model Blueprint (Detailed)

## 5.1 Module overview

Build a new model class and retire the current main implementation.

Suggested class names:
- SpectralEncoder
- KeyProjection
- TemporalEncoder
- TranscriptionHeads
- PianoTranscriptionModelV2

### 5.2 Spectral encoder per frame

Input: (B, 1, F, T)

Steps:
1. Rearrange to frame batches: (B*T, F, C0)
2. C0 options:
	 - minimal: C0 = 1 (amplitude only)
	 - better: small conv stem over frequency to C0 = 16 or 32
3. Add frequency positional embedding
4. Apply Nf transformer layers over frequency tokens

Output: spectral tokens (B*T, F, Ds)

Recommended starting values:
- Ds = 128
- Nf = 2
- heads_f = 4 or 8

### 5.3 Key projection (critical)

Replace mean pooling with key-conditioned projection.

Mechanism:
- Learn 88 key query embeddings Qkey shape (88, Dk)
- Cross-attention: queries are keys, context is spectral tokens
- Output per frame: (88, Dk)

Batch shape after projection:
- (B, T, 88, Dk)

Recommended:
- Dk = 256

### 5.4 Temporal encoder

Temporal modeling should run on each key sequence.

Reshape:
- (B, T, 88, Dk) -> (B*88, T, Dk)

Apply Nt transformer layers over time.

Output:
- (B, T, 88, Dk)

Recommended:
- Nt = 4
- heads_t = 8

Optional extension:
- Add lightweight cross-key mixer after temporal blocks (small MLP or attention across 88 keys).

### 5.5 Heads

Per key and frame heads from temporal output:
- onset_head: linear Dk -> 1 (per key)
- frame_head: linear Dk -> 1 (per key)
- duration_head:
	- regression: linear Dk -> 1
	- bins: linear Dk -> num_duration_bins

Final tensor shapes:
- onset logits: (B, T, 88)
- frame logits: (B, T, 88)
- duration:
	- regression: (B, T, 88)
	- bins: (B, T, 88, bins)


## 6. Loss, Sampling, and Optimization

### 6.1 Loss functions

Onset and frame:
- Implement configurable choice:
	- BCEWithLogitsLoss with pos_weight, or
	- Focal loss (actually wired)

Duration:
- Regression mode: smooth L1 (Huber) on log duration at onset positions only
- Bins mode: cross entropy at onset positions only

Total loss:
- Start with onset-heavy weighting
- Suggested starting weights:
	- onset_weight = 3.0
	- frame_weight = 1.0
	- duration_weight = 1.0

### 6.2 Imbalance handling

Required:
- Use computed pos_weight values in loss, or turn on focal loss for onset/frame.
- Do not leave these as dead config fields.

### 6.3 Data sampling

Add onset-aware snippet sampling:
- Ensure each batch contains a minimum fraction of snippets with onsets.
- Keep some random snippets for negative/background context.

### 6.4 Training stability

Required defaults:
- Gradient clipping (for example 1.0)
- Optional AMP where supported
- Deterministic seed option for debugging


## 7. Training Loop Rewrite Requirements

### 7.1 Phased sanity protocol

Phase A: overfit micro set
- 1 to 2 files, fixed snippets, 200 to 500 steps
- Expect onset F1 to rise clearly above zero

Phase B: tiny subset
- 1% data, 10 to 30 epochs
- Check onset precision and recall are both non-zero

Phase C: normal training
- full split

### 7.2 Metrics

Track and log:
- onset precision/recall/F1 at several thresholds (not only 0.5)
- PR-AUC for onset if feasible
- frame F1
- duration MAE on onset positions

### 7.3 Threshold calibration

During validation:
- Sweep onset threshold and store best threshold per epoch or per run.
- Use calibrated threshold in qualitative plots and inference defaults.

### 7.4 Visualization and runtime behavior

Required changes:
- Do not run expensive full inference visualization every epoch by default.
- Add config flag visualize_every_n_epochs, default 10 or disabled for sanity runs.


## 8. Inference Rewrite Requirements

### 8.1 Config consistency

Inference must load and use run-specific config and normalization stats.
Do not use global config constants for trained model inference path.

### 8.2 Chunked inference

Use overlap windowing for long audio:
- chunk length in frames
- overlap and blend strategy

### 8.3 Decoding

Decode notes from onset plus frame:
- onset gate starts notes
- frame maintains note activity
- duration head can refine note end estimates

### 8.4 Outputs

Produce:
- piano roll arrays
- optional MIDI export
- confidence diagnostics


## 9. File-Level Rewrite Plan

### 9.1 src_new/model.py

Actions:
- Replace current PianoTranscriptionModel with PianoTranscriptionModelV2.
- Remove mean pooling bottleneck in spectral stage.
- Implement key query projection and per-key temporal encoder.
- Keep tensor contracts expected by train and inference.

### 9.2 src_new/train.py

Actions:
- Rework loss class to truly support focal and/or pos_weight.
- Wire imbalance config into active loss construction.
- Add threshold sweep metrics.
- Add visualize_every_n_epochs and disable by default for sanity runs.
- Add short overfit mode command path.

### 9.3 src_new/dataset.py

Actions:
- Add optional onset-aware snippet sampling strategy.
- Add optional normalization output or standardized spectrogram mode.

### 9.4 src_new/inference.py

Actions:
- Ensure preprocessing uses saved run config.
- Add chunked inference and threshold calibration support.
- Keep output dictionary contract stable.

### 9.5 src_new/utils.py

Actions:
- Remove hard dependency on global CONFIG in visualization path.
- Visualization should accept explicit config passed from training run.

### 9.6 src_new/config.py

Actions:
- Clean dead fields or make all fields active.
- Add new fields for threshold sweep, visualization cadence, normalization mode, and sampling mode.


## 10. Acceptance Criteria

A rewrite is complete only if all are true:

1. Unit/smoke checks
- Forward pass shape tests pass for both duration modes.

2. Overfit check
- On 1 file overfit mode, onset F1 rises well above zero and predictions align visually.

3. Tiny subset sanity
- On 1% data, onset precision and recall both become non-zero within first 10 to 20 epochs.

4. Inference consistency
- Inference uses run config and matches training preprocessing settings.

5. Runtime usability
- Training loop runs without per-epoch heavy visualization unless enabled.


## 11. Suggested Execution Order For Fresh Agent

1. Implement model v2 with key-query projection.
2. Implement loss rewrite and imbalance wiring.
3. Add threshold sweep metrics.
4. Fix inference and visualization config consistency.
5. Run overfit test and inspect plots.
6. Run tiny subset sanity (10 to 30 epochs).
7. Tune weights and thresholds based on early metrics.


## 12. Minimal Sanity Command Template

Use a small run config override (epochs, data fraction, snippets per file, num workers).

For sanity runs:
- epochs: 10 to 30
- data_fraction: 0.01
- snippets_per_file: 2 to 5
- batch_size: 2 to 4
- num_workers: 0
- visualization cadence: disabled or every 10 epochs


## 13. Notes For The Next Agent

Do not attempt minor patching only. The architecture and loss path should be treated as a coordinated rewrite.

Most important conceptual correction:
- Preserve pitch-structured information from spectral stage into temporal stage.
- Avoid single-vector frame collapse before key prediction.

