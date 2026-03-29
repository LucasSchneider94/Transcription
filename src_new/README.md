# Piano Transcription

End-to-end automatic piano transcription: log-mel spectrogram → CNN + Transformer → onset / frame predictions per key.

---

## Table of Contents

1. [Project layout](#project-layout)
2. [Data preparation](#data-preparation)
3. [Model architecture](#model-architecture)
4. [Training pipeline](#training-pipeline)
5. [DataLoader design](#dataloader-design)
6. [Metrics](#metrics)
7. [Inference](#inference)
8. [Utilities](#utilities)
9. [Frontend (not wired up yet)](#frontend-not-wired-up-yet)
10. [Quick-start commands](#quick-start-commands)

---

## Project layout

```
src_new/
├── config.py              # Full-training hyperparameters
├── config_overfit.py      # Overfit sanity-check config (2 files, 500 epochs)
├── data_preparation.py    # MAESTRO → .npz preprocessing
├── dataset.py             # PianoTranscriptionDataset + FileGroupedSampler
├── model.py               # CNN encoder + Temporal Transformer
├── train.py               # Training loop, metrics, visualisation
├── inference.py           # Audio → piano roll (standalone script)
├── inference_config.py    # Paths / time range for inference script
├── utils.py               # run-folder management, checkpointing, curve plots
├── resume_config.py       # Config for resuming a previous run
├── processed_data_17_18/  # Preprocessed .npz files (not in git)
├── maestro-v3.0.0.json    # MAESTRO metadata (train/val/test splits)
├── Test_data/             # Small WAV + MIDI files for inference probing
├── training_run_XXX/      # Output folders, one per run (not in git)
└── frontend/              # FastAPI + Next.js viewer (not connected yet)
```

---

## Data preparation

**Script**: `data_preparation.py`  
**Run once** before training. Processes the MAESTRO v3 dataset and writes one `.npz` file per recording.

### What it does

For each `(audio.wav, score.midi)` pair:

1. **Spectrogram** — `librosa.feature.melspectrogram` at 48 kHz, 352 mel bins (= 88 × 4), n_fft=4096, hop=480 samples (100 fps). Converted to dB with `librosa.power_to_db`. Shape: `(352, T)`.
2. **Onset roll** — Binary `(T, 88)`. A note's onset is marked for `onset_frames=2` consecutive frames starting at the attack frame. Pedal onsets are not included.
3. **Frame roll** — Binary `(T, 88)`. Active when the note is acoustically sounding, including sustain-pedal extension. Derived from `pretty_midi` note end times (not MIDI NoteOff, which ignores pedal).
4. **Duration roll** — Exact duration in seconds at each onset frame, zero elsewhere. Stored for diagnostic use; the current model does not predict duration.
5. **Pedal roll** — Binary `(T, 1)`. Sustain-pedal activity. Also stored but not used by the current model.

All arrays are saved together in a single compressed `.npz` file named `<stem>_piano_roll_with_pedals.npz`. Temporal alignment is enforced: if the spectrogram and the MIDI roll differ in length by rounding, both are trimmed to `min(len_spec, len_roll)`.

### Output layout

```
processed_data_17_18/
└── <stem>_piano_roll_with_pedals.npz
       ├── spectrogram  float32  (352, T)
       ├── onset        float32  (T, 88)
       ├── frame        float32  (T, 88)
       ├── duration     float32  (T, 88)
       └── pedal        float32  (T, 1)
```

### Run it

```bash
cd src_new
python data_preparation.py   # reads CONFIG['split_year_folder'] = '2013' by default
```

The `split_year_folder` key in `config.py` controls which MAESTRO year subfolder is processed. The current processed set (`processed_data_17_18/`) covers years 2017 and 2018.

---

## Model architecture

**File**: `model.py`

```
Input: (B, 1, 352, T)   — single-channel log-mel spectrogram

CNNEncoder
  ConvBlock 1:  freq_kernel=87, time_kernel=9  →  32 ch  →  AvgPool freq ÷4  → (B,32,88,T)
  ConvBlock 2:  freq_kernel=31, time_kernel=9  →  64 ch  →  AvgPool freq ÷2  → (B,64,44,T)
  ConvBlock 3:  freq_kernel=15, time_kernel=9  → 128 ch  →  AvgPool freq ÷4  → (B,128,11,T)
  Flatten freq: 128×11 = 1408
  Linear + LayerNorm  →  (B, T, 256)

TemporalTransformer
  Sinusoidal positional encoding
  4× TransformerEncoderLayer (pre-norm, 8 heads, FFN dim 1024, GELU)
  →  (B, T, 256)

Two linear heads
  onset head:  Linear(256, 88)  bias init −4.0   →  raw logits (B, T, 88)
  frame head:  Linear(256, 88)  bias init −2.0   →  raw logits (B, T, 88)
```

### ConvBlock design

Each block applies a **tall frequency kernel first** (to capture harmonic/overtone relationships spanning many mel bins), then a **wide time kernel** (local temporal dynamics). Both convolutions use same-padding so the spatial size is preserved. A 1×1 residual shortcut allows free gradient flow. Normalisation uses **GroupNorm** (`min(8, channels)` groups), not BatchNorm — GroupNorm has no running statistics and behaves identically in train and eval mode, which matters for single-sample inference and was critical to fix a bug where BatchNorm's running stats diverged from per-batch stats causing visualisations to look wrong even at near-zero training loss.

### Frequency kernel schedule

The kernels shrink at each stage (87 → 31 → 15) because the effective receptive field grows through the pooling chain; a large kernel at a coarser scale would be wasteful and risk over-smoothing.

### Head bias initialisation

The negative bias initialisation pushes initial sigmoid outputs well below 0.5 to match the extreme class imbalance (onsets and active frames are rare relative to silence), avoiding saturated gradients on the overwhelming negative class at the start of training.

---

## Training pipeline

**Script**: `train.py`  
**Config**: `config.py` (full run) or `config_overfit.py` (sanity check)

```bash
cd src_new
python train.py             # full run
python train.py --overfit   # 2-file memorisation test
```

Each run creates `training_run_XXX/` (auto-incremented) containing:

```
training_run_XXX/
├── config.json               # snapshot of the config used
├── normalization_stats.json  # global mean/std computed from training files
├── best_model.pt             # checkpoint at best val loss
├── best_threshold.json       # {"onset_threshold": 0.XX}
├── last_model.pt             # checkpoint at final epoch
├── training_stats.json       # full loss + metric history (flushed every 10 epochs)
├── training_curves.png       # final loss + metric curves
├── tensorboard/              # TensorBoard event files
├── visualizations/           # vis_epoch_XXXX.png  — 2×2 sigmoid heatmap every N epochs
└── probe_inference/          # probe_epoch_XXXX.png — live inference on fixed audio
```

### Loss functions

- **Onset**: Focal loss, α=0.9, γ=2.0 — heavily down-weights easy negatives (silence).
- **Frame**: Focal loss, α=0.5, γ=2.0 — milder alpha since frames are less sparse than onsets.
- **Combined**: `3 × onset_loss + 1 × frame_loss` — onset is weighted higher because precise attack detection is harder and more perceptually important.
- **Overfit mode**: plain BCE with computed `pos_weight` — forces the model to memorise without focal loss smoothing.

### Learning rate schedule

- Linear warmup for `warmup_epochs=10` epochs.
- `ReduceLROnPlateau` on val loss: factor=0.5, patience=15, min_lr=1e-6.

### Visualisation

Every `visualize_every_n_epochs` (default 10) two things are saved and pushed to TensorBoard:

1. **Val-batch heatmap** (`vis_epoch_XXXX.png`) — 2×2 grid: GT onset | GT frame | predicted onset sigmoid | predicted frame sigmoid. Uses a fixed val batch picked at startup.
2. **Probe inference** (`probe_epoch_XXXX.png`) — the full inference pipeline (audio → spectrogram → model → threshold → decoded notes) run on the first 20 s of a fixed test WAV, compared against its MIDI ground truth. Defined in `inference_config.py`.

### Overfit sanity check

Before trusting any full run, always verify with:

```bash
python train.py --overfit
```

Expected behaviour: training loss should drop below 0.01 and `onset_joint_f1` should reach 1.000 within ~150 epochs. If it doesn't, something is broken in the model or data pipeline. The overfit config uses 2 files, batch size 1, no dropout, and fixed snippet positions.

---

## DataLoader design

**Key constraint**: each `.npz` file is ~88 MB. With the full 444 training files and `snippets_per_file=20`, naively shuffling and using multiple workers causes **8,880 full file reads per epoch** — one per snippet × one per worker process.

### Fix: FileGroupedSampler + LRU file cache

`FileGroupedSampler` (in `train.py`) yields indices so that **all 20 snippets from the same file are served consecutively**. File order is reshuffled each epoch via `set_epoch(epoch)`.

`PianoTranscriptionDataset._load_file` keeps the last 2 files in an in-process LRU cache. With consecutive indices, the second and subsequent snippet requests for a file are served from RAM — **reducing disk reads from 8,880 to 444 per epoch**.

`num_workers` must be **0** (single process). Worker processes have separate address spaces and cannot share the cache; multi-process loading would defeat the entire scheme.

### Normalisation stats

Computed once before training by sampling every 50th frame from 20 files (≈ 4 s of compute). Stored as `{"mean": float, "std": float}` and applied per-sample in `__getitem__`.

### Snippet sampling modes

- `random` — uniform random start position within the file.
- `onset_aware` — at least `onset_sampling_min_active_ratio=0.5` of sampled snippets are biased toward regions containing onsets, to avoid training on long silence patches.

The val dataset always uses `fixed_snippets=True` (deterministic positions seeded once) for reproducible evaluation.

---

## Metrics

All metrics are computed in `train.py:compute_metrics` and `sweep_thresholds`. Three streams:

| Metric | What it measures |
|---|---|
| `onset_joint_f1` | Onset with **correct pitch AND timing** within ±`tol_frames` |
| `onset_time_f1` | Onset with **correct timing** (any key fired within tolerance) |
| `frame_f1` | Frame-level binary F1 across all 88 keys |

`tol_frames=2` at 100 fps = **±20 ms tolerance**. This is strict — a prediction off by 3 frames counts as a false positive and a missed onset simultaneously.

The precision/recall/F1 calculation uses **max-dilation** (`maximum_filter1d`) to implement the tolerance window efficiently: GT onsets expanded by ±tol_frames and compared to predictions (and vice versa for recall), then subtracted to get FP/FN.

At each epoch a threshold sweep over `[0.2, 0.3, 0.4, 0.5, 0.6]` selects the threshold maximising `onset_joint_f1`. The best threshold is saved to `best_threshold.json` and loaded automatically by inference.

---

## Inference

**Script**: `inference.py`  
**Config**: `inference_config.py`

```bash
cd src_new
python inference.py
# or with flags:
python inference.py --audio path/to/audio.wav --model training_run_070/best_model.pt --out results/
```

### What it does

1. Loads `config.json` and `normalization_stats.json` from the model's run folder (falls back to `config.py` defaults if absent).
2. Runs `librosa.feature.melspectrogram` → log-mel → normalise with training stats.
3. Slices to the requested time range (`start_time` / `end_time` from `inference_config.py`).
4. Feeds spectrogram through model in eval mode, chunked if needed.
5. Applies onset threshold (loaded from `best_threshold.json`), optional onset–frame gating (a frame prediction is only kept if it overlaps a detected onset), and a temporal median filter to reduce flicker.
6. Saves a comparison PNG (GT vs predicted sigmoid maps + decoded binary roll).

### Probe during training

`_setup_probe` in `train.py` pre-processes the probe audio once at training startup and caches the result. `_save_probe_inference` runs the live model on that cached snippet every `visualize_every_n_epochs` epochs, saving a PNG and pushing it to TensorBoard under `Probe/inference`.

The probe file is set in `inference_config.py`:
```python
'audio_path': 'src_new/Test_data/MIDI-UNPROCESSED_01-03_R1_2014_MID--AUDIO_01_R1_2014_wav--2.wav',
# Scarlatti A-Major Sonata, 0–20 s
```

---

## Utilities

**File**: `utils.py`

| Function | Purpose |
|---|---|
| `get_next_run_folder()` | Scans for `training_run_*` dirs, returns next `training_run_XXX/` |
| `save_json / load_json` | Thin wrappers around `json.dump/load` |
| `save_checkpoint` | Saves epoch, model state dict, optimiser, scheduler, scaler, loss history |
| `plot_training_curves` | 4-panel matplotlib figure: loss curves + 3 metric curves |

---

## TensorBoard

TensorBoard is launched separately and watches the entire `src_new/` directory, so it automatically picks up every new run:

```bash
cd src_new
../myenv/bin/tensorboard --logdir . --port 6006
# → http://localhost:6006
```

Logged scalars per epoch: `Loss/train`, `Loss/val`, `Metrics/val/*`, `Metrics/train/*`, `LR`.  
Logged images: `ValBatch/sigmoid` (heatmap), `Probe/inference` (full pipeline).

---

## Frontend (not wired up yet)

**Location**: `src_new/frontend/`

Two separate processes:

### FastAPI backend (`frontend/api/`)

Exposes a single endpoint `POST /analyze` that accepts a WAV/MP3/FLAC file plus `start_time` / `end_time` form fields. Returns JSON with `piano_roll`, `onset_roll`, `notes[]`, `fps`, `duration`.

```bash
cd src_new/frontend/api
source ../../../myenv/bin/activate
pip install -r requirements.txt

export CHECKPOINT_PATH=../../training_run_070/best_model.pt
export ONSET_THRESH=0.4   # optional, default 0.5
export FRAME_THRESH=0.3   # optional, default 0.3

uvicorn main:app --reload --port 8000
```

The model is loaded lazily on first request. It reads `config.json` from the same directory as the checkpoint to reconstruct the model architecture.

**Note**: `frontend/api/inference.py` has its own model-loading path that uses the older `PianoTranscriptionModel` constructor signature. It needs to be updated to match the current `model.py` constructor before it works with a recent checkpoint.

### Next.js frontend (`frontend/webapp/`)

```bash
cd src_new/frontend/webapp
npm install
npm run dev
# → http://localhost:3000
```

All `/api/*` requests are proxied to `localhost:8000` via Next.js rewrites.

### Views

- **Piano Roll** — scrollable canvas, raw frame activations, 88-key range.
- **Score** — proportional score: noteheads on a grand staff, x-axis is real time (no barlines, no beat quantisation). Duration shown as a faint tail. The user adds barlines manually once they know the tempo. This is intentional — rhythm quantisation from a live recording is an unsolved research problem and forcing notes onto a beat grid always produces wrong results.

---

## Quick-start commands

```bash
# 0. Activate environment
source myenv/bin/activate   # from repo root

# 1. Preprocess (skip if processed_data_17_18/ already exists)
cd src_new && python data_preparation.py

# 2. Sanity check — must reach onset_joint_f1 = 1.000
python train.py --overfit

# 3. Full training run
python train.py

# 4. Monitor
../myenv/bin/tensorboard --logdir . --port 6006

# 5. Inference on a file
python inference.py --audio Test_data/your_file.wav \
                    --model training_run_070/best_model.pt \
                    --out inference_results/
```
    'output_dir': 'inference_results',
}

# Run inference
python inference.py
```

## Model Architecture

- **Input**: Mel-spectrogram (128 mel bins)
- **Encoder**: Convolutional layers for feature extraction
- **Transformer**: Multi-head self-attention with positional encoding
- **Output**: 91 binary predictions (88 keys + 3 pedals)

## Configuration

Key parameters in `config.py`:

- `num_epochs`: Training epochs
- `batch_size`: Batch size
- `learning_rate`: Learning rate
- `snippet_frames`: Frames per training snippet
- `hidden_size`: Transformer hidden dimension
- `num_heads`: Attention heads
- `num_layers`: Transformer layers

## Training Features

- **Reproducible splits**: Fixed random seed for train/val split
- **Data augmentation**: Random snippet sampling each epoch
- **Automatic checkpointing**: Save progress every epoch
- **Progress monitoring**: Training curves updated every 10 epochs
- **Early stopping**: Best model saved based on validation loss

## License

MIT License

## Acknowledgments

- MAESTRO dataset: https://magenta.tensorflow.org/datasets/maestro
- Inspired by Onsets and Frames architecture
