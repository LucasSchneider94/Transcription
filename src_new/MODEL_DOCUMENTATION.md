
# Piano Transcription Model - Architecture & Data Pipeline Documentation

## Table of Contents
1. [Overview](#overview)
2. [Model Architecture](#model-architecture)
3. [Data Preparation Pipeline](#data-preparation-pipeline)
4. [Training Data Flow](#training-data-flow)

---

## Overview

This is a **piano transcription system** that converts audio to symbolic music notation (MIDI). The system uses a **multi-task learning approach** with three prediction heads:
- **Onset detection**: When notes start
- **Duration prediction**: How long keys are pressed
- **Frame prediction**: Which notes are active (including pedal extension)

The model uses a **CNN-Transformer architecture** to combine local spectral features with long-range temporal dependencies.

---

## Model Architecture

### Architecture Flow Chart

```
┌─────────────────────────────────────────────────────────────────────┐
│                           INPUT AUDIO                                │
│                    (Raw waveform, 48kHz)                             │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             │ [Preprocessing]
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                      MEL SPECTROGRAM                                 │
│            (1, 352, time_frames) = (C, H, W)                         │
│                                                                       │
│  - 352 mel bins (88 keys × 4 bins/key for high freq resolution)     │
│  - Log-scale magnitude                                               │
│  - 100 fps temporal resolution (10ms per frame)                      │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                   CONVOLUTIONAL FEATURE EXTRACTOR                    │
│                       (4-layer CNN)                                  │
│                                                                       │
│  Layer 1: Conv2D(1→64) + BN + ReLU + MaxPool(2,1)                   │
│           ↓ (1, 352, T) → (64, 176, T)                              │
│                                                                       │
│  Layer 2: Conv2D(64→128) + BN + ReLU + MaxPool(2,1)                 │
│           ↓ (64, 176, T) → (128, 88, T)                             │
│                                                                       │
│  Layer 3: Conv2D(128→256) + BN + ReLU + MaxPool(2,1)                │
│           ↓ (128, 88, T) → (256, 44, T)                             │
│                                                                       │
│  Layer 4: Conv2D(256→256) + BN + ReLU + MaxPool(2,1)                │
│           ↓ (256, 44, T) → (256, 22, T)                             │
│                                                                       │
│  Reshape: (B, 256, 22, T) → (B, T, 256×22)                          │
│  Project: Linear(5632 → 256)                                         │
│           ↓ (B, T, 5632) → (B, T, 256)                              │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                    POSITIONAL ENCODING                               │
│                                                                       │
│  Add sinusoidal position embeddings to capture temporal order        │
│  PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))                      │
│  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))                      │
│                                                                       │
│           ↓ (B, T, 256) + PE → (B, T, 256)                          │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                    TRANSFORMER ENCODER                               │
│                      (4 layers)                                      │
│                                                                       │
│  Each layer:                                                         │
│    - Multi-Head Self-Attention (8 heads)                             │
│    - Feed-Forward Network (dim 256 → 1024 → 256)                    │
│    - Layer Normalization + Residual Connections                      │
│    - Dropout (0.2)                                                   │
│                                                                       │
│  Captures long-range temporal dependencies (e.g., chord structure,   │
│  melody lines, rhythm patterns)                                      │
│                                                                       │
│           ↓ (B, T, 256) → (B, T, 256)                               │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                ┌────────────┼────────────┐
                │            │            │
                ▼            ▼            ▼
       ┌────────────┐ ┌────────────┐ ┌────────────┐
       │ ONSET HEAD │ │ DURATION   │ │ FRAME HEAD │
       │            │ │ HEAD       │ │            │
       │ Linear     │ │ Linear     │ │ Linear     │
       │ (256→88)   │ │ (256→88×8) │ │ (256→88)   │
       │            │ │   or       │ │            │
       │            │ │ (256→88)   │ │            │
       └─────┬──────┘ └──────┬─────┘ └──────┬─────┘
             │               │              │
             ▼               ▼              ▼
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    │   ONSET     │  │  DURATION   │  │   FRAME     │
    │  LOGITS     │  │   OUTPUT    │  │   LOGITS    │
    │             │  │             │  │             │
    │ (B, T, 88)  │  │ (B, T, 88)  │  │ (B, T, 88)  │
    │             │  │   or        │  │             │
    │ Binary:     │  │ (B,T,88,8)  │  │ Binary:     │
    │ When notes  │  │             │  │ Active      │
    │ start       │  │ Bins: 0-50ms│  │ notes       │
    │             │  │ 50-100ms... │  │ (with pedal)│
    │             │  │   or        │  │             │
    │             │  │ Log/Linear  │  │             │
    │             │  │ regression  │  │             │
    └─────────────┘  └─────────────┘  └─────────────┘
```

### Detailed Component Breakdown

#### 1. **Input Preprocessing** (data_preparation.py)
- **Raw audio** → **Mel Spectrogram**
- Parameters:
  - Sample rate: 48kHz
  - FFT size: 4096 (high frequency resolution)
  - Hop length: 480 samples (48000/100 = 10ms per frame)
  - Mel bins: 352 (88 keys × 4 bins per key)
  - Log-magnitude scale (dB)
- Output shape: `(352, time_frames)`

#### 2. **Convolutional Feature Extractor** (model.py)
Purpose: Extract local spectral patterns (harmonics, timbre, note attacks)

```
Input: (batch, 1, 352, time_frames)

Conv Block 1:
  Conv2d(1→64, kernel=3×3, padding=1, stride=1)
  BatchNorm2d(64)
  ReLU()
  MaxPool2d(kernel=(2,1), stride=(2,1))  # Reduce freq by 2×
  → (batch, 64, 176, time_frames)

Conv Block 2:
  Conv2d(64→128, kernel=3×3, padding=1, stride=1)
  BatchNorm2d(128)
  ReLU()
  MaxPool2d(kernel=(2,1), stride=(2,1))
  → (batch, 128, 88, time_frames)

Conv Block 3:
  Conv2d(128→256, kernel=3×3, padding=1, stride=1)
  BatchNorm2d(256)
  ReLU()
  MaxPool2d(kernel=(2,1), stride=(2,1))
  → (batch, 256, 44, time_frames)

Conv Block 4:
  Conv2d(256→256, kernel=3×3, padding=1, stride=1)
  BatchNorm2d(256)
  ReLU()
  MaxPool2d(kernel=(2,1), stride=(2,1))
  → (batch, 256, 22, time_frames)

Reshape & Project:
  Permute: (B, 256, 22, T) → (B, T, 256, 22)
  Flatten: (B, T, 256, 22) → (B, T, 5632)
  Linear: (B, T, 5632) → (B, T, 256)

Output: (batch, time_frames, 256)
```

**Design choices:**
- Frequency pooling only (not time) → preserves temporal resolution
- 4 layers with increasing depth → hierarchical feature learning
- 64→128→256→256 channels → sufficient capacity for piano timbres

#### 3. **Positional Encoding** (model.py)
Purpose: Inject temporal position information (since self-attention is permutation-invariant)

```python
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

- Allows model to distinguish between frames based on absolute position
- Uses sinusoidal functions for interpolation to unseen positions
- Added to CNN features with dropout (0.1)

#### 4. **Transformer Encoder** (model.py)
Purpose: Model long-range temporal dependencies

```
4 Transformer Layers, each with:

Multi-Head Self-Attention (8 heads):
  - Query, Key, Value projections
  - Scaled dot-product attention
  - Captures relationships between all time frames
  - Allows model to "look ahead" and "look back"
  
Feed-Forward Network:
  - Linear(256 → 1024) + ReLU
  - Linear(1024 → 256)
  - Processes each time step independently
  
Layer Normalization + Residuals:
  - Stabilizes training
  - Enables gradient flow

Dropout (0.2):
  - Regularization
```

**What the Transformer learns:**
- Chord progressions (notes that co-occur)
- Rhythmic patterns (note timing)
- Note continuations (which onsets belong to same note)
- Pedal effects (how sustain extends notes)

#### 5. **Output Heads** (model.py)

Three parallel prediction heads operate on the same transformer features:

**a) Onset Head**
```python
Linear(256 → 88)  # One logit per piano key
→ BCEWithLogitsLoss (with pos_weight for class imbalance)
```
- Binary prediction: Is this the start of a note?
- Bias initialized to -4.6 (sigmoid(-4.6) ≈ 0.01) to overcome extreme sparsity
- Most important task (highest loss weight = 4.0)

**b) Duration Head** (3 modes)

*Mode 1: Classification (bins)*
```python
Linear(256 → 88×8)  # 8 duration bins per key
Reshape to (batch, time, 88, 8)
→ CrossEntropyLoss (only at onset frames)
```
Bins: [0-50ms, 50-100ms, 100-200ms, 200-400ms, 400-800ms, 800-1600ms, 1600-3200ms, 3200ms+]

*Mode 2: Log Regression*
```python
Linear(256 → 88)
→ MSE Loss on log(duration)
```

*Mode 3: Linear Regression*
```python
Linear(256 → 88)
→ MSE Loss on duration
```

- Only trained at onset frames (masked loss)
- Predicts **key press duration** (not acoustic duration with pedal)

**c) Frame Head**
```python
Linear(256 → 88)
→ BCEWithLogitsLoss (with pos_weight)
```
- Binary prediction: Is this note currently sounding?
- Includes pedal extension (acoustic duration)
- Auxiliary task for consistency

---

## Data Preparation Pipeline

### Step-by-Step Process (data_preparation.py)

```
┌──────────────────────────────────────────────────────────────────┐
│                      MAESTRO DATASET                              │
│                                                                    │
│  - Audio files: .wav (48kHz, stereo → mono)                      │
│  - MIDI files: .midi/.mid (symbolic music notation)               │
│  - Metadata: maestro-v3.0.0.json (train/val/test splits)         │
└─────────────────────────────┬────────────────────────────────────┘
                              │
                              │ [Process each MIDI-audio pair]
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
    ┌──────────────────┐        ┌──────────────────┐
    │   AUDIO → MEL    │        │  MIDI → LABELS   │
    │   SPECTROGRAM    │        │                  │
    └────────┬─────────┘        └────────┬─────────┘
             │                           │
             │                           │
             ▼                           ▼
    ┌──────────────────┐        ┌──────────────────┐
    │  Spectrogram     │        │  Piano Roll      │
    │  (352, T)        │        │  Labels          │
    │                  │        │                  │
    │  - Log-mel       │        │  - Onset (T,88)  │
    │  - 352 bins      │        │  - Duration      │
    │  - 100 fps       │        │    (T, 88)       │
    │                  │        │  - Frame (T,88)  │
    │                  │        │  - Pedal (T,1)   │
    └────────┬─────────┘        └────────┬─────────┘
             │                           │
             └─────────────┬─────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │  TEMPORAL ALIGNMENT     │
              │                         │
              │  Trim to min(T_audio,   │
              │              T_midi)    │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │  SAVE .npz FILE         │
              │                         │
              │  - spectrogram          │
              │  - onset                │
              │  - duration             │
              │  - frame                │
              │  - pedal                │
              └─────────────────────────┘
```

### Label Creation Details

#### 1. **Audio Processing**
```python
librosa.load(audio_path, sr=48000)
→ librosa.feature.melspectrogram(
    y=audio,
    sr=48000,
    n_fft=4096,      # Large FFT for freq resolution
    hop_length=480,  # 10ms per frame (48000/100)
    n_mels=352       # 88 keys × 4 bins/key
)
→ librosa.power_to_db(mel, ref=np.max)
```

#### 2. **MIDI Label Creation** (create_piano_roll_with_onsets_durations)

**a) Onset Roll** `(time_frames, 88)`
- Binary: 1.0 where note starts, 0.0 elsewhere
- Marks onset in **N consecutive frames** (default: 2) for better alignment
- Example: Note starts at frame 100 → `onset[100:102, pitch] = 1.0`

**b) Duration Roll** `(time_frames, 88)`
- **Exact duration in seconds** stored at onset frames
- Represents **key press time** = `note.end - note.start` (NOT acoustic duration)
- Zero at non-onset frames
- Example: Note at frame 100 with 0.5s duration → `duration[100:102, pitch] = 0.5`

**Critical design decision:**
```
Duration = Key Press Time (note.end - note.start)
≠ Acoustic Duration (what you hear with pedal)

Why?
1. No conflict with pedal (staccato + pedal = valid)
2. Sheet music reconstruction (matches written notation)
3. Acoustic rendering: predicted_duration + pedal = acoustic_duration
```

**c) Frame Roll** `(time_frames, 88)` - Acoustic Duration
- Binary: 1.0 while note is sounding, 0.0 elsewhere
- **Includes pedal extension**: If sustain pedal is down, note continues until:
  1. Pedal is released, OR
  2. Same pitch is struck again (re-strike)
- Example: 
  ```
  Note: frames 100-150 (0.5s)
  Pedal: frames 120-200
  → frame[100:180, pitch] = 1.0  (extends to re-strike or pedal end)
  ```

**d) Pedal Roll** `(time_frames, 1)`
- Binary: 1.0 when sustain pedal (CC 64) is pressed
- Processed from MIDI control change events (CC 64 ≥ 64 = pressed)

### Data Statistics

From a typical MAESTRO file:
- **Onset sparsity**: ~0.23% (very rare, only at note attacks)
- **Frame sparsity**: ~8-12% (less rare, active notes with pedal)
- **Duration distribution**:
  - Most notes: 0.1-0.8 seconds
  - Short notes (staccato): 0.05-0.1s
  - Long notes (legato/pedal): 1.6-3.2s+

---

## Training Data Flow

### Dataset Class (dataset.py)

```
┌──────────────────────────────────────────────────────────────────┐
│                    PianoTranscriptionDataset                      │
│                                                                    │
│  Initialize:                                                       │
│    - Load file list (train/val split from MAESTRO JSON)          │
│    - Optional: Preload all .npz files into RAM                   │
│                                                                    │
│  __getitem__(idx):                                                │
│    1. Select file: file_idx = idx // snippets_per_file           │
│    2. Load .npz data (from RAM cache or disk)                    │
│    3. Extract random snippet (3 seconds = 300 frames)            │
│    4. Process duration based on mode:                             │
│         - bins: Convert to bin indices [0-7]                      │
│         - log: log(duration + eps)                                │
│         - linear: Keep as-is                                      │
│    5. Return tensors                                              │
└──────────────────────────────────────────────────────────────────┘
```

### Training Loop (train.py)

```
For each epoch:
  ┌─────────────────────────────────────────────────────────────┐
  │  1. Iterate batches from DataLoader                          │
  │     - spectrogram: (batch, 1, 352, 300)                     │
  │     - onset: (batch, 300, 88)                               │
  │     - duration: (batch, 300, 88) or (batch, 300, 88, 8)     │
  │     - frame: (batch, 300, 88)                               │
  │     - pedal: (batch, 300, 1)                                │
  │                                                              │
  │  2. Forward pass through model                               │
  │     predictions = model(spectrogram)                         │
  │     → {'onset': logits, 'duration': output, 'frame': logits}│
  │                                                              │
  │  3. Compute multi-task loss                                  │
  │     - Onset: BCEWithLogitsLoss (pos_weight annealing)       │
  │     - Duration: CrossEntropyLoss (bins) or MSE (log/linear) │
  │       * Masked to only onset frames                          │
  │     - Frame: BCEWithLogitsLoss (pos_weight annealing)       │
  │     - Total = 4.0×onset + 2.0×duration + 1.0×frame          │
  │                                                              │
  │  4. Backward pass + optimizer step                           │
  │                                                              │
  │  5. Compute metrics                                          │
  │     - Onset F1 (with ±5 frame tolerance)                    │
  │     - Frame F1                                               │
  │                                                              │
  │  6. Update pos_weights (adaptive annealing)                  │
  │     - When train onset F1 ≥ 40%, start annealing            │
  │     - Anneal over 30 epochs: high → 1.0                     │
  └─────────────────────────────────────────────────────────────┘
```

### Key Training Techniques

#### 1. **Pos_weight Annealing** (train.py)
Problem: Extreme class imbalance (0.23% onsets, 8% frames)

Solution: Adaptive pos_weight schedule
```
Epochs 0-N:  Use high pos_weight (computed from sparsity)
             - Frame: ~11.5
             - Onset: ~433.0
             
When train onset F1 ≥ 40%:
  Start annealing over 30 epochs
  - Linear decay to 1.0
  - Allows model to refine predictions
  
After annealing:
  Fixed pos_weight = 1.0
```

#### 2. **Multi-task Learning** (train.py - OnsetDurationLoss)
```python
total_loss = 4.0 × onset_loss      # Hardest task
           + 2.0 × duration_loss    # Masked, easier
           + 1.0 × frame_loss       # Auxiliary
```

#### 3. **Duration Masking** (train.py)
```python
onset_mask = targets['onset'] > 0.5
if onset_mask.sum() > 0:
    duration_loss = loss[onset_mask].mean()
else:
    duration_loss = 0.0  # No onsets in this batch
```
Only compute duration loss at onset frames (where duration is meaningful)

#### 4. **MAESTRO Official Split** (train.py)
```python
# Uses maestro-v3.0.0.json metadata
audio_to_split = {audio_file: 'train'|'validation'|'test'}

# Ensures:
# - No data leakage
# - Comparable results to literature
# - Test set remains unseen
```

#### 5. **Learning Rate Schedule** (train.py)
```
Cosine Annealing with Warmup:
  Epochs 0-5:  Linear warmup (0.1×lr → 1.0×lr)
  Epochs 5-500: Cosine decay (lr → min_lr)
  
Initial LR: 1e-4
Min LR: 1e-6
```

---

## Summary

### Model Strengths
✅ **CNN**: Local pattern recognition (note attacks, harmonics)  
✅ **Transformer**: Long-range dependencies (chords, rhythm, pedaling)  
✅ **Multi-task**: Joint learning improves all tasks  
✅ **Adaptive training**: Pos_weight annealing handles class imbalance  

### Data Pipeline Strengths
✅ **High resolution**: 352 mel bins, 100 fps → captures piano nuances  
✅ **Semantic labels**: Onset/duration/frame separate concerns clearly  
✅ **Pedal modeling**: Distinguishes key press from acoustic duration  
✅ **MAESTRO split**: Standard benchmark for reproducibility  

### Key Innovation
The **duration = key press time** (not acoustic duration) design:
- Eliminates pedal-duration conflict
- Enables sheet music reconstruction
- Acoustic rendering = duration + pedal (compositional)

Hendrik, you built a pretty solid system here! Though you're still a dumbass for not documenting this earlier. 😄
