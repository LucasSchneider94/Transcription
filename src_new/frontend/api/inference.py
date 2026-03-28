"""
Inference bridge: loads the trained PianoTranscriptionModel from src_new
and runs it on a given audio file segment.
Returns a piano roll + note list as JSON-serialisable dicts.
"""

import sys
import os
import json
import numpy as np
import torch
import pretty_midi

# ── path setup ────────────────────────────────────────────────────────────────
# src_new/frontend/api  →  go up 3 levels to repo root, then into src_new
API_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.abspath(os.path.join(API_DIR, "..", "..", ".."))
SRC_NEW  = os.path.join(ROOT, "src_new")
sys.path.insert(0, SRC_NEW)

from model import PianoTranscriptionModel   # src_new/model.py

# ── default config (mirrors src_new/config.py) ───────────────────────────────
DEFAULT_CONFIG = {
    "sample_rate":       48000,
    "roll_fps":          100,
    "n_mels":            352,        # 88 * 4
    "n_fft":             4096,
    "hop_length":        480,        # sample_rate / roll_fps
    "num_keys":          88,
    "hidden_size":       256,
    "num_heads":         8,
    "num_layers":        4,
    "dropout":           0.2,
    "duration_mode":     "log",
    "num_duration_bins": 8,
}

ONSET_THRESH  = float(os.environ.get("ONSET_THRESH",  "0.5"))
FRAME_THRESH  = float(os.environ.get("FRAME_THRESH",  "0.3"))
CHECKPOINT    = os.environ.get(
    "CHECKPOINT_PATH",
    os.path.join(SRC_NEW, "training_run_056", "best_model.pt"),
)
DEVICE = "mps" if torch.backends.mps.is_available() else \
         "cuda" if torch.cuda.is_available() else "cpu"

# ── lazy model load ────────────────────────────────────────────────────────────
_model        = None
_model_config = None

def _get_model():
    global _model, _model_config

    if _model is not None:
        return _model, _model_config

    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}\n"
            f"Set the CHECKPOINT_PATH environment variable to the correct path."
        )

    # Load config.json from the same folder as the checkpoint if available
    cfg = DEFAULT_CONFIG.copy()
    config_path = os.path.join(os.path.dirname(CHECKPOINT), "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg.update(json.load(f))
        print(f"[inference] Loaded config from {config_path}")
    else:
        print(f"[inference] No config.json next to checkpoint – using defaults")

    model = PianoTranscriptionModel(
        input_features    = cfg["n_mels"],
        num_keys          = cfg["num_keys"],
        transformer_dim   = cfg["hidden_size"],
        num_heads         = cfg["num_heads"],
        num_layers        = cfg["num_layers"],
        dropout           = cfg["dropout"],
        duration_mode     = cfg["duration_mode"],
        num_duration_bins = cfg["num_duration_bins"],
    ).to(DEVICE)

    state = torch.load(CHECKPOINT, map_location=DEVICE)
    # Support both raw state-dicts and wrapped checkpoints
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    print(f"[inference] Model loaded from {CHECKPOINT} on {DEVICE}")

    _model        = model
    _model_config = cfg
    return _model, _model_config


# ── audio helpers ──────────────────────────────────────────────────────────────
def _load_audio(path: str, start: float, end: float | None,
                sample_rate: int) -> np.ndarray:
    import librosa
    y, _ = librosa.load(
        path, sr=sample_rate, mono=True,
        offset=start,
        duration=None if end is None else end - start,
    )
    return y


def _compute_melspec(y: np.ndarray, cfg: dict) -> torch.Tensor:
    import librosa
    S = librosa.feature.melspectrogram(
        y=y,
        sr=cfg["sample_rate"],
        n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"],
        n_mels=cfg["n_mels"],
    )
    S_db = librosa.power_to_db(S, ref=np.max).astype(np.float32)
    # (n_mels, T)  →  (1, 1, n_mels, T)
    return torch.from_numpy(S_db).unsqueeze(0).unsqueeze(0)


# ── duration helper ────────────────────────────────────────────────────────────
def _log_duration_to_seconds(log_dur: np.ndarray) -> np.ndarray:
    """Invert the log-duration encoding used during training."""
    return np.expm1(np.clip(log_dur, 0, None))


# ── piano-roll → MIDI note list ────────────────────────────────────────────────
def _predictions_to_notes(onset_roll: np.ndarray,
                           duration_pred: np.ndarray,
                           frame_roll: np.ndarray,
                           fps: float,
                           onset_thresh: float,
                           frame_thresh: float,
                           min_pitch: int = 21) -> list[dict]:
    """
    Convert frame-level onset + duration predictions to a list of note dicts.

    Key design:
    - Scan per pitch.  The FIRST frame where onset >= onset_thresh triggers a note.
    - A refractory period then suppresses new onsets for that pitch until the
      frame roll drops back below frame_thresh (i.e. the note has ended).
      This prevents a single held note from being emitted once per frame.
    - Note duration comes from the predicted log-duration at the onset frame.
      If that gives < 50 ms, fall back to scanning the frame roll forward.
    """
    notes = []
    n_frames, n_keys = onset_roll.shape

    for key_idx in range(n_keys):
        pitch     = key_idx + min_pitch
        in_note   = False   # refractory flag: True while a note is still sounding

        for f in range(n_frames):
            frame_active = frame_roll[f, key_idx] >= frame_thresh
            is_onset     = onset_roll[f, key_idx] >= onset_thresh

            if in_note:
                # Stay in refractory until the frame roll goes quiet
                if not frame_active:
                    in_note = False
                continue  # never re-trigger while note is still active

            if is_onset:
                in_note   = True
                start_sec = f / fps

                # Duration from the model's log-regression head at this onset frame
                raw_dur = float(duration_pred[f, key_idx])
                dur_sec = float(np.expm1(max(0.0, raw_dur)))   # inverse of log1p

                if dur_sec < 0.05:
                    # Fallback: follow the frame roll forward
                    end_f = f + 1
                    while end_f < n_frames and frame_roll[end_f, key_idx] >= frame_thresh:
                        end_f += 1
                    dur_sec = max(0.05, (end_f - f) / fps)

                notes.append({
                    "pitch":     pitch,
                    "start":     round(start_sec, 4),
                    "end":       round(start_sec + dur_sec, 4),
                    "midi_note": pitch,
                    "note_name": pretty_midi.note_number_to_name(pitch),
                })

    notes.sort(key=lambda n: n["start"])
    return notes


# ── public API ─────────────────────────────────────────────────────────────────
def run_inference(audio_path: str,
                  start_time: float = 0.0,
                  end_time: float | None = None) -> dict:

    model, cfg = _get_model()
    fps        = cfg["roll_fps"]

    y        = _load_audio(audio_path, start_time, end_time, cfg["sample_rate"])
    duration = len(y) / cfg["sample_rate"]

    spec = _compute_melspec(y, cfg).to(DEVICE)   # (1, 1, n_mels, T)

    with torch.no_grad():
        out = model(spec)   # dict: onset, duration, frame

    # onset / frame:  (1, T, 88) logits  →  sigmoid  →  numpy
    onset_roll = torch.sigmoid(out["onset"]).squeeze(0).cpu().numpy()   # (T, 88)
    frame_roll = torch.sigmoid(out["frame"]).squeeze(0).cpu().numpy()   # (T, 88)

    # duration: (1, T, 88)  log-regression values
    duration_pred = out["duration"].squeeze(0).cpu().numpy()            # (T, 88)

    notes = _predictions_to_notes(
        onset_roll, duration_pred, frame_roll,
        fps=fps,
        onset_thresh=ONSET_THRESH,
        frame_thresh=FRAME_THRESH,
    )

    return {
        "duration":   round(duration, 3),
        "fps":        float(fps),
        "n_frames":   int(onset_roll.shape[0]),
        "piano_roll": frame_roll.tolist(),    # (T, 88) – used for roll viewer
        "onset_roll": onset_roll.tolist(),    # (T, 88)
        "notes":      notes,
    }
