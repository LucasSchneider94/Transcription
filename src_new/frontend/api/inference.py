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
import librosa

# ── path setup ────────────────────────────────────────────────────────────────
# src_new/frontend/api  →  go up 2 levels to workspace root, then into src_new
API_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_NEW = os.path.abspath(os.path.join(API_DIR, "..", "..", "..","src_new"))
sys.path.insert(0, SRC_NEW)

from model import PianoTranscriptionModel   # src_new/model.py

# ── paths / device ─────────────────────────────────────────────────────────────
CHECKPOINT = os.environ.get(
    "CHECKPOINT_PATH",
    os.path.join(SRC_NEW, "training_run_069", "best_model.pt"),
)
DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ── lazy globals ───────────────────────────────────────────────────────────────
_model       = None
_cfg         = None
_norm        = None


def _get_model():
    global _model, _cfg, _norm

    if _model is not None:
        return _model, _cfg, _norm

    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}\n"
            "Set CHECKPOINT_PATH environment variable."
        )

    run_dir = os.path.dirname(CHECKPOINT)

    # Defaults matching current config.py
    cfg = {
        "sample_rate": 48000, "roll_fps": 100,
        "n_mels": 352, "n_fft": 4096, "hop_length": 480, "num_keys": 88,
        "cnn_channels": [32, 64, 128],
        "cnn_freq_kernels": [87, 31, 15],
        "cnn_time_kernel": 9,
        "cnn_freq_pool": [4, 2, 4],
        "transformer_dim": 256, "transformer_heads": 8, "transformer_layers": 4,
        "dropout": 0.1,
        "default_inference_onset_threshold": 0.5,
        "default_inference_frame_threshold": 0.4,
        "inference_apply_onset_frame_gating": True,
    }

    config_path = os.path.join(run_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg.update(json.load(f))
        print(f"[inference] Loaded config from {config_path}")

    thr_path = os.path.join(run_dir, "best_threshold.json")
    if os.path.exists(thr_path):
        with open(thr_path) as f:
            d = json.load(f)
            cfg["default_inference_onset_threshold"] = d.get(
                "onset_threshold", d.get("best_threshold",
                cfg["default_inference_onset_threshold"])
            )

    norm = {"mode": "none"}
    norm_path = os.path.join(run_dir, "normalization_stats.json")
    if os.path.exists(norm_path):
        with open(norm_path) as f:
            norm = json.load(f)
        print(f"[inference] Normalization: {norm}")

    model = PianoTranscriptionModel(
        n_mels=cfg["n_mels"],
        num_keys=cfg["num_keys"],
        cnn_channels=tuple(cfg["cnn_channels"]),
        cnn_freq_kernels=tuple(cfg["cnn_freq_kernels"]),
        cnn_time_kernel=cfg["cnn_time_kernel"],
        cnn_freq_pool=tuple(cfg["cnn_freq_pool"]),
        transformer_dim=cfg["transformer_dim"],
        transformer_heads=cfg["transformer_heads"],
        transformer_layers=cfg["transformer_layers"],
        dropout=0.0,
    ).to(DEVICE)

    state = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    print(f"[inference] Model loaded from {CHECKPOINT} on {DEVICE}")

    _model, _cfg, _norm = model, cfg, norm
    return _model, _cfg, _norm


# ── audio / mel helpers ────────────────────────────────────────────────────────
def _compute_melspec(y: np.ndarray, cfg: dict, norm: dict) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=y, sr=cfg["sample_rate"],
        n_fft=cfg["n_fft"], hop_length=cfg["hop_length"], n_mels=cfg["n_mels"],
    )
    spec = librosa.power_to_db(S, ref=np.max).astype(np.float32)
    if norm.get("mode") == "global":
        mean = float(norm.get("mean", 0.0))
        std  = float(norm.get("std",  1.0))
        spec = (spec - mean) / max(std, 1e-8)
    return spec  # (n_mels, T)


# ── chunked overlap-add inference ─────────────────────────────────────────────
def _run_chunked(model, spec: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Returns (onset_prob, frame_prob) each shaped (T, 88)."""
    chunk   = 600
    overlap = 120
    stride  = chunk - overlap
    n_mels, T = spec.shape
    num_keys  = cfg.get("num_keys", 88)

    onset_acc  = np.zeros((T, num_keys), np.float32)
    frame_acc  = np.zeros((T, num_keys), np.float32)
    weight_acc = np.zeros((T, 1),        np.float32)

    starts = list(range(0, max(1, T - chunk + 1), stride))
    if not starts or starts[-1] != max(0, T - chunk):
        starts.append(max(0, T - chunk))

    with torch.no_grad():
        for s in starts:
            e  = min(T, s + chunk)
            ch = spec[:, s:e]
            if ch.shape[1] < chunk:
                ch = np.pad(ch, ((0, 0), (0, chunk - ch.shape[1])))
            x   = torch.from_numpy(ch).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
            out = model(x)
            op  = torch.sigmoid(out["onset"]).squeeze(0).cpu().numpy()  # (chunk, 88)
            fp  = torch.sigmoid(out["frame"]).squeeze(0).cpu().numpy()
            vl  = e - s
            blend = np.ones((vl, 1), np.float32)
            if overlap > 0 and vl > overlap:
                ramp = np.linspace(0.0, 1.0, overlap, dtype=np.float32).reshape(-1, 1)
                if s > 0:
                    blend[:overlap] = ramp
                if e < T:
                    blend[-overlap:] = ramp[::-1]
            onset_acc[s:e]  += op[:vl] * blend
            frame_acc[s:e]  += fp[:vl] * blend
            weight_acc[s:e] += blend

    w = np.maximum(weight_acc, 1e-8)
    return onset_acc / w, frame_acc / w


# ── note decoding ──────────────────────────────────────────────────────────────
_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def _midi_to_name(n: int) -> str:
    return f"{_NOTE_NAMES[n % 12]}{n // 12 - 1}"


def _decode_notes(onset_prob: np.ndarray, frame_prob: np.ndarray,
                  cfg: dict) -> list[dict]:
    onset_thr    = float(cfg.get("default_inference_onset_threshold", 0.5))
    frame_thr    = float(cfg.get("default_inference_frame_threshold", 0.4))
    fps          = float(cfg.get("roll_fps", 100))
    apply_gating = cfg.get("inference_apply_onset_frame_gating", True)

    onset_bin = onset_prob >= onset_thr
    frame_bin = frame_prob >= frame_thr

    notes = []
    T, num_keys = onset_bin.shape
    MIDI_A0 = 21

    # Refractory window: suppress re-triggering within this many frames of onset
    REFRACTORY = int(fps * 0.05)  # 50 ms

    for k in range(num_keys):
        pitch      = k + MIDI_A0
        active     = False
        note_start = 0

        def _emit(end_t: int) -> None:
            dur = (end_t - note_start) / fps
            if dur >= 0.05:
                notes.append({
                    "pitch": pitch, "midi_note": pitch,
                    "start": round(note_start / fps, 4),
                    "end":   round(end_t / fps, 4),
                    "note_name": _midi_to_name(pitch),
                })

        for t in range(T):
            is_onset = onset_bin[max(0, t - 1):t + 2, k].any()

            if not active:
                if is_onset:
                    active     = True
                    note_start = t
            else:
                # Re-strike: new onset after the refractory window → end + restart
                if is_onset and t >= note_start + REFRACTORY:
                    _emit(t)
                    note_start = t  # restart; active stays True
                elif apply_gating and not frame_bin[t, k]:
                    _emit(t)
                    active = False

        if active:
            _emit(T)

    notes.sort(key=lambda n: n["start"])
    return notes


# ── (kept for legacy compat – unused) ─────────────────────────────────────────
def _load_audio(path: str, start: float, end: float | None,
                sample_rate: int) -> np.ndarray:
    y, _ = librosa.load(
        path, sr=sample_rate, mono=True,
        offset=start,
        duration=None if end is None else end - start,
    )
    return y


def _compute_melspec_tensor(y: np.ndarray, cfg: dict) -> torch.Tensor:
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


# ── public API ─────────────────────────────────────────────────────────────────
def run_inference(audio_path: str,
                  start_time: float = 0.0,
                  end_time: float | None = None) -> dict:

    model, cfg, norm = _get_model()

    y, _ = librosa.load(
        audio_path, sr=cfg["sample_rate"], mono=True,
        offset=start_time,
        duration=None if end_time is None else end_time - start_time,
    )
    duration = len(y) / cfg["sample_rate"]

    spec = _compute_melspec(y, cfg, norm)   # (n_mels, T) numpy

    onset_prob, frame_prob = _run_chunked(model, spec, cfg)

    notes = _decode_notes(onset_prob, frame_prob, cfg)

    return {
        "duration":   round(duration, 3),
        "fps":        float(cfg["roll_fps"]),
        "n_frames":   int(spec.shape[1]),
        "piano_roll": frame_prob.tolist(),   # (T, 88) float heatmap
        "onset_roll": onset_prob.tolist(),   # (T, 88)
        "notes":      notes,
    }
