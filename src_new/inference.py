"""Inference script for piano transcription (onset + frame model).

Usage
-----
    python inference.py                        # uses INFERENCE_CONFIG defaults
    python inference.py --audio a.wav --model run_001/model.pth --out results/

Outputs
-------
  • Sigmoid probability maps (onset + frame) with optional GT comparison.
  • Decoded binary piano roll (thresholded, optional onset–frame gating).
"""

import argparse
import json
import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import CONFIG
from data_preparation import create_piano_roll_with_onsets_durations
from inference_config import INFERENCE_CONFIG
from model import PianoTranscriptionModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_run_config(model_path: str):
    """Load per-run config.json and normalization_stats.json."""
    model_dir = os.path.dirname(model_path)
    run_config = CONFIG.copy()
    cfg = os.path.join(model_dir, "config.json")
    if os.path.exists(cfg):
        with open(cfg) as f:
            run_config.update(json.load(f))
        print(f"Loaded run config: {cfg}")
    norm_stats = {"mode": "none"}
    nrm = os.path.join(model_dir, "normalization_stats.json")
    if os.path.exists(nrm):
        with open(nrm) as f:
            norm_stats = json.load(f)
    thr = os.path.join(model_dir, "best_threshold.json")
    if os.path.exists(thr):
        with open(thr) as f:
            d = json.load(f)
            # train.py saves {"onset_threshold": ...}; fall back to legacy key
            run_config["default_inference_onset_threshold"] = d.get(
                "onset_threshold", d.get("best_threshold",
                    run_config.get("default_inference_onset_threshold", 0.5))
            )
    run_config.setdefault("hop_length",
                          int(run_config["sample_rate"] / run_config["roll_fps"]))
    return run_config, norm_stats


def build_model(model_path: str, device: torch.device, run_config: dict):
    model = PianoTranscriptionModel(
        n_mels=run_config.get("n_mels", 352),
        num_keys=run_config.get("num_keys", 88),
        cnn_channels=tuple(run_config.get("cnn_channels", (32, 64, 128))),
        cnn_freq_kernels=tuple(run_config.get("cnn_freq_kernels", (87, 31, 15))),
        cnn_time_kernel=run_config.get("cnn_time_kernel", 9),
        cnn_freq_pool=tuple(run_config.get("cnn_freq_pool", (4, 2, 4))),
        transformer_dim=run_config.get("transformer_dim", 256),
        transformer_heads=run_config.get("transformer_heads", 8),
        transformer_layers=run_config.get("transformer_layers", 4),
        dropout=0.0,
    ).to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded model: {model_path}")
    return model


def preprocess_audio(audio_path: str, config: dict, norm_stats: dict) -> np.ndarray:
    """Return log-mel spectrogram (n_mels, T) with optional normalisation."""
    audio, _ = librosa.load(audio_path, sr=config["sample_rate"])
    mel  = librosa.feature.melspectrogram(
        y=audio, sr=config["sample_rate"],
        n_fft=config["n_fft"], hop_length=config["hop_length"], n_mels=config["n_mels"],
    )
    spec = librosa.power_to_db(mel, ref=np.max)
    if norm_stats.get("mode") == "global":
        mean = float(norm_stats.get("mean", 0.0))
        std  = float(norm_stats.get("std",  1.0))
        spec = (spec - mean) / max(std, 1e-8)
    return spec


def load_ground_truth(midi_path, config):
    if not midi_path or not os.path.exists(midi_path):
        return None
    return create_piano_roll_with_onsets_durations(
        midi_path, fps=config["roll_fps"], onset_frames=config.get("onset_frames", 2),
    )


def extract_time_range(spectrogram, labels, start_time, end_time, config):
    hop    = config["hop_length"]
    sr     = config["sample_rate"]
    fps    = config["roll_fps"]
    s_spec = int(start_time * sr / hop)
    e_spec = int(end_time   * sr / hop)
    spec   = spectrogram[:, s_spec:e_spec]
    if labels is None:
        return spec, None
    s_lbl = int(start_time * fps)
    e_lbl = int(end_time   * fps)
    return spec, {k: v[s_lbl:e_lbl] for k, v in labels.items()}


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def run_inference(model, spectrogram: np.ndarray, device, config: dict) -> dict:
    """Chunked overlap-add inference for arbitrary-length audio.

    Returns raw sigmoid probabilities and decoded binary piano rolls.
    A triangular blend in overlap regions suppresses chunk-boundary artefacts.
    """
    chunk   = int(config.get("inference_chunk_frames", 600))
    overlap = int(config.get("inference_overlap_frames", 120))
    overlap = min(overlap, chunk // 2)
    stride  = chunk - overlap

    n_mels, T = spectrogram.shape
    num_keys  = config.get("num_keys", 88)
    onset_acc  = np.zeros((T, num_keys), np.float32)
    frame_acc  = np.zeros((T, num_keys), np.float32)
    weight_acc = np.zeros((T, 1),        np.float32)

    starts = list(range(0, max(1, T - chunk + 1), stride))
    if not starts or starts[-1] != max(0, T - chunk):
        starts.append(max(0, T - chunk))

    with torch.no_grad():
        for s in starts:
            e  = min(T, s + chunk)
            ch = spectrogram[:, s:e]
            if ch.shape[1] < chunk:
                ch = np.pad(ch, ((0, 0), (0, chunk - ch.shape[1])))
            x   = torch.from_numpy(ch).float().unsqueeze(0).unsqueeze(0).to(device)
            out = model(x)
            op  = torch.sigmoid(out["onset"]).squeeze(0).cpu().numpy()
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
    onset_prob = onset_acc / w
    frame_prob = frame_acc / w

    onset_thr = float(config.get("default_inference_onset_threshold", 0.5))
    frame_thr = float(config.get("default_inference_frame_threshold", 0.4))
    onset_bin = (onset_prob >= onset_thr).astype(np.float32)
    frame_bin = (frame_prob >= frame_thr).astype(np.float32)

    # Optional onset-gated frame decoding
    if config.get("inference_apply_onset_frame_gating", True):
        gated = np.zeros_like(frame_bin)
        for k in range(num_keys):
            active = False
            for t in range(T):
                if onset_bin[max(0, t - 2):t + 1, k].max() > 0.5:
                    active = True
                if active and frame_bin[t, k] > 0.5:
                    gated[t, k] = 1.0
                elif active and frame_bin[t, k] <= 0.5:
                    active = False
        frame_bin = gated

    return {
        "onset":          onset_prob,
        "frame":          frame_prob,
        "onset_decoded":  onset_bin,
        "frame_decoded":  frame_bin,
        "onset_threshold": onset_thr,
        "frame_threshold": frame_thr,
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize_comparison(ground_truth, predictions, save_path: str,
                         start_time: float, end_time: float, fps: int = 100):
    """Plot sigmoid predictions alongside ground truth.

    With GT    – 2 rows × 3 cols:
      Row 0: GT onset | GT frame | empty
      Row 1: Pred onset sigmoid | Pred frame sigmoid | Decoded overlay

    Without GT – 1 row × 2 cols:
      Pred onset sigmoid | Pred frame sigmoid
    """
    onset_thr = predictions.get("onset_threshold", 0.5)
    frame_thr = predictions.get("frame_threshold", 0.5)
    onset_sig = predictions["onset"]
    frame_sig = predictions["frame"]
    onset_dec = predictions.get("onset_decoded",
                                (onset_sig >= onset_thr).astype(np.float32))
    frame_dec = predictions.get("frame_decoded",
                                (frame_sig >= frame_thr).astype(np.float32))

    if ground_truth is not None:
        fig, axes = plt.subplots(2, 3, figsize=(22, 10))
        fig.suptitle(
            f"Transcription  {start_time:.1f}s – {end_time:.1f}s",
            fontsize=14, fontweight="bold",
        )
        axes[0, 0].imshow(ground_truth["onset"].T, aspect="auto", origin="lower",
                          cmap="hot", vmin=0, vmax=1)
        axes[0, 0].set_title("GT: Onset"); axes[0, 0].set_ylabel("Piano key")

        axes[0, 1].imshow(ground_truth["frame"].T, aspect="auto", origin="lower",
                          cmap="hot", vmin=0, vmax=1)
        axes[0, 1].set_title("GT: Frame")
        axes[0, 2].set_visible(False)

        im1 = axes[1, 0].imshow(onset_sig.T, aspect="auto", origin="lower",
                                 cmap="hot", vmin=0, vmax=1)
        axes[1, 0].set_title("Pred: Onset sigmoid")
        axes[1, 0].set_ylabel("Piano key"); axes[1, 0].set_xlabel("Time frame")
        plt.colorbar(im1, ax=axes[1, 0])

        im2 = axes[1, 1].imshow(frame_sig.T, aspect="auto", origin="lower",
                                 cmap="hot", vmin=0, vmax=1)
        axes[1, 1].set_title("Pred: Frame sigmoid")
        axes[1, 1].set_xlabel("Time frame")
        plt.colorbar(im2, ax=axes[1, 1])

        # Overlay: onset bright, frame dim
        axes[1, 2].imshow(onset_dec.T + frame_dec.T * 0.5, aspect="auto",
                          origin="lower", cmap="hot", vmin=0, vmax=1.5)
        axes[1, 2].set_title(
            f"Decoded  onset thr={onset_thr:.2f}  frame thr={frame_thr:.2f}"
        )
        axes[1, 2].set_xlabel("Time frame")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Predictions  {start_time:.1f}s – {end_time:.1f}s",
                     fontsize=13, fontweight="bold")
        im1 = axes[0].imshow(onset_sig.T, aspect="auto", origin="lower",
                              cmap="hot", vmin=0, vmax=1)
        axes[0].set_title("Onset sigmoid"); axes[0].set_ylabel("Piano key")
        plt.colorbar(im1, ax=axes[0])
        im2 = axes[1].imshow(frame_sig.T, aspect="auto", origin="lower",
                              cmap="hot", vmin=0, vmax=1)
        axes[1].set_title("Frame sigmoid")
        plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Piano transcription inference")
    parser.add_argument("--audio",  default=None)
    parser.add_argument("--midi",   default=None)
    parser.add_argument("--model",  default=None)
    parser.add_argument("--start",  type=float, default=None)
    parser.add_argument("--end",    type=float, default=None)
    parser.add_argument("--out",    default=None)
    args = parser.parse_args()

    audio_path = args.audio or INFERENCE_CONFIG.get("audio_path", "")
    midi_path  = args.midi  or INFERENCE_CONFIG.get("midi_path")
    model_path = args.model or INFERENCE_CONFIG.get("model_path", "")
    start_time = args.start if args.start is not None else INFERENCE_CONFIG.get("start_time", 0.0)
    end_time   = args.end   if args.end   is not None else INFERENCE_CONFIG.get("end_time",  20.0)
    out_dir    = args.out   or INFERENCE_CONFIG.get("output_dir", "inference_results")

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio not found: {audio_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    run_config, norm_stats = load_run_config(model_path)
    model  = build_model(model_path, device, run_config)
    spec   = preprocess_audio(audio_path, run_config, norm_stats)
    labels = load_ground_truth(midi_path, run_config)
    spec_snip, labels_snip = extract_time_range(spec, labels, start_time, end_time, run_config)
    preds  = run_inference(model, spec_snip, device, run_config)

    base     = os.path.splitext(os.path.basename(audio_path))[0]
    vis_path = os.path.join(out_dir, f"{base}_{int(start_time)}s_{int(end_time)}s.png")
    visualize_comparison(labels_snip, preds, vis_path, start_time, end_time,
                         fps=run_config["roll_fps"])


if __name__ == "__main__":
    main()

