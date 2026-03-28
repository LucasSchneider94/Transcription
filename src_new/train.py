"""Training script for piano transcription model.

Usage:
    python train.py            # full training with config.py
    python train.py --overfit  # overfit sanity-check with config_overfit.py
"""

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import maximum_filter1d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import CONFIG
from config_overfit import CONFIG_OVERFIT
from dataset import PianoTranscriptionDataset
from model import PianoTranscriptionModel
from inference import (
    extract_time_range,
    load_ground_truth,
    preprocess_audio,
    run_inference,
    visualize_comparison,
)
from inference_config import INFERENCE_CONFIG
from utils import (
    get_next_run_folder,
    load_json,
    plot_training_curves,
    save_checkpoint,
    save_json,
)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Binary focal loss for class-imbalanced binary prediction."""

    def __init__(self, alpha: float = 0.9, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        p_t = torch.sigmoid(logits) * targets + (1 - torch.sigmoid(logits)) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()


class OnsetFrameLoss(nn.Module):
    """Weighted sum of onset and frame losses."""

    def __init__(self, config: dict, onset_pos_weight=None, frame_pos_weight=None):
        super().__init__()
        self.onset_w = config.get("onset_weight", 3.0)
        self.frame_w = config.get("frame_weight", 1.0)

        if config.get("onset_loss_type", "focal") == "focal":
            self.onset_fn = FocalLoss(
                alpha=config.get("onset_focal_alpha", 0.9),
                gamma=config.get("onset_focal_gamma", 2.0),
            )
        else:
            self.onset_fn = nn.BCEWithLogitsLoss(pos_weight=onset_pos_weight)

        if config.get("frame_loss_type", "focal") == "focal":
            self.frame_fn = FocalLoss(
                alpha=config.get("frame_focal_alpha", 0.5),
                gamma=config.get("frame_focal_gamma", 2.0),
            )
        else:
            self.frame_fn = nn.BCEWithLogitsLoss(pos_weight=frame_pos_weight)

    def forward(self, preds: dict, onset_gt: torch.Tensor, frame_gt: torch.Tensor):
        onset_loss = self.onset_fn(preds["onset"], onset_gt)
        frame_loss = self.frame_fn(preds["frame"], frame_gt)
        total = self.onset_w * onset_loss + self.frame_w * frame_loss
        return total, {
            "onset_loss": onset_loss.item(),
            "frame_loss": frame_loss.item(),
            "total_loss": total.item(),
        }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _dilate(arr: np.ndarray, half_width: int) -> np.ndarray:
    """Max-dilation along axis 0 (time) via maximum_filter1d."""
    return maximum_filter1d(arr, size=2 * half_width + 1, axis=0)


def compute_metrics(
    onset_probs: np.ndarray,  # (T, 88)
    onset_gt: np.ndarray,     # (T, 88)
    frame_probs: np.ndarray,  # (T, 88)
    frame_gt: np.ndarray,     # (T, 88)
    threshold: float = 0.5,
    tol_frames: int = 5,
) -> dict:
    """Compute three metric streams:

    onset_joint  — correct pitch AND timing within ±tol_frames
    onset_time   — correct timing regardless of which key fired
    frame        — frame-level binary F1
    """
    onset_pred = (onset_probs >= threshold).astype(np.float32)
    frame_pred = (frame_probs >= threshold).astype(np.float32)
    onset_gt = onset_gt.astype(np.float32)
    frame_gt = frame_gt.astype(np.float32)

    # --- onset_joint ---
    gt_dil   = _dilate(onset_gt,   tol_frames)   # GT expanded in time
    pred_dil = _dilate(onset_pred, tol_frames)   # pred expanded for recall direction
    tp_p = (onset_pred * gt_dil).sum()
    fp_j = onset_pred.sum() - tp_p
    tp_r = (onset_gt * pred_dil).sum()
    fn_j = onset_gt.sum() - tp_r
    prec_j = tp_p / (tp_p + fp_j + 1e-8)
    rec_j  = tp_r / (tp_r + fn_j + 1e-8)
    f1_j   = 2 * prec_j * rec_j / (prec_j + rec_j + 1e-8)

    # --- onset_time ---
    pred_any = onset_pred.any(axis=1).astype(np.float32)
    gt_any   = onset_gt.any(axis=1).astype(np.float32)
    gt_any_d   = _dilate(gt_any[:, None],   tol_frames)[:, 0]
    pred_any_d = _dilate(pred_any[:, None], tol_frames)[:, 0]
    tp_tp = (pred_any * gt_any_d).sum()
    fp_t  = pred_any.sum() - tp_tp
    tp_tr = (gt_any * pred_any_d).sum()
    fn_t  = gt_any.sum() - tp_tr
    prec_t = tp_tp / (tp_tp + fp_t + 1e-8)
    rec_t  = tp_tr / (tp_tr + fn_t + 1e-8)
    f1_t   = 2 * prec_t * rec_t / (prec_t + rec_t + 1e-8)

    # --- frame ---
    tp_f = (frame_pred * frame_gt).sum()
    fp_f = frame_pred.sum() - tp_f
    fn_f = frame_gt.sum() - tp_f
    prec_f = tp_f / (tp_f + fp_f + 1e-8)
    rec_f  = tp_f / (tp_f + fn_f + 1e-8)
    f1_f   = 2 * prec_f * rec_f / (prec_f + rec_f + 1e-8)

    return {
        "onset_joint_precision": float(prec_j),
        "onset_joint_recall":    float(rec_j),
        "onset_joint_f1":        float(f1_j),
        "onset_time_precision":  float(prec_t),
        "onset_time_recall":     float(rec_t),
        "onset_time_f1":         float(f1_t),
        "frame_precision":       float(prec_f),
        "frame_recall":          float(rec_f),
        "frame_f1":              float(f1_f),
    }


def sweep_thresholds(
    onset_probs: np.ndarray,
    onset_gt: np.ndarray,
    frame_probs: np.ndarray,
    frame_gt: np.ndarray,
    thresholds: list,
    tol_frames: int = 5,
) -> dict:
    """Sweep thresholds and return the metrics dict with best onset_joint_f1."""
    best = None
    for thr in thresholds:
        m = compute_metrics(onset_probs, onset_gt, frame_probs, frame_gt, thr, tol_frames)
        m["threshold"] = thr
        if best is None or m["onset_joint_f1"] > best["onset_joint_f1"]:
            best = m
    return best or {}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _compute_pos_weights(dataset: PianoTranscriptionDataset, max_batches: int = 20):
    """Estimate onset/frame positive-class weights from a small data sample."""
    onset_pos = onset_neg = frame_pos = frame_neg = 1e-8  # avoid div-by-zero
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        o = batch["onset"].numpy()
        f = batch["frame"].numpy()
        onset_pos += o.sum()
        onset_neg += (1 - o).sum()
        frame_pos += f.sum()
        frame_neg += (1 - f).sum()
    return onset_neg / onset_pos, frame_neg / frame_pos


def build_file_splits(config: dict):
    """Return (train_npz_paths, val_npz_paths) using the MAESTRO split field."""
    json_path = config["maestro_json"]
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"MAESTRO JSON not found: {json_path}")

    with open(json_path) as fh:
        maestro = json.load(fh)

    # MAESTRO v3 JSON is a dict-of-dicts keyed by string integers
    af_map  = maestro["audio_filename"]   # {"0": "2018/...", ...}
    spl_map = maestro["split"]            # {"0": "train", ...}

    train_stems, val_stems = set(), set()
    for k in af_map:
        stem = os.path.splitext(os.path.basename(af_map[k]))[0].lower()
        if spl_map[k] == "train":
            train_stems.add(stem)
        elif spl_map[k] == "validation":
            val_stems.add(stem)

    data_dir = config["data_dir"]
    all_npz = sorted(
        f for f in os.listdir(data_dir) if f.endswith("_piano_roll_with_pedals.npz")
    )
    train_paths, val_paths = [], []
    for npz in all_npz:
        stem = npz.replace("_piano_roll_with_pedals.npz", "").lower()
        path = os.path.join(data_dir, npz)
        if stem in train_stems:
            train_paths.append(path)
        elif stem in val_stems:
            val_paths.append(path)

    return train_paths, val_paths


def _compute_norm_stats(paths: list) -> dict:
    """Global mean/std over up to 50 spectrograms (fast approximate)."""
    s = s2 = n = 0.0
    for p in paths[:50]:
        spec = np.load(p)["spectrogram"].astype(np.float64)
        s  += spec.sum()
        s2 += (spec ** 2).sum()
        n  += spec.size
    mean = s / n if n > 0 else 0.0
    std  = float(np.sqrt(max(s2 / n - mean ** 2, 1e-8))) if n > 0 else 1.0
    return {"mode": "global", "mean": float(mean), "std": std}


def create_dataloaders(config: dict):
    """Build train and val DataLoaders. Returns (train_loader, val_loader, norm_stats)."""
    data_dir = config["data_dir"]
    overfit = config.get("overfit_mode", False)

    if overfit:
        n = config.get("overfit_num_files", 2)
        all_npz = sorted(
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith("_piano_roll_with_pedals.npz")
        )
        train_paths = all_npz[:n]
        val_paths   = all_npz[:n]
    else:
        train_paths, val_paths = build_file_splits(config)
        frac = config.get("data_fraction", 1.0)
        if frac < 1.0:
            train_paths = train_paths[: max(1, int(len(train_paths) * frac))]
            val_paths   = val_paths[: max(1, int(len(val_paths)   * frac))]

    if not train_paths:
        raise RuntimeError(f"No training files found in {data_dir}")

    norm_stats = _compute_norm_stats(train_paths)

    common = dict(
        data_dir=data_dir,
        snippet_frames=config["snippet_frames"],
        snippets_per_file=config["snippets_per_file"],
        duration_mode=config.get("duration_mode", "log"),
        sampling_mode=config.get("sampling_mode", "random"),
        onset_sampling_min_active_ratio=config.get("onset_sampling_min_active_ratio", 0.5),
        normalization_stats=norm_stats,
    )

    train_ds = PianoTranscriptionDataset(
        file_list=[os.path.basename(p) for p in train_paths],
        file_indices=list(range(len(train_paths))),
        seed=config.get("deterministic_seed", 42),
        fixed_snippets=config.get("fixed_snippets", False),
        **common,
    )
    val_ds = PianoTranscriptionDataset(
        file_list=[os.path.basename(p) for p in val_paths],
        file_indices=list(range(len(val_paths))),
        seed=config.get("deterministic_seed", 42) + 1,
        fixed_snippets=True,   # validation is always deterministic
        **common,
    )

    nw = config.get("num_workers", 0)
    pin = config.get("pin_memory", False)
    persist = nw > 0 and config.get("persistent_workers", False)

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=config.get("shuffle_train", True),
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=persist,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=persist,
    )
    return train_loader, val_loader, norm_stats


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _pick_viz_batch(val_loader: DataLoader) -> dict:
    """Return the first val batch that contains at least one onset event."""
    for batch in val_loader:
        if batch["onset"].sum() > 0:
            return batch
    return next(iter(val_loader))


def save_epoch_visualization(
    model: nn.Module,
    viz_batch: dict,
    device: torch.device,
    save_path: str,
    epoch: int,
):
    """2×2 grid: GT onset | GT frame | pred onset sigmoid | pred frame sigmoid."""
    model.eval()
    with torch.no_grad():
        spec  = viz_batch["spectrogram"].to(device).unsqueeze(1)
        preds = model(spec)
        onset_sig = torch.sigmoid(preds["onset"]).cpu().numpy()
        frame_sig = torch.sigmoid(preds["frame"]).cpu().numpy()

    gt_onset = viz_batch["onset"].numpy()
    gt_frame = viz_batch["frame"].numpy()

    # First sample in batch, transposed to (keys, time) for imshow
    go = gt_onset[0].T
    gf = gt_frame[0].T
    po = onset_sig[0].T
    pf = frame_sig[0].T

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    fig.suptitle(f"Epoch {epoch}", fontsize=12)
    for ax, data, title in [
        (axes[0, 0], go, "GT Onset"),
        (axes[0, 1], gf, "GT Frame"),
        (axes[1, 0], po, "Pred Onset (sigmoid)"),
        (axes[1, 1], pf, "Pred Frame (sigmoid)"),
    ]:
        ax.imshow(data, aspect="auto", origin="lower", vmin=0, vmax=1, cmap="inferno")
        ax.set_title(title)
        ax.set_xlabel("Time frames")
        ax.set_ylabel("Piano key")
    plt.savefig(save_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Probe inference on a fixed audio file
# ---------------------------------------------------------------------------

def _setup_probe(config: dict, norm_stats: dict, script_dir: str):
    """Pre-process the fixed probe audio once at training start.

    Paths in inference_config.py are relative to the project root
    (one level above src_new).  Returns a dict ready for _save_probe_inference,
    or None if the audio file is not found.
    """
    audio_path = INFERENCE_CONFIG.get("audio_path", "")
    midi_path  = INFERENCE_CONFIG.get("midi_path")
    start_time = float(INFERENCE_CONFIG.get("start_time", 0.0))
    end_time   = float(INFERENCE_CONFIG.get("end_time",  20.0))

    def _abs(p):
        if not p:
            return p
        return p if os.path.isabs(p) else os.path.normpath(
            os.path.join(script_dir, "..", p)
        )

    audio_path = _abs(audio_path)
    midi_path  = _abs(midi_path)

    if not audio_path or not os.path.exists(audio_path):
        print(f"[probe] Audio not found, skipping file-based snapshots: {audio_path}")
        return None

    print(f"[probe] Pre-processing: {os.path.basename(audio_path)}  "
          f"{start_time:.0f}s – {end_time:.0f}s")
    spec   = preprocess_audio(audio_path, config, norm_stats)
    labels = load_ground_truth(midi_path, config)
    spec_s, labels_s = extract_time_range(spec, labels, start_time, end_time, config)
    return {
        "spec":       spec_s,
        "labels":     labels_s,
        "start_time": start_time,
        "end_time":   end_time,
    }


def _save_probe_inference(
    model: nn.Module,
    probe: dict,
    config: dict,
    device: torch.device,
    save_path: str,
    epoch: int,
    writer,
):
    """Run the live model on the cached probe snippet and save a comparison PNG."""
    model.eval()
    with torch.no_grad():
        preds = run_inference(model, probe["spec"], device, config)
    visualize_comparison(
        probe["labels"], preds, save_path,
        probe["start_time"], probe["end_time"],
        fps=config["roll_fps"],
    )
    # Push image into TensorBoard (read back the saved PNG)
    img = plt.imread(save_path)            # (H, W, 3 or 4)
    writer.add_image("Probe/inference", img[:, :, :3], global_step=epoch,
                     dataformats="HWC")


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(config: dict, device: torch.device) -> PianoTranscriptionModel:
    model = PianoTranscriptionModel(
        n_mels=config["n_mels"],
        num_keys=config["num_keys"],
        cnn_channels=config["cnn_channels"],
        cnn_freq_kernels=config["cnn_freq_kernels"],
        cnn_time_kernel=config["cnn_time_kernel"],
        cnn_freq_pool=config["cnn_freq_pool"],
        transformer_dim=config["transformer_dim"],
        transformer_heads=config["transformer_heads"],
        transformer_layers=config["transformer_layers"],
        dropout=config.get("dropout", 0.1),
    )
    return model.to(device)


# ---------------------------------------------------------------------------
# Epoch runner
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: OnsetFrameLoss,
    optimizer,
    scaler,
    device: torch.device,
    config: dict,
    is_train: bool = True,
) -> tuple:
    """Run one epoch. Returns (mean_loss, metrics_dict)."""
    model.train(is_train)
    use_amp = config.get("use_amp", False) and device.type == "cuda"
    max_collect = config.get("max_collect_batches", 32)

    total_loss = 0.0
    n_batches  = 0
    onset_p_list, onset_g_list = [], []
    frame_p_list, frame_g_list = [], []

    for batch in loader:
        spec  = batch["spectrogram"].to(device).unsqueeze(1)  # (B,1,F,T)
        onset = batch["onset"].to(device)                     # (B,T,88)
        frame = batch["frame"].to(device)                     # (B,T,88)

        with torch.set_grad_enabled(is_train):
            with torch.autocast(device_type=device.type, enabled=use_amp):
                preds = model(spec)
                loss, _ = loss_fn(preds, onset, frame)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.get("grad_clip_norm", 1.0)
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.get("grad_clip_norm", 1.0)
                )
                optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

        if n_batches <= max_collect:
            onset_p_list.append(torch.sigmoid(preds["onset"]).detach().cpu().numpy())
            onset_g_list.append(onset.detach().cpu().numpy())
            frame_p_list.append(torch.sigmoid(preds["frame"]).detach().cpu().numpy())
            frame_g_list.append(frame.detach().cpu().numpy())

    mean_loss = total_loss / max(n_batches, 1)

    if onset_p_list:
        K = config["num_keys"]
        op = np.concatenate(onset_p_list).reshape(-1, K)
        og = np.concatenate(onset_g_list).reshape(-1, K)
        fp_ = np.concatenate(frame_p_list).reshape(-1, K)
        fg  = np.concatenate(frame_g_list).reshape(-1, K)
        metrics = sweep_thresholds(
            op, og, fp_, fg,
            thresholds=config.get("threshold_sweep_values", [0.3, 0.5]),
            tol_frames=config.get("onset_tolerance_frames", 5),
        )
    else:
        metrics = {}

    return mean_loss, metrics


# ---------------------------------------------------------------------------
# Config resolver
# ---------------------------------------------------------------------------

def get_active_config(use_overfit: bool) -> dict:
    """Return a resolved config dict with absolute paths and derived fields."""
    cfg = dict(CONFIG_OVERFIT if use_overfit else CONFIG)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for key in ("data_dir", "maestro_json"):
        val = cfg.get(key, "")
        if val and not os.path.isabs(val):
            cfg[key] = os.path.normpath(os.path.join(script_dir, val))
    cfg["hop_length"]     = int(cfg["sample_rate"] / cfg["roll_fps"])
    cfg["snippet_frames"] = int(cfg["snippet_duration"] * cfg["roll_fps"])
    return cfg


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(config: dict):
    # Seed
    seed = config.get("deterministic_seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device: {device}")

    # Run folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    run_dir = get_next_run_folder()
    print(f"Run folder: {run_dir}")
    save_json(config, os.path.join(run_dir, "config.json"))

    # Data
    print("Building dataloaders...")
    train_loader, val_loader, norm_stats = create_dataloaders(config)
    save_json(norm_stats, os.path.join(run_dir, "normalization_stats.json"))
    print(f"  train batches: {len(train_loader)},  val batches: {len(val_loader)}")

    # Positive weights (only needed for BCE mode)
    onset_pw = frame_pw = None
    if config.get("use_computed_pos_weight", False) and (
        config.get("onset_loss_type") == "bce" or config.get("frame_loss_type") == "bce"
    ):
        print("Computing positive weights...")
        ow, fw = _compute_pos_weights(train_loader.dataset)
        max_pw = config.get("max_pos_weight", 1000.0)
        ow = min(ow, max_pw)
        fw = min(fw, max_pw)
        onset_pw = torch.tensor(ow, dtype=torch.float32).to(device)
        frame_pw = torch.tensor(fw, dtype=torch.float32).to(device)
        print(f"  onset_pos_weight={ow:.1f},  frame_pos_weight={fw:.1f}")

    # Model
    model = build_model(config, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    loss_fn   = OnsetFrameLoss(config, onset_pos_weight=onset_pw, frame_pos_weight=frame_pw)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 0.0),
    )

    scheduler = None
    if config.get("use_lr_scheduler", False):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.get("scheduler_factor", 0.5),
            patience=config.get("scheduler_patience", 8),
            min_lr=config.get("scheduler_min_lr", 1e-7),
        )

    scaler = None
    if config.get("use_amp", False) and device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()

    # Visualization setup
    enable_viz = config.get("enable_visualization", True)
    viz_batch  = _pick_viz_batch(val_loader) if enable_viz else None
    viz_dir    = os.path.join(run_dir, "visualizations")
    if viz_batch is not None:
        os.makedirs(viz_dir, exist_ok=True)

    # Probe: fixed audio file inference snapshots
    probe     = _setup_probe(config, norm_stats, script_dir) if enable_viz else None
    probe_dir = os.path.join(run_dir, "probe_inference")
    if probe is not None:
        os.makedirs(probe_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(run_dir, "tensorboard"))
    print(f"TensorBoard: tensorboard --logdir {os.path.join(run_dir, 'tensorboard')}")

    train_losses, val_losses          = [], []
    train_metrics_hist, val_metrics_hist = [], []
    best_val_loss  = float("inf")
    best_threshold = config.get("default_inference_onset_threshold", 0.5)
    num_epochs     = config["num_epochs"]
    viz_every      = config.get("visualize_every_n_epochs", 5)
    warmup         = config.get("warmup_epochs", 0)

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        # Linear LR warmup
        if warmup > 0 and epoch <= warmup:
            lr_scale = epoch / warmup
            for pg in optimizer.param_groups:
                pg["lr"] = config["learning_rate"] * lr_scale

        train_loss, train_m = run_epoch(
            model, train_loader, loss_fn, optimizer, scaler, device, config, is_train=True
        )
        val_loss, val_m = run_epoch(
            model, val_loader, loss_fn, optimizer, scaler, device, config, is_train=False
        )

        if scheduler is not None and epoch > warmup:
            scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_metrics_hist.append(train_m)
        val_metrics_hist.append(val_m)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:4d}/{num_epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"onset_j={val_m.get('onset_joint_f1', 0):.3f}  "
            f"onset_t={val_m.get('onset_time_f1', 0):.3f}  "
            f"frame={val_m.get('frame_f1', 0):.3f}  "
            f"thr={val_m.get('threshold', 0):.2f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  "
            f"({elapsed:.1f}s)"
        )

        # TensorBoard — losses, val metrics, and train metrics
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val",   val_loss,   epoch)
        for k, v in val_m.items():
            if isinstance(v, float):
                writer.add_scalar(f"Metrics/val/{k}", v, epoch)
        for k, v in train_m.items():
            if isinstance(v, float):
                writer.add_scalar(f"Metrics/train/{k}", v, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_threshold = val_m.get("threshold", 0.5)
            save_checkpoint(
                epoch, model, optimizer, scheduler, scaler,
                train_losses, val_losses, train_metrics_hist, val_metrics_hist,
                best_val_loss, os.path.join(run_dir, "best_model.pt"),
            )
            save_json(
                {"onset_threshold": best_threshold},
                os.path.join(run_dir, "best_threshold.json"),
            )

        # Periodic visualisation
        if enable_viz and epoch % viz_every == 0:
            # 1. Val-batch sigmoid heatmap (fast, no audio I/O)
            if viz_batch is not None:
                viz_path = os.path.join(viz_dir, f"vis_epoch_{epoch:04d}.png")
                save_epoch_visualization(model, viz_batch, device, viz_path, epoch)
                img = plt.imread(viz_path)
                writer.add_image("ValBatch/sigmoid", img[:, :, :3],
                                 global_step=epoch, dataformats="HWC")
            # 2. Probe inference on fixed audio file (richer, real pipeline)
            if probe is not None:
                probe_path = os.path.join(probe_dir, f"probe_epoch_{epoch:04d}.png")
                _save_probe_inference(model, probe, config, device,
                                      probe_path, epoch, writer)
            model.train()

        # Periodically flush stats to JSON (useful for long training runs)
        if epoch % 10 == 0 or epoch == num_epochs:
            save_json(
                {
                    "train_losses":   train_losses,
                    "val_losses":     val_losses,
                    "train_metrics":  train_metrics_hist,
                    "val_metrics":    val_metrics_hist,
                },
                os.path.join(run_dir, "training_stats.json"),
            )

    # Final artefacts
    save_checkpoint(
        num_epochs, model, optimizer, scheduler, scaler,
        train_losses, val_losses, train_metrics_hist, val_metrics_hist,
        best_val_loss, os.path.join(run_dir, "last_model.pt"),
    )
    plot_training_curves(
        train_losses, val_losses, train_metrics_hist, val_metrics_hist,
        os.path.join(run_dir, "training_curves.png"),
    )
    writer.close()

    print(f"\nDone. Best val loss: {best_val_loss:.4f}  Best threshold: {best_threshold:.2f}")
    print(f"Results saved to: {run_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train piano transcription model")
    parser.add_argument("--overfit", action="store_true", help="Use overfit config")
    args = parser.parse_args()
    train(get_active_config(args.overfit))
