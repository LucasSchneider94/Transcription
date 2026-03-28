"""Utility functions for training piano transcription models."""

import glob
import json
import os

import matplotlib.pyplot as plt
import torch


def get_next_run_folder():
    """Find the next available training run folder number."""
    existing_runs = glob.glob("training_run_*")
    if not existing_runs:
        run_number = 1
    else:
        numbers = []
        for run in existing_runs:
            try:
                numbers.append(int(run.split("_")[-1]))
            except ValueError:
                continue
        run_number = max(numbers) + 1 if numbers else 1

    folder_name = f"training_run_{run_number:03d}"
    os.makedirs(folder_name, exist_ok=True)
    return folder_name


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4, sort_keys=True)


def load_json(path, default=None):
    if not os.path.exists(path):
        return {} if default is None else default
    with open(path, "r") as f:
        return json.load(f)


def load_run_config(run_folder):
    return load_json(os.path.join(run_folder, "config.json"), default={})


def compute_metrics(predictions, targets, threshold=0.5):
    """Compute precision, recall, and F1 score for binary tensors."""
    pred_binary = (predictions > threshold).float()
    pred_flat = pred_binary.view(-1)
    target_flat = targets.view(-1)

    tp = (pred_flat * target_flat).sum().item()
    fp = (pred_flat * (1 - target_flat)).sum().item()
    fn = ((1 - pred_flat) * target_flat).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def visualize_predictions(*args, **kwargs):
    """Superseded by save_epoch_visualization in train.py. No-op."""
    pass


def plot_training_curves(train_losses, val_losses, train_metrics, val_metrics, save_path):
    """Plot training/validation losses and the three new metric streams."""
    epochs = range(1, len(train_losses) + 1)

    def _get(metrics_list, key):
        return [m.get(key, 0.0) for m in metrics_list]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    axes[0, 0].plot(epochs, train_losses, "b-", label="Train", linewidth=2)
    axes[0, 0].plot(epochs, val_losses,   "r-", label="Val",   linewidth=2)
    axes[0, 0].set_title("Loss (log scale)")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Onset joint F1 (time + pitch)
    axes[0, 1].plot(epochs, _get(train_metrics, "onset_joint_f1"), "b-", label="Train", linewidth=2)
    axes[0, 1].plot(epochs, _get(val_metrics,   "onset_joint_f1"), "r-", label="Val",   linewidth=2)
    axes[0, 1].set_title("Onset Joint F1 (time + pitch)")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Onset time F1 (any pitch)
    axes[1, 0].plot(epochs, _get(train_metrics, "onset_time_f1"), "b-", label="Train", linewidth=2)
    axes[1, 0].plot(epochs, _get(val_metrics,   "onset_time_f1"), "r-", label="Val",   linewidth=2)
    axes[1, 0].set_title("Onset Time F1 (any pitch)")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Frame F1
    axes[1, 1].plot(epochs, _get(train_metrics, "frame_f1"), "b-", label="Train", linewidth=2)
    axes[1, 1].plot(epochs, _get(val_metrics,   "frame_f1"), "r-", label="Val",   linewidth=2)
    axes[1, 1].set_title("Frame F1")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_checkpoint(
    epoch,
    model,
    optimizer,
    scheduler,
    scaler,
    train_losses,
    val_losses,
    train_metrics,
    val_metrics,
    best_val_loss,
    save_path,
):
    """Save training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "best_val_loss": best_val_loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }
    torch.save(checkpoint, save_path)


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler, device):
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler and checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"Resuming from epoch {checkpoint['epoch'] + 1}")
    print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
    return checkpoint
