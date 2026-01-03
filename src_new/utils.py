"""
Utility functions for training piano transcription models.
"""

import torch
import matplotlib.pyplot as plt
import glob
import os


def get_next_run_folder():
    """Find the next available training run folder number."""
    existing_runs = glob.glob("training_run_*")
    if not existing_runs:
        run_number = 1
    else:
        numbers = []
        for run in existing_runs:
            try:
                num = int(run.split("_")[-1])
                numbers.append(num)
            except ValueError:
                continue
        run_number = max(numbers) + 1 if numbers else 1
    
    folder_name = f"training_run_{run_number:03d}"
    os.makedirs(folder_name, exist_ok=True)
    return folder_name


def compute_metrics(predictions, targets, threshold=0.5):
    """
    Compute precision, recall, and F1 score.
    
    Args:
        predictions: Model predictions (batch_size, seq_len, num_outputs)
        targets: Ground truth (batch_size, seq_len, num_outputs)
        threshold: Threshold for binary classification
    
    Returns:
        dict with precision, recall, and f1
    """
    pred_binary = (predictions > threshold).float()
    pred_flat = pred_binary.view(-1)
    target_flat = targets.view(-1)
    
    tp = (pred_flat * target_flat).sum().item()
    fp = (pred_flat * (1 - target_flat)).sum().item()
    fn = ((1 - pred_flat) * target_flat).sum().item()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {'precision': precision, 'recall': recall, 'f1': f1}


def visualize_predictions(model, dataloader, device, save_path):
    """
    Visualize model predictions vs ground truth.
    Uses the file specified in inference_config.py for consistency.
    """
    from inference import visualize_comparison, run_inference, preprocess_audio_to_spectrogram, extract_time_range
    from inference_config import INFERENCE_CONFIG
    from config import CONFIG
    from data_preparation import create_piano_roll_with_onsets_durations
    import os
    
    model.eval()
    
    # Use the same file as inference_config
    if not os.path.exists(INFERENCE_CONFIG['audio_path']):
        print(f"Warning: Audio file not found: {INFERENCE_CONFIG['audio_path']}")
        print("Skipping visualization")
        return
    
    # Load and process audio
    full_spectrogram = preprocess_audio_to_spectrogram(
        INFERENCE_CONFIG['audio_path'],
        CONFIG
    )
    
    # Load ground truth if available
    full_labels = None
    if INFERENCE_CONFIG['midi_path'] is not None and os.path.exists(INFERENCE_CONFIG['midi_path']):
        full_labels = create_piano_roll_with_onsets_durations(
            INFERENCE_CONFIG['midi_path'],
            fps=CONFIG['roll_fps'],
            onset_frames=CONFIG.get('onset_frames', 2)
        )
    
    # Extract the time range from inference_config
    start_time = INFERENCE_CONFIG['start_time']
    end_time = INFERENCE_CONFIG['end_time']
    
    spectrogram_snippet, labels_snippet = extract_time_range(
        full_spectrogram,
        full_labels,
        start_time,
        end_time,
        CONFIG['roll_fps'],
        CONFIG['hop_length'],
        CONFIG['sample_rate']
    )
    
    # Run inference
    predictions = run_inference(model, spectrogram_snippet, device, CONFIG)
    
    # Visualize using the same function as inference.py
    visualize_comparison(
        labels_snippet,
        predictions,
        save_path,
        start_time,
        end_time,
        CONFIG['roll_fps']
    )


def plot_training_curves(train_losses, val_losses, train_metrics, val_metrics, save_path):
    """
    Plot training and validation losses and metrics.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        train_metrics: List of training metrics dicts per epoch
        val_metrics: List of validation metrics dicts per epoch
        save_path: Path to save the plot
    """
    epochs = range(1, len(train_losses) + 1)
    
    # Extract metrics from dicts
    train_precision = [m['precision'] for m in train_metrics]
    train_recall = [m['recall'] for m in train_metrics]
    train_f1 = [m['f1'] for m in train_metrics]
    
    val_precision = [m['precision'] for m in val_metrics]
    val_recall = [m['recall'] for m in val_metrics]
    val_f1 = [m['f1'] for m in val_metrics]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Loss (LOGARITHMIC Y-AXIS)
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (log scale)')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_yscale('log')  # Set logarithmic y-axis
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Precision
    axes[0, 1].plot(epochs, train_precision, 'b-', label='Train Precision', linewidth=2)
    axes[0, 1].plot(epochs, val_precision, 'r-', label='Val Precision', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Frame-Level Precision')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    # Plot 3: Recall
    axes[1, 0].plot(epochs, train_recall, 'b-', label='Train Recall', linewidth=2)
    axes[1, 0].plot(epochs, val_recall, 'r-', label='Val Recall', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Recall')
    axes[1, 0].set_title('Frame-Level Recall')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])
    
    # Plot 4: F1 Score
    axes[1, 1].plot(epochs, train_f1, 'b-', label='Train F1', linewidth=2)
    axes[1, 1].plot(epochs, val_f1, 'r-', label='Val F1', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].set_title('Frame-Level F1 Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_checkpoint(epoch, model, optimizer, scheduler, train_losses, val_losses,
                   train_metrics, val_metrics, best_val_loss, save_path):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
    }
    torch.save(checkpoint, save_path)


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"Resuming from epoch {checkpoint['epoch'] + 1}")
    print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
    
    return checkpoint
