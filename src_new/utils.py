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
    """Visualize model predictions vs ground truth."""
    model.eval()
    spectrograms, piano_rolls = next(iter(dataloader))
    spectrograms = spectrograms.to(device)
    
    with torch.no_grad():
        predictions = model(spectrograms)
    
    pred = predictions[0].cpu().numpy()
    gt = piano_rolls[0].cpu().numpy()
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    axes[0].imshow(gt.T, aspect='auto', origin='lower', cmap='hot', interpolation='nearest')
    axes[0].set_title('Ground Truth')
    axes[0].set_xlabel('Time Frames')
    axes[0].set_ylabel('Keys + Pedals')
    
    axes[1].imshow(pred.T, aspect='auto', origin='lower', cmap='hot', interpolation='nearest')
    axes[1].set_title('Predictions')
    axes[1].set_xlabel('Time Frames')
    axes[1].set_ylabel('Keys + Pedals')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Visualization saved to {save_path}")


def plot_training_curves(train_losses, val_losses, train_metrics, val_metrics, save_path):
    """
    Plot training curves.
    
    Args:
        train_losses, val_losses: Lists of loss values
        train_metrics, val_metrics: Lists of dicts with 'precision', 'recall', 'f1'
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Extract metrics
    train_precisions = [m['precision'] for m in train_metrics]
    train_recalls = [m['recall'] for m in train_metrics]
    train_f1s = [m['f1'] for m in train_metrics]
    val_precisions = [m['precision'] for m in val_metrics]
    val_recalls = [m['recall'] for m in val_metrics]
    val_f1s = [m['f1'] for m in val_metrics]
    
    # Loss
    axes[0, 0].plot(train_losses, label='Train')
    axes[0, 0].plot(val_losses, label='Val')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Precision
    axes[0, 1].plot(train_precisions, label='Train')
    axes[0, 1].plot(val_precisions, label='Val')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Recall
    axes[1, 0].plot(train_recalls, label='Train')
    axes[1, 0].plot(val_recalls, label='Val')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Recall')
    axes[1, 0].set_title('Recall')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # F1
    axes[1, 1].plot(train_f1s, label='Train')
    axes[1, 1].plot(val_f1s, label='Val')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].set_title('F1 Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
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
