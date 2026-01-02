"""
Training script for piano transcription model.
Supports onset + duration prediction with multiple modes (bins/log/linear).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import json
from tqdm import tqdm

from config import CONFIG, NUM_OUTPUTS
from model import PianoTranscriptionModel, PianoTranscriptionModelCNNOnly
from dataset import PianoTranscriptionDataset, duration_to_log_duration, log_duration_to_duration
from utils import (
    get_next_run_folder,
    compute_metrics,
    visualize_predictions,
    plot_training_curves,
    save_checkpoint,
    load_checkpoint
)


class OnsetDurationLoss(nn.Module):
    """
    Multi-task loss for onset, duration, and frame prediction.
    Supports three duration modes:
    - 'bins': Classification with cross-entropy loss
    - 'log': Log-duration regression with MSE loss
    - 'linear': Linear duration regression with MSE loss
    """
    def __init__(self, onset_weight=4.0, duration_weight=2.0, frame_weight=1.0, 
                 duration_mode='log'):
        super(OnsetDurationLoss, self).__init__()
        self.onset_weight = onset_weight
        self.duration_weight = duration_weight
        self.frame_weight = frame_weight
        self.duration_mode = duration_mode
        
        # Loss functions
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.ce = nn.CrossEntropyLoss(reduction='none')  # Use 'none' for masking
        self.mse = nn.MSELoss(reduction='none')  # Use 'none' for masking
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Dict with 'onset', 'duration', 'frame' outputs
                - onset: (batch, time, 88) - logits
                - duration: (batch, time, 88, num_bins) for 'bins' mode
                           or (batch, time, 88) for 'log'/'linear' modes - logits/values
                - frame: (batch, time, 88) - logits
            targets: Dict with 'onset', 'duration', 'frame' labels
                - onset: (batch, time, 88) - binary
                - duration: (batch, time, 88) - bin indices for 'bins' mode
                           or log/linear duration values for 'log'/'linear' modes
                - frame: (batch, time, 88) - binary
        
        Returns:
            Total weighted loss and individual losses
        """
        # Onset loss (BCE)
        onset_loss = self.bce(predictions['onset'], targets['onset'])
        
        # Duration loss (masked - only compute where onsets occur)
        onset_mask = targets['onset'] > 0.5  # (batch, time, 88)
        
        if self.duration_mode == 'bins':
            # Classification: cross-entropy loss
            batch_size, time_steps, num_keys, num_bins = predictions['duration'].shape
            
            # Reshape for cross-entropy: (batch*time*keys, num_bins)
            duration_pred_flat = predictions['duration'].reshape(-1, num_bins)
            duration_target_flat = targets['duration'].reshape(-1).long()
            onset_mask_flat = onset_mask.reshape(-1)
            
            # Compute CE loss for all positions
            ce_loss = self.ce(duration_pred_flat, duration_target_flat)
            
            # Apply mask and take mean over onset positions only
            if onset_mask_flat.sum() > 0:
                duration_loss = ce_loss[onset_mask_flat].mean()
            else:
                duration_loss = torch.tensor(0.0, device=predictions['onset'].device)
        
        else:
            # Regression: MSE loss (log or linear)
            mse_loss = self.mse(predictions['duration'], targets['duration'])
            
            # Apply mask and take mean over onset positions only
            if onset_mask.sum() > 0:
                duration_loss = mse_loss[onset_mask].mean()
            else:
                duration_loss = torch.tensor(0.0, device=predictions['onset'].device)
        
        # Frame loss (BCE) - for consistency/auxiliary task
        frame_loss = self.bce(predictions['frame'], targets['frame'])
        
        # Total loss
        total_loss = (self.onset_weight * onset_loss + 
                     self.duration_weight * duration_loss + 
                     self.frame_weight * frame_loss)
        
        return total_loss, {
            'onset_loss': onset_loss.item(),
            'duration_loss': duration_loss.item(),
            'frame_loss': frame_loss.item(),
            'total_loss': total_loss.item()
        }


def save_config_to_run_folder(config, run_folder):
    """Save the configuration to the run folder as JSON."""
    config_path = os.path.join(run_folder, "config.json")
    
    # Create a serializable copy of config
    config_dict = {}
    for key, value in config.items():
        # Convert non-serializable types to strings
        if isinstance(value, (int, float, str, bool, list, dict, type(None))):
            config_dict[key] = value
        else:
            config_dict[key] = str(value)
    
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=4, sort_keys=True)
    
    print(f"Config saved to: {config_path}")


def create_datasets_and_loaders(data_dir, config):
    """Create train and validation datasets and dataloaders."""
    # Get all files
    all_files = sorted([f for f in os.listdir(data_dir) 
                       if f.endswith("_piano_roll_with_pedals.npz")])
    num_files = len(all_files)
    print(f"Total files found: {num_files}")
    
    # Apply data fraction if specified
    data_fraction = config.get('data_fraction', 1.0)
    if data_fraction < 1.0:
        num_files_to_use = max(1, int(num_files * data_fraction))
        print(f"Using {data_fraction*100:.1f}% of data ({num_files_to_use}/{num_files} files)")
        np.random.seed(42)
        selected_indices = np.random.permutation(num_files)[:num_files_to_use]
        selected_indices = sorted(selected_indices)
        all_files = [all_files[i] for i in selected_indices]
        num_files = len(all_files)
        np.random.seed(None)
    
    # Split files into train/val (80/20)
    train_count = int(0.8 * num_files)
    val_count = num_files - train_count
    
    np.random.seed(42)
    file_indices = np.random.permutation(num_files)
    train_indices = file_indices[:train_count].tolist()
    val_indices = file_indices[train_count:].tolist()
    np.random.seed(None)
    
    print(f"\nFile-based split (prevents data leakage):")
    print(f"  Train: {train_count} files ({train_count/num_files*100:.1f}%)")
    print(f"  Val:   {val_count} files ({val_count/num_files*100:.1f}%)")
    
    # Create datasets with duration_mode from config
    train_dataset = PianoTranscriptionDataset(
        data_dir,
        config['snippet_frames'],
        snippets_per_file=config.get('snippets_per_file', 20),
        file_indices=train_indices,
        file_list=all_files,
        seed=42,
        fixed_snippets=config.get('fixed_snippets', False),
        preload_into_ram=config.get('preload_into_ram', True),
        duration_mode=config.get('duration_mode', 'log')  # NEW: pass duration mode
    )
    
    val_dataset = PianoTranscriptionDataset(
        data_dir,
        config['snippet_frames'],
        snippets_per_file=config.get('snippets_per_file', 20),
        file_indices=val_indices,
        file_list=all_files,
        seed=42,
        fixed_snippets=config.get('fixed_snippets', False),
        preload_into_ram=config.get('preload_into_ram', True),
        duration_mode=config.get('duration_mode', 'log')  # NEW: pass duration mode
    )
    
    print(f"\nDataset summary:")
    print(f"  Duration mode: {config.get('duration_mode', 'log')}")
    print(f"  Train snippets: {len(train_dataset)}")
    print(f"  Val snippets:   {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=config.get('shuffle_train', True),  # Use config setting
        num_workers=config.get('num_workers', 0),
        pin_memory=config.get('pin_memory', False),
        prefetch_factor=config.get('prefetch_factor', 2) if config.get('num_workers', 0) > 0 else None,
        persistent_workers=config.get('persistent_workers', False) if config.get('num_workers', 0) > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 0),
        pin_memory=config.get('pin_memory', False),
        prefetch_factor=config.get('prefetch_factor', 2) if config.get('num_workers', 0) > 0 else None,
        persistent_workers=config.get('persistent_workers', False) if config.get('num_workers', 0) > 0 else False
    )
    
    return train_loader, val_loader


def create_scheduler(optimizer, config, num_epochs):
    """Create learning rate scheduler based on config."""
    if not config.get('use_lr_scheduler', False):
        return None
    
    scheduler_type = config['scheduler_type']
    
    if scheduler_type == 'cosine_warmup':
        warmup_epochs = config.get('warmup_epochs', 5)
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs - warmup_epochs,
            eta_min=config.get('scheduler_min_lr', 1e-6)
        )
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs]
        )
        print(f"Using cosine scheduler with {warmup_epochs}-epoch warmup")
    
    elif scheduler_type == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.get('scheduler_factor', 0.5),
            patience=config.get('scheduler_patience', 10),
            min_lr=config.get('scheduler_min_lr', 1e-6),
            threshold=config.get('scheduler_threshold', 1e-4),
            threshold_mode='rel'
        )
        print(f"Using ReduceLROnPlateau scheduler")
    
    elif scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=config.get('scheduler_min_lr', 1e-6)
        )
        print(f"Using cosine annealing scheduler")
    
    else:
        print(f"Unknown scheduler type: {scheduler_type}")
        return None
    
    return scheduler


def compute_frame_metrics(predictions, targets, threshold=0.5):
    """
    Compute frame-level precision, recall, and F1 score.
    
    Args:
        predictions: Frame predictions (batch, time, 88) - logits or probabilities
        targets: Frame targets (batch, time, 88) - binary
        threshold: Threshold for binary classification
    
    Returns:
        dict: Metrics dictionary with precision, recall, f1
    """
    # Apply sigmoid if needed and threshold
    if predictions.min() < 0 or predictions.max() > 1:
        predictions = torch.sigmoid(predictions)
    
    pred_binary = (predictions > threshold).float()
    
    # Compute true positives, false positives, false negatives
    tp = (pred_binary * targets).sum().item()
    fp = (pred_binary * (1 - targets)).sum().item()
    fn = ((1 - pred_binary) * targets).sum().item()
    
    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_losses = {'onset_loss': 0, 'duration_loss': 0, 'frame_loss': 0, 'total_loss': 0}
    total_metrics = {'precision': 0, 'recall': 0, 'f1': 0}
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        spectrogram = batch['spectrogram'].unsqueeze(1).to(device)
        onset = batch['onset'].to(device)
        duration = batch['duration'].to(device)
        frame = batch['frame'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(spectrogram)
        
        # Compute loss
        targets = {'onset': onset, 'duration': duration, 'frame': frame}
        loss, loss_dict = criterion(predictions, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Compute frame metrics
        with torch.no_grad():
            metrics = compute_frame_metrics(predictions['frame'], frame)
        
        # Accumulate losses and metrics
        for key in total_losses:
            total_losses[key] += loss_dict[key]
        for key in total_metrics:
            total_metrics[key] += metrics[key]
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss_dict['total_loss']:.4f}",
            'f1': f"{metrics['f1']:.3f}"
        })
    
    # Average losses and metrics
    num_batches = len(dataloader)
    for key in total_losses:
        total_losses[key] /= num_batches
    for key in total_metrics:
        total_metrics[key] /= num_batches
    
    return total_losses, total_metrics


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_losses = {'onset_loss': 0, 'duration_loss': 0, 'frame_loss': 0, 'total_loss': 0}
    total_metrics = {'precision': 0, 'recall': 0, 'f1': 0}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            # Move data to device
            spectrogram = batch['spectrogram'].unsqueeze(1).to(device)
            onset = batch['onset'].to(device)
            duration = batch['duration'].to(device)
            frame = batch['frame'].to(device)
            
            # Forward pass
            predictions = model(spectrogram)
            
            # Compute loss
            targets = {'onset': onset, 'duration': duration, 'frame': frame}
            loss, loss_dict = criterion(predictions, targets)
            
            # Compute frame metrics
            metrics = compute_frame_metrics(predictions['frame'], frame)
            
            # Accumulate losses and metrics
            for key in total_losses:
                total_losses[key] += loss_dict[key]
            for key in total_metrics:
                total_metrics[key] += metrics[key]
    
    # Average losses and metrics
    num_batches = len(dataloader)
    for key in total_losses:
        total_losses[key] /= num_batches
    for key in total_metrics:
        total_metrics[key] /= num_batches
    
    return total_losses, total_metrics


def train(data_dir, run_folder, config):
    """Main training loop."""
    # Setup
    device = torch.device('mps' if torch.backends.mps.is_available() else
                         'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Saving to: {run_folder}\n")
    
    # Create datasets and loaders
    train_loader, val_loader = create_datasets_and_loaders(data_dir, config)
    
    # Create model - choose based on config
    if config.get('use_cnn_only', False):
        print("\n" + "="*80)
        print("ABLATION STUDY: Using CNN-only model (no Transformer)")
        print("="*80 + "\n")
        model = PianoTranscriptionModelCNNOnly(
            n_mels=config['n_mels'],
            hidden_size=config['hidden_size'],
            num_outputs=NUM_OUTPUTS,
            dropout=config['dropout']
        ).to(device)
    else:
        model = PianoTranscriptionModel(
            input_features=config['n_mels'],
            num_keys=config['num_keys'],
            transformer_dim=config['hidden_size'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            duration_mode=config.get('duration_mode', 'log'),  # ADD: pass duration_mode
            num_duration_bins=config.get('num_duration_bins', 8)  # ADD: pass num_duration_bins
        ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer - use config weights
    criterion = OnsetDurationLoss(
        onset_weight=config.get('onset_weight', 4.0),
        duration_weight=config.get('duration_weight', 2.0),
        frame_weight=config.get('frame_weight', 1.0),
        duration_mode=config.get('duration_mode', 'log')
    )
    
    print(f"\nLoss weights:")
    print(f"  Onset: {config.get('onset_weight', 4.0)}")
    print(f"  Duration: {config.get('duration_weight', 2.0)}")
    print(f"  Frame: {config.get('frame_weight', 1.0)}")
    print(f"  Duration mode: {config.get('duration_mode', 'log')}")
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 1e-5)
    )
    
    # Scheduler
    scheduler = create_scheduler(optimizer, config, config['num_epochs'])
    
    # Checkpoint paths
    model_path = os.path.join(run_folder, "model.pth")
    curves_path = os.path.join(run_folder, "training_curves.png")
    checkpoint_path = os.path.join(run_folder, "checkpoint.pth")
    
    # Initialize training state
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_metrics = []
    val_metrics = []
    start_epoch = 0
    
    # Resume from checkpoint if available
    if os.path.exists(checkpoint_path):
        checkpoint = load_checkpoint(checkpoint_path, model, optimizer, scheduler, device)
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        train_metrics = checkpoint['train_metrics']
        val_metrics = checkpoint['val_metrics']
    
    # Save config to run folder
    save_config_to_run_folder(config, run_folder)
    
    # Training loop
    print(f"\nStarting training from epoch {start_epoch + 1}\n")
    
    for epoch in range(start_epoch, config['num_epochs']):
        print(f"Epoch {epoch + 1}/{config['num_epochs']}")
        
        # Train and validate
        train_loss, train_metric = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metric = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss['total_loss'])
        val_losses.append(val_loss['total_loss'])
        train_metrics.append(train_metric)
        val_metrics.append(val_metric)
        
        # Print stats
        print(f"Loss - Train: {train_loss['total_loss']:.4f}, Val: {val_loss['total_loss']:.4f}")
        print(f"Train - Onset: {train_loss['onset_loss']:.4f}, "
              f"Duration: {train_loss['duration_loss']:.4f}, Frame: {train_loss['frame_loss']:.4f}")
        print(f"Val   - Onset: {val_loss['onset_loss']:.4f}, "
              f"Duration: {val_loss['duration_loss']:.4f}, Frame: {val_loss['frame_loss']:.4f}")
        print(f"Train - Precision: {train_metric['precision']:.4f}, Recall: {train_metric['recall']:.4f}, F1: {train_metric['f1']:.4f}")
        print(f"Val   - Precision: {val_metric['precision']:.4f}, Recall: {val_metric['recall']:.4f}, F1: {val_metric['f1']:.4f}")
        
        # Update scheduler
        if scheduler is not None:
            if config['scheduler_type'] == 'reduce_on_plateau':
                scheduler.step(val_loss['total_loss'])
            else:
                scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"LR: {current_lr:.2e}")
        
        # Save best model
        if val_loss['total_loss'] < best_val_loss:
            best_val_loss = val_loss['total_loss']
            torch.save(model.state_dict(), model_path)
            print(f"✓ Best model saved")
        
        # Save checkpoint
        save_checkpoint(epoch, model, optimizer, scheduler, train_losses, val_losses,
                       train_metrics, val_metrics, best_val_loss, checkpoint_path)
        
        # Plot curves
        plot_training_curves(train_losses, val_losses, train_metrics, val_metrics, curves_path)
        
        # Visualize predictions periodically
        if (epoch + 1) % 50 == 0:
            vis_path = os.path.join(run_folder, f"predictions_epoch_{epoch + 1}.png")
            visualize_predictions(model, val_loader, device, vis_path)
        
        print()


if __name__ == "__main__":
    data_dir = CONFIG['data_dir']
    run_folder = get_next_run_folder()
    
    print(f"Starting training run: {run_folder}")
    print(f"Data directory: {data_dir}\n")
    
    train(data_dir, run_folder, CONFIG)