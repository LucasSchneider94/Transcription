"""
Training script for piano transcription model.
Refactored for clarity and maintainability.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm

from config import CONFIG, NUM_OUTPUTS
from model import PianoTranscriptionModel
from dataset import SnippetDataset
from utils import (
    get_next_run_folder,
    compute_metrics,
    visualize_predictions,
    plot_training_curves,
    save_checkpoint,
    load_checkpoint
)


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
    
    # Create datasets
    train_dataset = SnippetDataset(
        data_dir,
        config['snippet_frames'],
        snippets_per_file=config.get('snippets_per_file', 20),
        file_indices=train_indices,
        file_list=all_files,
        seed=42
    )
    
    val_dataset = SnippetDataset(
        data_dir,
        config['snippet_frames'],
        snippets_per_file=config.get('snippets_per_file', 20),
        file_indices=val_indices,
        file_list=all_files,
        seed=42
    )
    
    print(f"\nDataset summary:")
    print(f"  Train snippets: {len(train_dataset)}")
    print(f"  Val snippets:   {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
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


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    for spectrograms, piano_rolls in tqdm(dataloader, desc="Training"):
        spectrograms = spectrograms.to(device)
        piano_rolls = piano_rolls.to(device)
        
        predictions = model(spectrograms)
        loss = criterion(predictions, piano_rolls)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_predictions.append(predictions.detach())
        all_targets.append(piano_rolls.detach())
    
    avg_loss = total_loss / len(dataloader)
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(all_predictions, all_targets)
    
    return avg_loss, metrics


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for spectrograms, piano_rolls in tqdm(dataloader, desc="Validating"):
            spectrograms = spectrograms.to(device)
            piano_rolls = piano_rolls.to(device)
            
            predictions = model(spectrograms)
            loss = criterion(predictions, piano_rolls)
            
            total_loss += loss.item()
            all_predictions.append(predictions)
            all_targets.append(piano_rolls)
    
    avg_loss = total_loss / len(dataloader)
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(all_predictions, all_targets)
    
    return avg_loss, metrics


def train(data_dir, run_folder, config):
    """Main training loop."""
    # Setup
    device = torch.device('mps' if torch.backends.mps.is_available() else
                         'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Saving to: {run_folder}\n")
    
    # Create datasets and loaders
    train_loader, val_loader = create_datasets_and_loaders(data_dir, config)
    
    # Create model
    model = PianoTranscriptionModel(
        n_mels=config['n_mels'],
        hidden_size=config['hidden_size'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        num_outputs=NUM_OUTPUTS,
        dropout=config['dropout']
    ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.BCELoss()
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
    
    # Training loop
    print(f"\nStarting training from epoch {start_epoch + 1}\n")
    
    for epoch in range(start_epoch, config['num_epochs']):
        print(f"Epoch {epoch + 1}/{config['num_epochs']}")
        
        # Train and validate
        train_loss, train_metric = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metric = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_metrics.append(train_metric)
        val_metrics.append(val_metric)
        
        # Print stats
        print(f"Loss - Train: {train_loss:.4f}, Val: {val_loss:.4f}")
        print(f"Train - P: {train_metric['precision']:.4f}, "
              f"R: {train_metric['recall']:.4f}, F1: {train_metric['f1']:.4f}")
        print(f"Val   - P: {val_metric['precision']:.4f}, "
              f"R: {val_metric['recall']:.4f}, F1: {val_metric['f1']:.4f}")
        
        # Update scheduler
        if scheduler is not None:
            if config['scheduler_type'] == 'reduce_on_plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"LR: {current_lr:.2e}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
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