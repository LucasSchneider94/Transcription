"""
Script to resume training from a checkpoint.
Simply run this script and it will automatically detect and resume from checkpoint.pth
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os

from train import SnippetDataset, train_epoch, validate, visualize_predictions, plot_training_curves, save_checkpoint, load_checkpoint
from model import PianoTranscriptionModel
from config import CONFIG, NUM_OUTPUTS
from resume_config import RESUME_CONFIG


def resume_training():
    """
    Resume training from a checkpoint.
    """
    checkpoint_path = RESUME_CONFIG['checkpoint_path']
    
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print("Error: No checkpoint specified or checkpoint file not found!")
        print(f"Please set RESUME_CONFIG['checkpoint_path'] in resume_config.py")
        print(f"Example: 'checkpoint_path': 'training_run_012/checkpoint.pth'")
        return
    
    # Determine run folder
    if RESUME_CONFIG['continue_in_same_folder']:
        run_folder = os.path.dirname(checkpoint_path)
        print(f"Continuing training in existing folder: {run_folder}")
    else:
        from train import get_next_run_folder
        run_folder = get_next_run_folder()
        print(f"Starting new training run in: {run_folder}")
        # Copy checkpoint to new folder
        import shutil
        new_checkpoint_path = os.path.join(run_folder, "checkpoint.pth")
        shutil.copy(checkpoint_path, new_checkpoint_path)
        checkpoint_path = new_checkpoint_path
    
    # Setup device
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                         'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset and dataloader
    data_dir = CONFIG['data_dir']
    dataset = SnippetDataset(data_dir, CONFIG['snippet_frames'])
    print(f"Dataset size: {len(dataset)} snippets")
    
    # Split into train/val (same split as original training)
    # IMPORTANT: The dataset files are now sorted, ensuring reproducibility
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)  # Same seed as train.py
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True, 
        num_workers=CONFIG['num_workers'],
        pin_memory=CONFIG['pin_memory'],
        prefetch_factor=CONFIG['prefetch_factor'] if CONFIG['num_workers'] > 0 else None,
        persistent_workers=CONFIG['persistent_workers'] if CONFIG['num_workers'] > 0 else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False, 
        num_workers=CONFIG['num_workers'],
        pin_memory=CONFIG['pin_memory'],
        prefetch_factor=CONFIG['prefetch_factor'] if CONFIG['num_workers'] > 0 else None,
        persistent_workers=CONFIG['persistent_workers'] if CONFIG['num_workers'] > 0 else False
    )
    
    # Create model and optimizer
    model = PianoTranscriptionModel(
        n_mels=CONFIG['n_mels'],
        hidden_size=CONFIG['hidden_size'],
        num_heads=CONFIG['num_heads'],
        num_layers=CONFIG['num_layers'],
        num_outputs=NUM_OUTPUTS,
        dropout=CONFIG['dropout']
    ).to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-5)
    
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path, model, optimizer, device)
    
    # Extract checkpoint data
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    train_precisions = checkpoint['train_precisions']
    val_precisions = checkpoint['val_precisions']
    train_recalls = checkpoint['train_recalls']
    val_recalls = checkpoint['val_recalls']
    train_f1s = checkpoint['train_f1s']
    val_f1s = checkpoint['val_f1s']
    
    # Determine total epochs
    if RESUME_CONFIG['additional_epochs'] is not None:
        total_epochs = start_epoch + RESUME_CONFIG['additional_epochs']
        print(f"Training for {RESUME_CONFIG['additional_epochs']} additional epochs (to epoch {total_epochs})")
    else:
        total_epochs = CONFIG['num_epochs']
        print(f"Training until epoch {total_epochs}")
    
    # Define output paths
    model_save_path = os.path.join(run_folder, "model.pth")
    curves_save_path = os.path.join(run_folder, "training_curves.png")
    checkpoint_save_path = os.path.join(run_folder, "checkpoint.pth")
    
    # Training loop
    for epoch in range(start_epoch, total_epochs):
        print(f"\nEpoch {epoch + 1}/{total_epochs}")
        
        # Train
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_precisions.append(train_metrics['precision'])
        train_recalls.append(train_metrics['recall'])
        train_f1s.append(train_metrics['f1'])
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_precisions.append(val_metrics['precision'])
        val_recalls.append(val_metrics['recall'])
        val_f1s.append(val_metrics['f1'])
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Train - P: {train_metrics['precision']:.4f}, R: {train_metrics['recall']:.4f}, F1: {train_metrics['f1']:.4f}")
        print(f"Val   - P: {val_metrics['precision']:.4f}, R: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Best model updated and saved to {model_save_path}")
        
        # Save checkpoint every epoch
        save_checkpoint(epoch, model, optimizer, train_losses, val_losses,
                       train_precisions, val_precisions, train_recalls, val_recalls,
                       train_f1s, val_f1s, best_val_loss, checkpoint_save_path)
        
        # Save training curves every 10 epochs
        if (epoch + 1) % 10 == 0:
            plot_training_curves(
                train_losses, val_losses,
                train_precisions, val_precisions,
                train_recalls, val_recalls,
                train_f1s, val_f1s,
                curves_save_path
            )
        
        # Visualize predictions every 100 epochs
        if (epoch + 1) % 100 == 0:
            vis_path = os.path.join(run_folder, f"predictions_epoch_{epoch + 1}.png")
            visualize_predictions(model, val_loader, device, vis_path)
    
    # Final training curves
    plot_training_curves(
        train_losses, val_losses,
        train_precisions, val_precisions,
        train_recalls, val_recalls,
        train_f1s, val_f1s,
        curves_save_path
    )
    
    print(f"\n✅ Training complete! Final epoch: {total_epochs}")


if __name__ == "__main__":
    resume_training()
