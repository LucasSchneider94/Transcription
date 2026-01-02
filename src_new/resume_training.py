"""
Resume training from a checkpoint.
Usage: python resume_training.py <run_folder>
Example: python resume_training.py training_run_015
"""

import sys
import os
import json
import torch

from config import CONFIG
from train import train

def resume_training(run_folder):
    """Resume training from a checkpoint."""
    
    # Check if run folder exists
    if not os.path.exists(run_folder):
        print(f"Error: Run folder '{run_folder}' not found")
        sys.exit(1)
    
    # Check if checkpoint exists
    checkpoint_path = os.path.join(run_folder, "checkpoint.pth")
    if not os.path.exists(checkpoint_path):
        print(f"Error: No checkpoint found in '{run_folder}'")
        sys.exit(1)
    
    # Load config from run folder
    config_path = os.path.join(run_folder, "config.json")
    if (os.path.exists(config_path)):
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
        
        # Update CONFIG with saved values
        for key, value in saved_config.items():
            if key in CONFIG:
                CONFIG[key] = value
        
        print(f"Loaded config from {config_path}")
    else:
        print(f"Warning: No config.json found, using current CONFIG")
    
    # Load checkpoint to check status
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    current_epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']
    
    print(f"\n{'='*80}")
    print(f"RESUMING TRAINING")
    print(f"{'='*80}")
    print(f"Run folder: {run_folder}")
    print(f"Last completed epoch: {current_epoch + 1}")
    print(f"Resuming from epoch: {current_epoch + 2}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"{'='*80}\n")
    
    # Ask for confirmation
    response = input("Continue training? (y/n): ")
    if response.lower() != 'y':
        print("Training cancelled")
        sys.exit(0)
    
    # Start training (will automatically load checkpoint)
    data_dir = CONFIG['data_dir']
    train(data_dir, run_folder, CONFIG)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python resume_training.py <run_folder>")
        print("Example: python resume_training.py training_run_015")
        sys.exit(1)
    
    run_folder = sys.argv[1]
    resume_training(run_folder)
