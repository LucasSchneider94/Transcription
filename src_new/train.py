import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob

from config import CONFIG, NUM_OUTPUTS
from model import PianoTranscriptionModel
from resume_config import RESUME_CONFIG


def get_next_run_folder():
    """
    Find the next available training run folder number.
    
    Returns:
        str: Path to the next training run folder (e.g., "training_run_002")
    """
    existing_runs = glob.glob("training_run_*")
    if not existing_runs:
        run_number = 1
    else:
        # Extract numbers from folder names
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


class SnippetDataset(Dataset):
    """
    Dataset that randomly samples snippets from spectrograms and piano rolls.
    Each epoch will see different random snippets for better generalization.
    Uses memory-efficient loading to handle large datasets.
    
    The dataset ensures that:
    1. All files are seen at least once per epoch
    2. Additional random snippets provide data augmentation
    3. Snippet positions are randomized on every access
    """
    def __init__(self, data_dir, snippet_frames, snippets_per_file=10):
        """
        Args:
            data_dir (str): Directory containing processed .npz and .npy files.
            snippet_frames (int): Number of frames per snippet.
            snippets_per_file (int): Number of random snippets to sample per audio file per epoch.
        """
        self.data_dir = data_dir
        self.snippet_frames = snippet_frames
        self.snippets_per_file = snippets_per_file
        self.files = []
        
        # Collect all valid files (SORTED for reproducibility)
        files = sorted([f for f in os.listdir(data_dir) if f.endswith("_piano_roll_with_pedals.npz")])
        
        for file_name in files:
            piano_roll_path = os.path.join(data_dir, file_name)
            spectrogram_path = piano_roll_path.replace("_piano_roll_with_pedals.npz", "_spectrogram.npy")
            
            if not os.path.exists(spectrogram_path):
                continue
            
            # Memory-efficient: Use mmap_mode to only read metadata, not the full file
            spectrogram = np.load(spectrogram_path, mmap_mode='r')
            piano_roll_data = np.load(piano_roll_path, mmap_mode='r')
            piano_roll = piano_roll_data["piano_roll"]
            
            # Get dimensions without loading full data into memory
            min_frames = min(spectrogram.shape[1], piano_roll.shape[0])
            
            # Only include files that are long enough
            if min_frames >= snippet_frames:
                self.files.append({
                    'spectrogram_path': spectrogram_path,
                    'piano_roll_path': piano_roll_path,
                    'max_frames': min_frames
                })
            
            # Close memory-mapped files to free resources
            del spectrogram
            if hasattr(piano_roll_data, 'close'):
                piano_roll_data.close()
            del piano_roll_data
        
        print(f"Loaded {len(self.files)} audio files")
        print(f"Total snippets per epoch: {len(self)} ({self.snippets_per_file} per file)")
    
    def __len__(self):
        return len(self.files) * self.snippets_per_file
    
    def __getitem__(self, idx):
        """
        Returns a random snippet from the audio file corresponding to idx.
        
        The mapping ensures each file appears snippets_per_file times per epoch,
        but the DataLoader's shuffle=True randomizes the order they're seen.
        Each access samples a NEW random position within the file.
        """
        # Determine which file to use (deterministic mapping from idx)
        file_idx = idx % len(self.files)
        file_info = self.files[file_idx]
        
        # Randomly sample a starting position (different every time this idx is accessed)
        max_start = file_info['max_frames'] - self.snippet_frames
        start_frame = np.random.randint(0, max_start + 1)
        end_frame = start_frame + self.snippet_frames
        
        # Memory-efficient: Load only the required snippet using memory mapping
        # This loads only the necessary data from disk, not the entire file
        spectrogram_mmap = np.load(file_info['spectrogram_path'], mmap_mode='r')
        spectrogram_snippet = np.array(spectrogram_mmap[:, start_frame:end_frame])  # Copy only the snippet
        del spectrogram_mmap  # Free the memory-mapped file
        
        piano_roll_data = np.load(file_info['piano_roll_path'], mmap_mode='r')
        piano_roll = piano_roll_data["piano_roll"]
        piano_roll_snippet = np.array(piano_roll[start_frame:end_frame, :])  # Copy only the snippet
        if hasattr(piano_roll_data, 'close'):
            piano_roll_data.close()
        del piano_roll_data  # Free the memory-mapped file
        
        # Convert to tensors
        spectrogram_tensor = torch.tensor(spectrogram_snippet, dtype=torch.float32).unsqueeze(0)  # (1, n_mels, snippet_frames)
        piano_roll_tensor = torch.tensor(piano_roll_snippet, dtype=torch.float32)  # (snippet_frames, num_outputs)
        
        return spectrogram_tensor, piano_roll_tensor


def compute_metrics(predictions, targets, threshold=0.5):
    """
    Compute precision, recall, and F1 score for binary predictions.
    
    Args:
        predictions (torch.Tensor): Model predictions (batch_size, seq_len, num_outputs)
        targets (torch.Tensor): Ground truth (batch_size, seq_len, num_outputs)
        threshold (float): Threshold to convert probabilities to binary predictions
    
    Returns:
        dict: Dictionary containing precision, recall, and F1 score
    """
    # Convert predictions to binary (0 or 1)
    pred_binary = (predictions > threshold).float()
    
    # Flatten tensors
    pred_flat = pred_binary.view(-1)
    target_flat = targets.view(-1)
    
    # Calculate true positives, false positives, false negatives
    tp = (pred_flat * target_flat).sum().item()
    fp = (pred_flat * (1 - target_flat)).sum().item()
    fn = ((1 - pred_flat) * target_flat).sum().item()
    
    # Calculate metrics (avoid division by zero)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train for one epoch and return loss and metrics.
    """
    model.train()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    for spectrograms, piano_rolls in tqdm(dataloader, desc="Training"):
        spectrograms = spectrograms.to(device)
        piano_rolls = piano_rolls.to(device)
        
        # Forward pass
        predictions = model(spectrograms)
        
        # Compute loss
        loss = criterion(predictions, piano_rolls)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Store predictions and targets for metrics (detach to save memory)
        all_predictions.append(predictions.detach())
        all_targets.append(piano_rolls.detach())
    
    avg_loss = total_loss / len(dataloader)
    
    # Compute metrics on accumulated predictions
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(all_predictions, all_targets)
    
    return avg_loss, metrics


def validate(model, dataloader, criterion, device):
    """
    Validate the model.
    """
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for spectrograms, piano_rolls in tqdm(dataloader, desc="Validating"):
            spectrograms = spectrograms.to(device)
            piano_rolls = piano_rolls.to(device)
            
            # Forward pass
            predictions = model(spectrograms)
            
            # Compute loss
            loss = criterion(predictions, piano_rolls)
            total_loss += loss.item()
            
            # Store predictions and targets for metrics
            all_predictions.append(predictions)
            all_targets.append(piano_rolls)
    
    # Compute metrics
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(all_predictions, all_targets)
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, metrics


def visualize_predictions(model, dataloader, device, save_path):
    """
    Visualize model predictions vs ground truth.
    """
    model.eval()
    
    # Get one batch
    spectrograms, piano_rolls = next(iter(dataloader))
    spectrograms = spectrograms.to(device)
    
    with torch.no_grad():
        predictions = model(spectrograms)
    
    # Take first sample
    pred = predictions[0].cpu().numpy()  # (snippet_frames, num_outputs)
    gt = piano_rolls[0].cpu().numpy()    # (snippet_frames, num_outputs)
    
    # Plot
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


def plot_training_curves(train_losses, val_losses, train_precisions, val_precisions,
                         train_recalls, val_recalls, train_f1s, val_f1s, save_path):
    """
    Plot training curves showing loss and metrics.
    
    Args:
        train_losses, val_losses: Lists of loss values
        train_precisions, val_precisions: Lists of precision values
        train_recalls, val_recalls: Lists of recall values
        train_f1s, val_f1s: Lists of F1 score values
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    axes[0, 0].plot(train_losses, label='Train Loss')
    axes[0, 0].plot(val_losses, label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Precision plot
    axes[0, 1].plot(train_precisions, label='Train Precision')
    axes[0, 1].plot(val_precisions, label='Val Precision')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Recall plot
    axes[1, 0].plot(train_recalls, label='Train Recall')
    axes[1, 0].plot(val_recalls, label='Val Recall')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Recall')
    axes[1, 0].set_title('Recall')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # F1 plot
    axes[1, 1].plot(train_f1s, label='Train F1')
    axes[1, 1].plot(val_f1s, label='Val F1')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].set_title('F1 Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training curves saved to {save_path}")


def save_checkpoint(epoch, model, optimizer, train_losses, val_losses, 
                   train_precisions, val_precisions, train_recalls, val_recalls,
                   train_f1s, val_f1s, best_val_loss, save_path):
    """
    Save a complete training checkpoint.
    
    Args:
        epoch (int): Current epoch number
        model: The model to save
        optimizer: The optimizer to save
        train_losses, val_losses: Loss history
        train_precisions, val_precisions: Precision history
        train_recalls, val_recalls: Recall history
        train_f1s, val_f1s: F1 score history
        best_val_loss (float): Best validation loss so far
        save_path (str): Path to save the checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_precisions': train_precisions,
        'val_precisions': val_precisions,
        'train_recalls': train_recalls,
        'val_recalls': val_recalls,
        'train_f1s': train_f1s,
        'val_f1s': val_f1s,
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(checkpoint_path, model, optimizer, device):
    """
    Load a training checkpoint.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        model: The model to load weights into
        optimizer: The optimizer to load state into
        device: torch device
        
    Returns:
        dict: Checkpoint data including epoch, histories, and best_val_loss
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"Resuming from epoch {checkpoint['epoch'] + 1}")
    print(f"Best validation loss so far: {checkpoint['best_val_loss']:.4f}")
    
    return checkpoint


def train(data_dir, run_folder, num_epochs, batch_size, learning_rate):
    """
    Main training loop.
    
    Args:
        data_dir (str): Directory containing processed data
        run_folder (str): Folder to save all training outputs
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size
        learning_rate (float): Learning rate
    """
    # Setup device
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                         'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Saving outputs to: {run_folder}")
    
    # Create dataset and dataloader
    dataset = SnippetDataset(data_dir, CONFIG['snippet_frames'])
    print(f"Dataset size: {len(dataset)} snippets")
    
    # Split into train/val (80/20) with fixed seed for reproducibility
    # This ensures the same files are always in train vs validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)  # Fixed seed for reproducible splits
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Create model
    model = PianoTranscriptionModel(
        n_mels=CONFIG['n_mels'],
        hidden_size=CONFIG['hidden_size'],
        num_heads=CONFIG['num_heads'],
        num_layers=CONFIG['num_layers'],
        num_outputs=NUM_OUTPUTS,
        dropout=CONFIG['dropout']
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer with weight decay
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Define output paths in run folder
    model_save_path = os.path.join(run_folder, "model.pth")
    curves_save_path = os.path.join(run_folder, "training_curves.png")
    checkpoint_save_path = os.path.join(run_folder, "checkpoint.pth")
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_precisions = []
    train_recalls = []
    train_f1s = []
    val_precisions = []
    val_recalls = []
    val_f1s = []
    
    start_epoch = 0
    
    # Resume from checkpoint if available
    if os.path.exists(checkpoint_save_path):
        checkpoint = load_checkpoint(checkpoint_save_path, model, optimizer, device)
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
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
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
            print(f"Model saved to {model_save_path}")
        
        # Save checkpoint every epoch
        save_checkpoint(epoch, model, optimizer, train_losses, val_losses, 
                        train_precisions, val_precisions, train_recalls, val_recalls,
                        train_f1s, val_f1s, best_val_loss, checkpoint_save_path)
        
        # Save training curves every 10 epochs (overwrite the same file)
        if (epoch + 1) % 10 == 0:
            plot_training_curves(
                train_losses, val_losses,
                train_precisions, val_precisions,
                train_recalls, val_recalls,
                train_f1s, val_f1s,
                curves_save_path  # Overwrite the same file
            )
        
        # Visualize predictions every 100 epochs
        if (epoch + 1) % 100 == 0:
            vis_path = os.path.join(run_folder, f"predictions_epoch_{epoch + 1}.png")
            visualize_predictions(model, val_loader, device, vis_path)
    
    # Plot training curves with metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    axes[0, 0].plot(train_losses, label='Train Loss')
    axes[0, 0].plot(val_losses, label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Precision plot
    axes[0, 1].plot(train_precisions, label='Train Precision')
    axes[0, 1].plot(val_precisions, label='Val Precision')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Recall plot
    axes[1, 0].plot(train_recalls, label='Train Recall')
    axes[1, 0].plot(val_recalls, label='Val Recall')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Recall')
    axes[1, 0].set_title('Recall')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # F1 plot
    axes[1, 1].plot(train_f1s, label='Train F1')
    axes[1, 1].plot(val_f1s, label='Val F1')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].set_title('F1 Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(curves_save_path)
    plt.close()
    print(f"Training curves saved to {curves_save_path}")


if __name__ == "__main__":
    # Use data directory from config
    data_dir = CONFIG['data_dir']
    
    # Get next training run folder
    run_folder = get_next_run_folder()
    print(f"Starting new training run in: {run_folder}")
    print(f"Using data from: {data_dir}")
    
    train(
        data_dir=data_dir,
        run_folder=run_folder,
        num_epochs=CONFIG['num_epochs'],
        batch_size=CONFIG['batch_size'],
        learning_rate=CONFIG['learning_rate']
    )