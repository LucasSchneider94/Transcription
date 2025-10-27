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


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Focuses training on hard examples and reduces weight of easy negatives.
    
    This helps when the dataset has many more 0s (silence) than 1s (notes),
    preventing the model from being too aggressive with predictions.
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha (float): Weighting factor for positive class (0-1). 
                      Higher = more weight on positive examples (notes).
                      Default 0.25 means positives are weighted 4x more than negatives.
        gamma (float): Focusing parameter. Higher = more focus on hard examples.
                      gamma=0 is equivalent to standard BCE loss.
                      Default 2.0 is commonly used.
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Model predictions (after sigmoid), shape (batch, seq_len, num_outputs)
            targets: Ground truth, shape (batch, seq_len, num_outputs)
        """
        # BCE loss for each element
        bce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        
        # Calculate p_t: probability of the true class
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        
        # Calculate focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Calculate alpha weight
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Combine all components
        focal_loss = alpha_weight * focal_weight * bce_loss
        
        return focal_loss.mean()


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
    Uses memory-efficient loading to handle large datasets.
    
    IMPORTANT: Supports file-based train/val splitting to prevent data leakage.
    When file_indices is provided, only snippets from those files are included.
    
    NEW: Samples random snippet positions on each __getitem__ call, providing
    strong data augmentation and preventing overfitting to specific positions.
    
    AUGMENTATION: Applies SpecAugment-style masking (time and frequency) to spectrograms
    during training to further prevent overfitting.
    
    The dataset ensures that:
    1. Train and validation sets use completely different files (no leakage)
    2. Each epoch sees different random snippets from the same files (data augmentation)
    3. All files are seen multiple times per epoch (snippets_per_file times)
    4. Memory-efficient loading with on-demand random sampling
    5. Optional time/frequency masking for robustness
    """
    def __init__(self, data_dir, snippet_frames, snippets_per_file=10, seed=42, 
                 file_indices=None, augment=False, time_mask_param=30, freq_mask_param=20):
        """
        Args:
            data_dir (str): Directory containing processed .npz and .npy files.
            snippet_frames (int): Number of frames per snippet.
            snippets_per_file (int): Number of snippets to sample per file per epoch.
            seed (int): Random seed for reproducible file selection (NOT snippet positions).
            file_indices (list, optional): List of file indices to include. If None, use all files.
                                          Use this to create train/val splits based on files, not snippets.
            augment (bool): Whether to apply data augmentation (time/freq masking).
            time_mask_param (int): Maximum number of consecutive time frames to mask.
            freq_mask_param (int): Maximum number of consecutive frequency bins to mask.
        """
        self.data_dir = data_dir
        self.snippet_frames = snippet_frames
        self.snippets_per_file = snippets_per_file
        self.file_list = []  # List of file_info dicts
        self.augment = augment
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        
        # Set random seed for reproducible FILE selection (not snippet positions)
        np.random.seed(seed)
        
        # Collect all valid files (SORTED for reproducibility)
        all_files = sorted([f for f in os.listdir(data_dir) if f.endswith("_piano_roll_with_pedals.npz")])
        
        # Filter files if file_indices is provided (for train/val split)
        if file_indices is not None:
            files = [all_files[i] for i in file_indices if i < len(all_files)]
        else:
            files = all_files
        
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
                file_info = {
                    'spectrogram_path': spectrogram_path,
                    'piano_roll_path': piano_roll_path,
                    'max_frames': min_frames,
                    'max_start': min_frames - snippet_frames
                }
                self.file_list.append(file_info)
            
            # Close memory-mapped files to free resources
            del spectrogram
            if hasattr(piano_roll_data, 'close'):
                piano_roll_data.close()
            del piano_roll_data
        
        # Reset random seed to not affect other random operations
        np.random.seed(None)
        
        num_files = len(self.file_list)
        total_snippets = num_files * self.snippets_per_file
        
        print(f"Loaded {num_files} audio files")
        print(f"Snippets per epoch: {total_snippets} ({self.snippets_per_file} per file)")
        print(f"🎲 RANDOM snippet positions - different every epoch (prevents overfitting!)")
        if self.augment:
            print(f"🎨 Data augmentation ENABLED: Time masking (max {time_mask_param} frames), Freq masking (max {freq_mask_param} bins)")
        if file_indices is not None:
            print(f"Using file-based subset (indices: {len(file_indices)} files)")
    
    def _apply_time_mask(self, spectrogram):
        """Apply time masking (SpecAugment) to spectrogram."""
        n_mels, n_frames = spectrogram.shape
        if n_frames < self.time_mask_param:
            return spectrogram
        
        # Random mask width
        mask_width = np.random.randint(1, self.time_mask_param + 1)
        # Random start position
        mask_start = np.random.randint(0, n_frames - mask_width + 1)
        
        # Apply mask (set to zero)
        spectrogram_masked = spectrogram.copy()
        spectrogram_masked[:, mask_start:mask_start + mask_width] = 0
        return spectrogram_masked
    
    def _apply_freq_mask(self, spectrogram):
        """Apply frequency masking (SpecAugment) to spectrogram."""
        n_mels, n_frames = spectrogram.shape
        if n_mels < self.freq_mask_param:
            return spectrogram
        
        # Random mask width
        mask_width = np.random.randint(1, self.freq_mask_param + 1)
        # Random start position
        mask_start = np.random.randint(0, n_mels - mask_width + 1)
        
        # Apply mask (set to zero)
        spectrogram_masked = spectrogram.copy()
        spectrogram_masked[mask_start:mask_start + mask_width, :] = 0
        return spectrogram_masked
    
    def __len__(self):
        return len(self.file_list) * self.snippets_per_file
    
    def __getitem__(self, idx):
        """
        Returns a RANDOM snippet from a file.
        Different snippet positions are sampled on each call (strong data augmentation).
        
        Args:
            idx: Index that determines which file to sample from.
                 Multiple indices map to the same file (snippets_per_file times).
        """
        # Determine which file to sample from
        file_idx = idx // self.snippets_per_file
        file_info = self.file_list[file_idx]
        
        # RANDOM snippet position - different every time!
        start_frame = np.random.randint(0, file_info['max_start'] + 1)
        end_frame = start_frame + self.snippet_frames
        
        # Memory-efficient: Load only the required snippet using memory mapping
        spectrogram_mmap = np.load(file_info['spectrogram_path'], mmap_mode='r')
        spectrogram_snippet = np.array(spectrogram_mmap[:, start_frame:end_frame])  # Copy only the snippet
        del spectrogram_mmap  # Free the memory-mapped file
        
        # Apply data augmentation if enabled (training only)
        if self.augment:
            # 50% chance to apply time masking
            if np.random.rand() > 0.5:
                spectrogram_snippet = self._apply_time_mask(spectrogram_snippet)
            # 50% chance to apply frequency masking
            if np.random.rand() > 0.5:
                spectrogram_snippet = self._apply_freq_mask(spectrogram_snippet)
        
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
        optimizer: The optimizer to save state into
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
    
    # FILE-BASED TRAIN/VAL SPLIT (prevents data leakage)
    # First, get the list of all files
    all_files = sorted([f for f in os.listdir(data_dir) if f.endswith("_piano_roll_with_pedals.npz")])
    num_files = len(all_files)
    print(f"Total files found: {num_files}")
    
    # Split files into train/val (80/20)
    train_file_count = int(0.8 * num_files)
    val_file_count = num_files - train_file_count
    
    # Create indices for train and val files (deterministic with seed)
    np.random.seed(42)
    file_indices = np.random.permutation(num_files)
    train_file_indices = file_indices[:train_file_count].tolist()
    val_file_indices = file_indices[train_file_count:].tolist()
    np.random.seed(None)  # Reset seed
    
    print(f"\n📂 FILE-BASED SPLIT (prevents data leakage):")
    print(f"  Train files: {train_file_count} ({train_file_count/num_files*100:.1f}%)")
    print(f"  Val files:   {val_file_count} ({val_file_count/num_files*100:.1f}%)")
    print(f"  Train and validation sets use COMPLETELY DIFFERENT files!")
    
    # Create datasets with file-based filtering
    train_dataset = SnippetDataset(
        data_dir, 
        CONFIG['snippet_frames'], 
        snippets_per_file=CONFIG.get('snippets_per_file', 10),
        seed=42,
        file_indices=train_file_indices,
        augment=True,  # Enable augmentation for training
        time_mask_param=CONFIG.get('time_mask_param', 30),
        freq_mask_param=CONFIG.get('freq_mask_param', 20)
    )
    val_dataset = SnippetDataset(
        data_dir, 
        CONFIG['snippet_frames'], 
        snippets_per_file=CONFIG.get('snippets_per_file', 10),
        seed=42,
        file_indices=val_file_indices,
        augment=False  # No augmentation for validation
    )
    
    print(f"\n📊 DATASET SUMMARY:")
    print(f"  Train snippets: {len(train_dataset)}")
    print(f"  Val snippets:   {len(val_dataset)}")
    print(f"  Total snippets: {len(train_dataset) + len(val_dataset)}")
    
    # Optimized DataLoader settings for faster training
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=CONFIG['num_workers'],
        pin_memory=CONFIG['pin_memory'],
        prefetch_factor=CONFIG['prefetch_factor'] if CONFIG['num_workers'] > 0 else None,
        persistent_workers=CONFIG['persistent_workers'] if CONFIG['num_workers'] > 0 else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=CONFIG['num_workers'],
        pin_memory=CONFIG['pin_memory'],
        prefetch_factor=CONFIG['prefetch_factor'] if CONFIG['num_workers'] > 0 else None,
        persistent_workers=CONFIG['persistent_workers'] if CONFIG['num_workers'] > 0 else False
    )
    
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
    
    # Loss function - use Focal Loss for imbalanced data or BCE for standard training
    if CONFIG.get('use_focal_loss', True):
        criterion = FocalLoss(alpha=CONFIG['focal_alpha'], gamma=CONFIG['focal_gamma'])
        print(f"Using Focal Loss (alpha={CONFIG['focal_alpha']}, gamma={CONFIG['focal_gamma']}) to handle class imbalance")
    else:
        criterion = nn.BCELoss()
        print("Using standard BCE Loss")
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=CONFIG.get('weight_decay', 1e-5))
    
    # Learning rate scheduler
    scheduler = None
    if CONFIG.get('use_lr_scheduler', False):
        if CONFIG['scheduler_type'] == 'reduce_on_plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min',
                factor=CONFIG['scheduler_factor'],
                patience=CONFIG['scheduler_patience'],
                min_lr=CONFIG['scheduler_min_lr'],
                threshold=CONFIG.get('scheduler_threshold', 1e-4),
                threshold_mode='rel'
            )
            print(f"Using ReduceLROnPlateau scheduler (patience={CONFIG['scheduler_patience']}, factor={CONFIG['scheduler_factor']}, threshold={CONFIG.get('scheduler_threshold', 1e-4)})")
        elif CONFIG['scheduler_type'] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=num_epochs,
                eta_min=CONFIG['scheduler_min_lr']
            )
            print(f"Using CosineAnnealingLR scheduler")
        elif CONFIG['scheduler_type'] == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=30,
                gamma=CONFIG['scheduler_factor']
            )
            print(f"Using StepLR scheduler")
    
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
        
        # Update learning rate scheduler
        if scheduler is not None:
            if CONFIG['scheduler_type'] == 'reduce_on_plateau':
                scheduler.step(val_loss)  # ReduceLROnPlateau needs the metric
            else:
                scheduler.step()  # Other schedulers don't need metrics
            
            # Print current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Current Learning Rate: {current_lr:.2e}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")
        
        # Save checkpoint every epoch
        save_checkpoint(epoch, model, optimizer, train_losses, val_losses, 
                        train_precisions, val_precisions, train_recalls, val_recalls,
                        train_f1s, val_f1s, best_val_loss, checkpoint_save_path)
        
        # Save training curves every 1 epochs (overwrite the same file)
        if (epoch + 1) % 1 == 0:
            plot_training_curves(
                train_losses, val_losses,
                train_precisions, val_precisions,
                train_recalls, val_recalls,
                train_f1s, val_f1s,
                curves_save_path  # Overwrite the same file
            )
        
        # Visualize predictions every 50 epochs
        if (epoch + 1) % 50 == 0:
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