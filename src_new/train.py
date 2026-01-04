"""
Training script for piano transcription model.
Supports onset + duration prediction with multiple modes (bins/log/linear).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import json
from tqdm import tqdm

from config import CONFIG, NUM_OUTPUTS
# For overfitting test, uncomment the line below and comment out the line above:
#from config_overfit import CONFIG_OVERFIT as CONFIG, NUM_OUTPUTS

from model import PianoTranscriptionModel, PianoTranscriptionModelCNNOnly, PianoTranscriptionModelUNet
from dataset import PianoTranscriptionDataset, duration_to_log_duration, log_duration_to_duration
from utils import (
    get_next_run_folder,
    compute_metrics,
    visualize_predictions,
    plot_training_curves,
    save_checkpoint,
    load_checkpoint
)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in binary classification.
    
    FL = -α * (1-p_t)^γ * log(p_t)
    
    where:
        p_t = p if y=1, else (1-p)
        α = balancing factor for positive/negative classes
        γ = focusing parameter (higher = more focus on hard examples)
    
    Reference: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: Balancing factor for positive class (0 to 1)
                   - Use higher α (0.75-0.9) when positives are very rare
                   - Use lower α (0.25) when positives are common
            gamma: Focusing parameter (typically 2.0)
                   - γ=0: equivalent to standard BCE
                   - γ=2: balanced (recommended)
                   - γ=5: aggressive focusing on hard examples
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, ...) raw logits (before sigmoid)
            targets: (N, ...) binary labels (0 or 1)
        
        Returns:
            Focal loss value
        """
        # Get probabilities
        p = torch.sigmoid(inputs)
        
        # Calculate standard BCE loss (without reduction)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Calculate p_t: probability of correct class
        p_t = p * targets + (1 - p) * (1 - targets)
        
        # Calculate focal weight: (1 - p_t)^gamma
        # This down-weights easy examples (high p_t) and focuses on hard examples (low p_t)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha balancing
        # alpha for positive class, (1-alpha) for negative class
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Combine everything
        loss = alpha_t * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class OnsetDurationLoss(nn.Module):
    """
    Multi-task loss for onset, duration, and frame prediction.
    Supports pos_weight annealing for onset and frame losses.
    Supports Gaussian smoothing for onset targets (sigma annealing).
    """
    def __init__(self, onset_weight=4.0, duration_weight=2.0, frame_weight=1.0, 
                 duration_mode='log',
                 frame_pos_weight=1.0, onset_pos_weight=1.0,
                 current_sigma=0.0):
        super(OnsetDurationLoss, self).__init__()
        self.onset_weight = onset_weight
        self.duration_weight = duration_weight
        self.frame_weight = frame_weight
        self.duration_mode = duration_mode
        self.current_sigma = current_sigma
        
        # Store pos_weights (will be updated each epoch)
        self.frame_pos_weight = frame_pos_weight
        self.onset_pos_weight = onset_pos_weight
        
        # Loss functions with pos_weight
        self.onset_loss_fn = None
        self.frame_loss_fn = None
        self.update_pos_weights(frame_pos_weight, onset_pos_weight)
        
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.mse = nn.MSELoss(reduction='none')
    
    def update_pos_weights(self, frame_pos_weight, onset_pos_weight):
        """Update pos_weights for onset and frame losses."""
        self.frame_pos_weight = frame_pos_weight
        self.onset_pos_weight = onset_pos_weight
        
        # Recreate loss functions with new pos_weights
        self.frame_loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([frame_pos_weight]),
            reduction='mean'
        )
        self.onset_loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([onset_pos_weight]),
            reduction='mean'
        )

    def update_sigma(self, sigma):
        """Update sigma for Gaussian smoothing."""
        self.current_sigma = sigma

    def apply_gaussian_smoothing(self, targets, sigma):
        """
        Apply Gaussian smoothing to binary targets along time dimension.
        Args:
            targets: (Batch, Time, 88)
            sigma: Standard deviation in frames
        Returns:
            Smoothed targets (Batch, Time, 88)
        """
        if sigma <= 0.1:
            return targets
            
        B, T, K = targets.shape
        
        # Create kernel
        kernel_size = int(6 * sigma) + 1
        if kernel_size % 2 == 0: kernel_size += 1
        
        x = torch.arange(kernel_size, device=targets.device) - kernel_size // 2
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        # Normalize peak to 1.0 so onsets remain at 1.0 (soft labels)
        kernel = kernel / kernel.max()
        
        kernel = kernel.view(1, 1, -1) # (Out, In, Time) -> (1, 1, K)
        
        # Reshape targets for conv1d: (B*K, 1, T)
        targets_reshaped = targets.permute(0, 2, 1).reshape(B*K, 1, T)
        
        # Pad to maintain size
        pad = kernel_size // 2
        
        # Conv1d
        smoothed = F.conv1d(targets_reshaped, kernel, padding=pad)
        
        # Reshape back: (B*K, 1, T) -> (B, K, T) -> (B, T, K)
        smoothed = smoothed.view(B, K, T).permute(0, 2, 1)
        
        # Clamp to [0, 1]
        return torch.clamp(smoothed, 0, 1)
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Dict with 'onset', 'duration', 'frame' outputs
            targets: Dict with 'onset', 'duration', 'frame' labels
        
        Returns:
            Total weighted loss and individual losses
        """
        # Move pos_weight tensors to correct device
        self.frame_loss_fn.pos_weight = self.frame_loss_fn.pos_weight.to(predictions['frame'].device)
        self.onset_loss_fn.pos_weight = self.onset_loss_fn.pos_weight.to(predictions['onset'].device)
        
        # Onset loss with optional smoothing
        onset_targets = targets['onset']
        if self.current_sigma > 0.1:
            onset_targets = self.apply_gaussian_smoothing(onset_targets, self.current_sigma)
            
        onset_loss = self.onset_loss_fn(predictions['onset'], onset_targets)
        
        # Duration loss (masked)
        onset_mask = targets['onset'] > 0.5
        
        if self.duration_mode == 'bins':
            batch_size, time_steps, num_keys, num_bins = predictions['duration'].shape
            duration_pred_flat = predictions['duration'].reshape(-1, num_bins)
            duration_target_flat = targets['duration'].reshape(-1).long()
            onset_mask_flat = onset_mask.reshape(-1)
            
            ce_loss = self.ce(duration_pred_flat, duration_target_flat)
            
            if onset_mask_flat.sum() > 0:
                duration_loss = ce_loss[onset_mask_flat].mean()
            else:
                duration_loss = torch.tensor(0.0, device=predictions['onset'].device)
        
        else:
            mse_loss = self.mse(predictions['duration'], targets['duration'])
            
            if onset_mask.sum() > 0:
                duration_loss = mse_loss[onset_mask].mean()
            else:
                duration_loss = torch.tensor(0.0, device=predictions['onset'].device)
        
        # Frame loss
        frame_loss = self.frame_loss_fn(predictions['frame'], targets['frame'])
        
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
    """Create train and validation datasets and dataloaders using MAESTRO's official split."""
    
    # Load MAESTRO metadata
    maestro_json_path = config.get('maestro_json', './maestro-v3.0.0.json')
    if not os.path.exists(maestro_json_path):
        raise FileNotFoundError(f"MAESTRO JSON not found at: {maestro_json_path}")
    
    with open(maestro_json_path, 'r') as f:
        maestro_data = json.load(f)
    
    print(f"Loaded MAESTRO metadata from: {maestro_json_path}")
    
    # Get all processed files
    all_files = sorted([f for f in os.listdir(data_dir) 
                       if f.endswith("_piano_roll_with_pedals.npz")])
    num_files = len(all_files)
    print(f"Total processed files found: {num_files}")
    
    # Build mapping from audio filename base to split
    # MAESTRO filenames are like: "2018/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1.wav"
    # Our processed files are like: "MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1_piano_roll_with_pedals.npz"
    audio_to_split = {}
    for idx in maestro_data['audio_filename']:
        audio_path = maestro_data['audio_filename'][idx]
        split = maestro_data['split'][idx]
        # Extract base filename without extension and directory
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        audio_to_split[base_name] = split
    
    print(f"Loaded {len(audio_to_split)} entries from MAESTRO metadata")
    
    # Classify files by split
    train_files = []
    val_files = []
    test_files = []
    unmatched_files = []
    
    for file in all_files:
        # Extract base name from processed filename
        # Remove "_piano_roll_with_pedals.npz" suffix
        base_name = file.replace("_piano_roll_with_pedals.npz", "")
        
        if base_name in audio_to_split:
            split = audio_to_split[base_name]
            if split == 'train':
                train_files.append(file)
            elif split == 'validation':
                val_files.append(file)
            elif split == 'test':
                test_files.append(file)
            else:
                raise ValueError(f"Unknown split '{split}' for file: {file}")
        else:
            unmatched_files.append(file)
    
    # Fail early if we have unmatched files (as requested)
    if unmatched_files:
        raise ValueError(
            f"Found {len(unmatched_files)} files not in MAESTRO metadata. "
            f"First few: {unmatched_files[:5]}\n"
            f"This indicates a mismatch between processed files and MAESTRO dataset."
        )
    
    print(f"\nMAESTRO official split:")
    print(f"  Train:      {len(train_files)} files")
    print(f"  Validation: {len(val_files)} files")
    print(f"  Test:       {len(test_files)} files")
    
    # Apply data fraction if specified
    data_fraction = config.get('data_fraction', 1.0)
    if data_fraction < 1.0:
        print(f"\nApplying data fraction: {data_fraction*100:.1f}%")
        train_keep = max(1, int(len(train_files) * data_fraction))
        val_keep = max(1, int(len(val_files) * data_fraction))
        
        np.random.seed(42)
        train_files = sorted(np.random.choice(train_files, train_keep, replace=False).tolist())
        val_files = sorted(np.random.choice(val_files, val_keep, replace=False).tolist())
        np.random.seed(None)
        
        print(f"  Using Train:      {len(train_files)} files")
        print(f"  Using Validation: {len(val_files)} files")
    
    # Create file lists with full paths
    all_files_dict = {f: os.path.join(data_dir, f) for f in all_files}
    
    # Create index mappings for dataset
    train_indices = [all_files.index(f) for f in train_files]
    val_indices = [all_files.index(f) for f in val_files]
    
    # Create datasets with duration_mode from config
    train_dataset = PianoTranscriptionDataset(
        data_dir,
        config['snippet_bins'],
        snippets_per_file=config.get('snippets_per_file', 20),
        file_indices=train_indices,
        file_list=all_files,
        seed=42,
        fixed_snippets=config.get('fixed_snippets', False),
        preload_into_ram=config.get('preload_into_ram', True),
        duration_mode=config.get('duration_mode', 'log'),
        clip_duration=config.get('clip_duration_to_snippet', True)
    )
    
    val_dataset = PianoTranscriptionDataset(
        data_dir,
        config['snippet_bins'],
        snippets_per_file=config.get('snippets_per_file', 20),
        file_indices=val_indices,
        file_list=all_files,
        seed=42,
        fixed_snippets=config.get('fixed_snippets', False),
        preload_into_ram=config.get('preload_into_ram', True),
        duration_mode=config.get('duration_mode', 'log'),
        clip_duration=config.get('clip_duration_to_snippet', True)
    )
    
    print(f"\nDataset summary:")
    print(f"  Duration mode: {config.get('duration_mode', 'log')}")
    print(f"  Train snippets: {len(train_dataset)}")
    print(f"  Val snippets:   {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=config.get('shuffle_train', True),
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


def compute_onset_metrics(pred_onset, target_onset, tolerance_frames=5, threshold=0.5):
    """
    Compute note-level onset detection metrics with temporal tolerance.
    An onset is correctly detected if it's within ±tolerance_frames of a ground truth onset
    at the correct pitch.
    
    Args:
        pred_onset: (batch, time, 88) - onset predictions (logits or probabilities)
        target_onset: (batch, time, 88) - ground truth onsets (binary)
        tolerance_frames: temporal tolerance window (±frames), default 5 = ±50ms at 100fps
        threshold: prediction threshold for binarization
    
    Returns:
        dict with onset_precision, onset_recall, onset_f1, and counts (tp, fp, fn)
    """
    # Apply sigmoid if needed
    if pred_onset.min() < 0 or pred_onset.max() > 1:
        pred_onset = torch.sigmoid(pred_onset)
    
    # Binarize predictions
    pred_binary = (pred_onset > threshold).float()
    
    # Initialize counters
    tp = 0  # True positives
    fp = 0  # False positives
    fn = 0  # False negatives
    
    batch_size = pred_onset.shape[0]
    
    # Process each sample in batch
    for batch_idx in range(batch_size):
        # Process each pitch separately
        for pitch_idx in range(88):
            # Find ground truth onset frames for this pitch
            gt_frames = torch.where(target_onset[batch_idx, :, pitch_idx] > 0.5)[0]
            
            # Find predicted onset frames for this pitch
            pred_frames = torch.where(pred_binary[batch_idx, :, pitch_idx] > 0.5)[0]
            
            if len(gt_frames) == 0 and len(pred_frames) == 0:
                continue  # No onsets for this pitch, skip
            
            # Convert to numpy for easier processing
            gt_frames_np = gt_frames.cpu().numpy()
            pred_frames_np = pred_frames.cpu().numpy()
            
            # Match predictions to ground truth with tolerance
            matched_gt = set()
            matched_pred = set()
            
            for pred_frame in pred_frames_np:
                # Check if this prediction is within tolerance of any GT
                best_match = None
                best_distance = tolerance_frames + 1
                
                for gt_frame in gt_frames_np:
                    distance = abs(int(pred_frame) - int(gt_frame))
                    if distance <= tolerance_frames and distance < best_distance:
                        if gt_frame not in matched_gt:
                            best_match = gt_frame
                            best_distance = distance
                
                if best_match is not None:
                    matched_gt.add(best_match)
                    matched_pred.add(pred_frame)
                    tp += 1
            
            # Count unmatched predictions as false positives
            fp += len(pred_frames_np) - len(matched_pred)
            
            # Count unmatched ground truth as false negatives
            fn += len(gt_frames_np) - len(matched_gt)
    
    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'onset_precision': precision,
        'onset_recall': recall,
        'onset_f1': f1,
        'onset_tp': tp,
        'onset_fp': fp,
        'onset_fn': fn
    }


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_losses = {'onset_loss': 0, 'duration_loss': 0, 'frame_loss': 0, 'total_loss': 0}
    total_onset_metrics = {'onset_precision': 0, 'onset_recall': 0, 'onset_f1': 0}
    
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
        
        # Compute onset metrics only
        with torch.no_grad():
            onset_metrics = compute_onset_metrics(predictions['onset'], onset)
        
        # Accumulate losses and metrics
        for key in total_losses:
            total_losses[key] += loss_dict[key]
        for key in total_onset_metrics:
            total_onset_metrics[key] += onset_metrics[key]
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss_dict['total_loss']:.4f}",
            'onset_f1': f"{onset_metrics['onset_f1']:.3f}"
        })
    
    # Average losses and metrics
    num_batches = len(dataloader)
    for key in total_losses:
        total_losses[key] /= num_batches
    for key in total_onset_metrics:
        total_onset_metrics[key] /= num_batches
    
    return total_losses, total_onset_metrics


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_losses = {'onset_loss': 0, 'duration_loss': 0, 'frame_loss': 0, 'total_loss': 0}
    total_onset_metrics = {'onset_precision': 0, 'onset_recall': 0, 'onset_f1': 0}
    
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
            
            # Compute onset metrics only
            onset_metrics = compute_onset_metrics(predictions['onset'], onset)
            
            # Accumulate losses and metrics
            for key in total_losses:
                total_losses[key] += loss_dict[key]
            for key in total_onset_metrics:
                total_onset_metrics[key] += onset_metrics[key]
    
    # Average losses and metrics
    num_batches = len(dataloader)
    for key in total_losses:
        total_losses[key] /= num_batches
    for key in total_onset_metrics:
        total_onset_metrics[key] /= num_batches
    
    return total_losses, total_onset_metrics


def train(data_dir, run_folder, config):
    """Main training loop."""
    # Setup
    device = torch.device('mps' if torch.backends.mps.is_available() else
                         'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Saving to: {run_folder}\n")
    
    # Create datasets and loaders
    train_loader, val_loader = create_datasets_and_loaders(data_dir, config)
    
    # Compute initial pos_weights from data sparsity
    print("\nComputing pos_weights from data sparsity...")
    sample_batch = next(iter(train_loader))
    frame_ratio = sample_batch['frame'].mean().item()
    onset_ratio = sample_batch['onset'].mean().item()
    
    initial_frame_pos_weight = (1.0 - frame_ratio) / frame_ratio if frame_ratio > 0 else 1.0
    initial_onset_pos_weight = (1.0 - onset_ratio) / onset_ratio if onset_ratio > 0 else 1.0
    
    print(f"Frame sparsity: {frame_ratio*100:.2f}% → initial pos_weight: {initial_frame_pos_weight:.1f}")
    print(f"Onset sparsity: {onset_ratio*100:.2f}% → initial pos_weight: {initial_onset_pos_weight:.1f}")
    
    # Create model - choose based on config
    if config.get('use_unet', False):
        print("\n" + "="*80)
        print("Using U-Net Architecture")
        print("="*80 + "\n")
        model = PianoTranscriptionModelUNet(config).to(device)
    elif config.get('use_cnn_only', False):
        print("\n" + "="*80)
        print("ABLATION STUDY: Using CNN-only model (no Transformer)")
        print("="*80 + "\n")
        model = PianoTranscriptionModelCNNOnly(
            n_mels=config['n_mels'],
            hidden_size=config['hidden_size'],
            num_keys=config['num_keys'],
            dropout=config['dropout'],
            duration_mode=config.get('duration_mode', 'log'),
            num_duration_bins=config.get('num_duration_bins', 8)
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
    
    # Loss with initial pos_weights
    criterion = OnsetDurationLoss(
        onset_weight=config.get('onset_loss_weight', 4.0),
        duration_weight=config.get('duration_loss_weight', 2.0),
        frame_weight=config.get('frame_loss_weight', 1.0),
        duration_mode=config.get('duration_mode', 'log'),
        frame_pos_weight=initial_frame_pos_weight,
        onset_pos_weight=initial_onset_pos_weight,
        current_sigma=config.get('onset_tolerance_initial_sigma', 0.0)
    )
    
    print(f"\nLoss weights:")
    print(f"  Onset: {config.get('onset_loss_weight', 4.0)}")
    print(f"  Duration: {config.get('duration_loss_weight', 2.0)}")
    print(f"  Frame: {config.get('frame_loss_weight', 1.0)}")
    print(f"  Duration mode: {config.get('duration_mode', 'log')}")
    print(f"\nPos_weight annealing: epochs 0-50")
    
    # Sigma annealing setup
    initial_sigma = config.get('onset_tolerance_initial_sigma', 0.0)
    final_sigma = config.get('onset_tolerance_final_sigma', 0.0)
    sigma_anneal_epochs = config.get('onset_tolerance_anneal_epochs', 50)
    sigma_anneal_threshold = config.get('onset_tolerance_anneal_f1_threshold', 0.4)
    
    if initial_sigma > 0:
        print(f"Gaussian Onset Smoothing enabled:")
        print(f"  Initial sigma: {initial_sigma}")
        print(f"  Final sigma: {final_sigma}")
        print(f"  Annealing starts when F1 > {sigma_anneal_threshold}")
        print(f"  Annealing duration: {sigma_anneal_epochs} epochs")
    
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
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=run_folder)
    
    # Training loop
    print(f"\nStarting training from epoch {start_epoch + 1}\n")
    
    # Adaptive annealing state
    annealing_triggered = False
    annealing_start_epoch = None
    anneal_duration = 30
    
    # Sigma annealing state
    sigma_annealing_triggered = False
    sigma_annealing_start_epoch = None
    
    for epoch in range(start_epoch, config['num_epochs']):
        print(f"Epoch {epoch + 1}/{config['num_epochs']}")
        
        # Train and validate
        train_loss, train_metric = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metric = validate(model, val_loader, criterion, device)
        
        # Check if we should trigger pos_weight annealing
        if not annealing_triggered and train_metric['onset_f1'] >= 0.40:
            annealing_triggered = True
            annealing_start_epoch = epoch
            print(f"✓ Train onset F1 reached 40%, starting pos_weight annealing over next {anneal_duration} epochs")
            
        # Check if we should trigger sigma annealing
        if initial_sigma > 0 and not sigma_annealing_triggered and train_metric['onset_f1'] >= sigma_anneal_threshold:
            sigma_annealing_triggered = True
            sigma_annealing_start_epoch = epoch
            print(f"✓ Train onset F1 reached {sigma_anneal_threshold}, starting sigma annealing")
        
        # Update pos_weights with adaptive annealing schedule
        if not annealing_triggered:
            # Keep initial high weights
            current_frame_pos_weight = initial_frame_pos_weight
            current_onset_pos_weight = initial_onset_pos_weight
        elif (epoch - annealing_start_epoch) < anneal_duration:
            # Anneal to 1.0
            progress = (epoch - annealing_start_epoch) / anneal_duration
            current_frame_pos_weight = 1.0 + (initial_frame_pos_weight - 1.0) * (1.0 - progress)
            current_onset_pos_weight = 1.0 + (initial_onset_pos_weight - 1.0) * (1.0 - progress)
        else:
            # Fixed at 1.0
            current_frame_pos_weight = 1.0
            current_onset_pos_weight = 1.0
        
        criterion.update_pos_weights(current_frame_pos_weight, current_onset_pos_weight)
        
        # Update sigma
        current_sigma = initial_sigma
        if initial_sigma > 0:
            if sigma_annealing_triggered:
                progress = min(1.0, (epoch - sigma_annealing_start_epoch) / sigma_anneal_epochs)
                current_sigma = initial_sigma + (final_sigma - initial_sigma) * progress
            
            criterion.update_sigma(current_sigma)
        
        # Print pos_weights and sigma
        if epoch < 5 or (annealing_triggered and (epoch - annealing_start_epoch) < anneal_duration) or (sigma_annealing_triggered and (epoch - sigma_annealing_start_epoch) < sigma_anneal_epochs):
            status_str = f"Weights: frame={current_frame_pos_weight:.2f}, onset={current_onset_pos_weight:.2f}"
            if initial_sigma > 0:
                status_str += f", sigma={current_sigma:.2f}"
            print(status_str)
        
        train_losses.append(train_loss['total_loss'])
        val_losses.append(val_loss['total_loss'])
        train_metrics.append(train_metric)
        val_metrics.append(val_metric)
        
        # Log to TensorBoard
        # Losses
        writer.add_scalar('Loss/Train/Total', train_loss['total_loss'], epoch)
        writer.add_scalar('Loss/Train/Onset', train_loss['onset_loss'], epoch)
        writer.add_scalar('Loss/Train/Duration', train_loss['duration_loss'], epoch)
        writer.add_scalar('Loss/Train/Frame', train_loss['frame_loss'], epoch)
        
        writer.add_scalar('Loss/Val/Total', val_loss['total_loss'], epoch)
        writer.add_scalar('Loss/Val/Onset', val_loss['onset_loss'], epoch)
        writer.add_scalar('Loss/Val/Duration', val_loss['duration_loss'], epoch)
        writer.add_scalar('Loss/Val/Frame', val_loss['frame_loss'], epoch)
        
        # Metrics
        writer.add_scalar('Metrics/Train/Onset_F1', train_metric['onset_f1'], epoch)
        writer.add_scalar('Metrics/Train/Onset_Precision', train_metric['onset_precision'], epoch)
        writer.add_scalar('Metrics/Train/Onset_Recall', train_metric['onset_recall'], epoch)
        
        writer.add_scalar('Metrics/Val/Onset_F1', val_metric['onset_f1'], epoch)
        writer.add_scalar('Metrics/Val/Onset_Precision', val_metric['onset_precision'], epoch)
        writer.add_scalar('Metrics/Val/Onset_Recall', val_metric['onset_recall'], epoch)
        
        # Parameters
        writer.add_scalar('Params/Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Params/PosWeight/Frame', current_frame_pos_weight, epoch)
        writer.add_scalar('Params/PosWeight/Onset', current_onset_pos_weight, epoch)
        if initial_sigma > 0:
            writer.add_scalar('Params/Sigma', current_sigma, epoch)
        
        # Print stats
        print(f"Loss - Train: {train_loss['total_loss']:.4f}, Val: {val_loss['total_loss']:.4f}")
        print(f"Train - Onset: {train_loss['onset_loss']:.4f}, "
              f"Duration: {train_loss['duration_loss']:.4f}, Frame: {train_loss['frame_loss']:.4f}")
        print(f"Val   - Onset: {val_loss['onset_loss']:.4f}, "
              f"Duration: {val_loss['duration_loss']:.4f}, Frame: {val_loss['frame_loss']:.4f}")
        print(f"Train Onset - P: {train_metric['onset_precision']:.4f}, R: {train_metric['onset_recall']:.4f}, F1: {train_metric['onset_f1']:.4f}")
        print(f"Val   Onset - P: {val_metric['onset_precision']:.4f}, R: {val_metric['onset_recall']:.4f}, F1: {val_metric['onset_f1']:.4f}")
        
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
        if (epoch + 1) % 10 == 0:
            vis_path = os.path.join(run_folder, f"predictions_epoch_{epoch + 1}.png")
            visualize_predictions(model, val_loader, device, vis_path)
            
            # Log image to TensorBoard
            try:
                import matplotlib.image as mpimg
                if os.path.exists(vis_path):
                    img = mpimg.imread(vis_path)
                    # matplotlib reads as (H, W, C), TensorBoard wants (C, H, W)
                    # Also handle RGBA (4 channels) vs RGB (3 channels)
                    if img.ndim == 3:
                        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
                        writer.add_image('Predictions/Val', img_tensor, epoch)
            except Exception as e:
                print(f"Warning: Could not log image to TensorBoard: {e}")
        
        print()
    
    writer.close()


if __name__ == "__main__":
    data_dir = CONFIG['data_dir']
    run_folder = get_next_run_folder()
    
    print(f"Starting training run: {run_folder}")
    print(f"Data directory: {data_dir}\n")
    
    train(data_dir, run_folder, CONFIG)