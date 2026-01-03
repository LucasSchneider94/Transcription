"""
Inference script for piano transcription model with onset + duration prediction.
Visualizes ground truth vs model predictions.
"""

import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt
import os
import pretty_midi
import json
from pathlib import Path

from model import PianoTranscriptionModel
from config import CONFIG, MIN_PITCH
from data_preparation import create_piano_roll_with_onsets_durations, DURATION_BINS, NUM_DURATION_BINS, get_duration_bin_label
from inference_config import INFERENCE_CONFIG
from dataset import duration_to_log_duration, log_duration_to_duration


def load_model(model_path, device):
    """Load trained model from checkpoint."""
    model_dir = os.path.dirname(model_path)
    config_path = os.path.join(model_dir, "config.json")
    
    # Load config if available
    model_config = CONFIG.copy()
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
            model_config.update(saved_config)
            print(f"✓ Loaded config from {config_path}")
    else:
        print("⚠️  No config.json found, using default CONFIG")
    
    # Create model
    model = PianoTranscriptionModel(
        input_features=model_config.get('n_mels', 352),
        num_keys=model_config.get('num_keys', 88),
        transformer_dim=model_config.get('hidden_size', 256),
        num_heads=model_config.get('num_heads', 8),
        num_layers=model_config.get('num_layers', 4),
        dropout=model_config.get('dropout', 0.2),
        duration_mode=model_config.get('duration_mode', 'log'),
        num_duration_bins=model_config.get('num_duration_bins', 8)
    ).to(device)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"✓ Model loaded from {model_path}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Duration mode: {model_config.get('duration_mode', 'log')}")
    
    return model, model_config


def preprocess_audio_to_spectrogram(audio_path, config):
    """
    Convert audio to Mel-spectrogram using same parameters as training.
    
    Args:
        audio_path (str): Path to audio file
        config (dict): Model config with spectrogram parameters
        
    Returns:
        np.ndarray: Mel-spectrogram of shape (n_mels, time_frames)
    """
    print(f"Computing spectrogram...")
    
    # Load audio
    audio, _ = librosa.load(audio_path, sr=config['sample_rate'])
    
    # Compute Mel-spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio, 
        sr=config['sample_rate'], 
        n_fft=config['n_fft'], 
        hop_length=config['hop_length'], 
        n_mels=config['n_mels']
    )
    
    # Convert to log scale
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    print(f"  Spectrogram shape: {log_mel_spectrogram.shape}")
    
    return log_mel_spectrogram


def extract_time_range(spectrogram, labels, start_time, end_time, fps, hop_length, sample_rate):
    """Extract a time range from spectrogram and labels."""
    # Calculate frame indices for spectrogram
    spec_start_frame = int(start_time * sample_rate / hop_length)
    spec_end_frame = int(end_time * sample_rate / hop_length)
    
    # Extract spectrogram snippet
    spectrogram_snippet = spectrogram[:, spec_start_frame:spec_end_frame]
    
    # Extract label snippets
    labels_snippet = None
    if labels is not None:
        label_start_frame = int(start_time * fps)
        label_end_frame = int(end_time * fps)
        labels_snippet = {
            'onset': labels['onset'][label_start_frame:label_end_frame, :],
            'duration': labels['duration'][label_start_frame:label_end_frame, :],
            'frame': labels['frame'][label_start_frame:label_end_frame, :],
            'pedal': labels['pedal'][label_start_frame:label_end_frame, :]
        }
    
    return spectrogram_snippet, labels_snippet


def run_inference(model, spectrogram, device, config):
    """
    Run inference on spectrogram.
    
    Args:
        model: Trained model
        spectrogram (np.ndarray): Spectrogram (n_mels, time_frames)
        device: torch device
        config: Model config
        
    Returns:
        Dictionary with 'onset', 'duration', 'frame' predictions
    """
    print(f"Running inference...")
    
    # Convert to tensor
    spec_tensor = torch.FloatTensor(spectrogram).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, n_mels, time)
    
    # Run inference
    with torch.no_grad():
        predictions = model(spec_tensor)
    
    # Process predictions based on duration mode
    onset_pred = torch.sigmoid(predictions['onset']).squeeze(0).cpu().numpy()  # (time, 88)
    frame_pred = torch.sigmoid(predictions['frame']).squeeze(0).cpu().numpy()  # (time, 88)
    
    duration_mode = config.get('duration_mode', 'log')
    
    if duration_mode == 'bins':
        # Classification: take argmax over bins
        duration_logits = predictions['duration'].squeeze(0).cpu()  # (time, 88, num_bins)
        duration_bins = torch.argmax(duration_logits, dim=-1).numpy()  # (time, 88)
        # Convert bins to actual durations (use bin centers)
        duration_pred = np.zeros_like(duration_bins, dtype=float)
        for i in range(NUM_DURATION_BINS):
            mask = duration_bins == i
            bin_center = (DURATION_BINS[i] + DURATION_BINS[i+1]) / 2
            if np.isinf(DURATION_BINS[i+1]):
                bin_center = DURATION_BINS[i] + 1.0
            duration_pred[mask] = bin_center
    
    elif duration_mode == 'log':
        # Regression: convert log-duration back to duration
        log_duration_pred = predictions['duration'].squeeze(0).cpu().numpy()  # (time, 88)
        duration_pred = log_duration_to_duration(log_duration_pred)  # Convert back
    
    else:  # linear
        # Regression: use as-is
        duration_pred = predictions['duration'].squeeze(0).cpu().numpy()  # (time, 88)
    
    print(f"  Predictions shape: onset={onset_pred.shape}, duration={duration_pred.shape}, frame={frame_pred.shape}")
    
    return {
        'onset': onset_pred,
        'duration': duration_pred,
        'frame': frame_pred
    }


def visualize_comparison(ground_truth, predictions, save_path, start_time, end_time, fps=100):
    """
    Visualize ground truth vs predictions side by side.
    
    Args:
        ground_truth: Dict with 'onset', 'duration', 'frame', 'pedal' (or None)
        predictions: Dict with 'onset', 'duration', 'frame'
        save_path: Path to save visualization
        start_time: Start time in seconds
        end_time: End time in seconds
        fps: Frames per second
    """
    duration = end_time - start_time
    
    if ground_truth is not None:
        # Plot ground truth and predictions side by side - 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(18, 10))
        
        # Ground truth
        # Panel 1: Onset + Duration
        gt_onset = ground_truth['onset']
        gt_duration = ground_truth['duration']
        duration_binned = np.zeros_like(gt_duration)
        for i in range(NUM_DURATION_BINS):
            mask = (gt_duration >= DURATION_BINS[i]) & (gt_duration < DURATION_BINS[i+1])
            duration_binned[mask] = i
        onset_duration_combined = duration_binned.astype(float).copy()
        onset_duration_combined[gt_onset == 0] = 0
        onset_duration_combined[gt_onset == 1] = duration_binned[gt_onset == 1] + 1
        
        im1 = axes[0, 0].imshow(onset_duration_combined.T, aspect='auto', origin='lower', 
                                cmap='hot', interpolation='nearest', vmin=0, vmax=NUM_DURATION_BINS+1)
        axes[0, 0].set_title(f'Ground Truth: Onsets + Duration', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Piano Keys (88)', fontsize=11)
        cbar1 = plt.colorbar(im1, ax=axes[0, 0], ticks=[0] + list(range(1, NUM_DURATION_BINS+1)))
        tick_labels = ['Silence'] + [get_duration_bin_label(i) for i in range(NUM_DURATION_BINS)]
        cbar1.ax.set_yticklabels(tick_labels, fontsize=8)
        
        # Panel 2: Frame (FIXED: Added vmin=0, vmax=1)
        im_frame_gt = axes[0, 1].imshow(ground_truth['frame'].T, aspect='auto', origin='lower', 
                                         cmap='hot', interpolation='nearest', vmin=0, vmax=1)
        axes[0, 1].set_title(f'Ground Truth: Frame (Active Notes)', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Piano Keys (88)', fontsize=11)
        plt.colorbar(im_frame_gt, ax=axes[0, 1])
        
        # Predictions
        # Panel 3: Onset + Duration
        pred_onset = predictions['onset']
        pred_duration = predictions['duration']
        pred_duration_binned = np.zeros_like(pred_duration)
        for i in range(NUM_DURATION_BINS):
            mask = (pred_duration >= DURATION_BINS[i]) & (pred_duration < DURATION_BINS[i+1])
            pred_duration_binned[mask] = i
        pred_onset_duration_combined = pred_duration_binned.astype(float).copy()
        pred_onset_duration_combined[pred_onset < 0.5] = 0
        pred_onset_duration_combined[pred_onset >= 0.5] = pred_duration_binned[pred_onset >= 0.5] + 1
        
        im2 = axes[1, 0].imshow(pred_onset_duration_combined.T, aspect='auto', origin='lower', 
                                cmap='hot', interpolation='nearest', vmin=0, vmax=NUM_DURATION_BINS+1)
        axes[1, 0].set_title(f'Predicted: Onsets + Duration', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Piano Keys (88)', fontsize=11)
        axes[1, 0].set_xlabel('Time Frames', fontsize=11)
        cbar2 = plt.colorbar(im2, ax=axes[1, 0], ticks=[0] + list(range(1, NUM_DURATION_BINS+1)))
        cbar2.ax.set_yticklabels(tick_labels, fontsize=8)
        
        # Panel 4: Frame (FIXED: Added vmin=0, vmax=1)
        im_frame_pred = axes[1, 1].imshow(predictions['frame'].T, aspect='auto', origin='lower', 
                                           cmap='hot', interpolation='nearest', vmin=0, vmax=1)
        axes[1, 1].set_title(f'Predicted: Frame (Active Notes)', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Piano Keys (88)', fontsize=11)
        axes[1, 1].set_xlabel('Time Frames', fontsize=11)
        plt.colorbar(im_frame_pred, ax=axes[1, 1])
        
    else:
        # Plot only predictions
        fig, axes = plt.subplots(1, 2, figsize=(18, 5))
        
        # Panel 1: Onset + Duration
        pred_onset = predictions['onset']
        pred_duration = predictions['duration']
        pred_duration_binned = np.zeros_like(pred_duration)
        for i in range(NUM_DURATION_BINS):
            mask = (pred_duration >= DURATION_BINS[i]) & (pred_duration < DURATION_BINS[i+1])
            pred_duration_binned[mask] = i
        pred_onset_duration_combined = pred_duration_binned.astype(float).copy()
        pred_onset_duration_combined[pred_onset < 0.5] = 0
        pred_onset_duration_combined[pred_onset >= 0.5] = pred_duration_binned[pred_onset >= 0.5] + 1
        
        im = axes[0].imshow(pred_onset_duration_combined.T, aspect='auto', origin='lower', 
                           cmap='hot', interpolation='nearest', vmin=0, vmax=NUM_DURATION_BINS+1)
        axes[0].set_title(f'Predicted: Onsets + Duration', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Piano Keys (88)', fontsize=11)
        axes[0].set_xlabel('Time Frames', fontsize=11)
        cbar = plt.colorbar(im, ax=axes[0], ticks=[0] + list(range(1, NUM_DURATION_BINS+1)))
        tick_labels = ['Silence'] + [get_duration_bin_label(i) for i in range(NUM_DURATION_BINS)]
        cbar.ax.set_yticklabels(tick_labels, fontsize=8)
        
        # Panel 2: Frame (FIXED: Added vmin=0, vmax=1)
        im_frame = axes[1].imshow(predictions['frame'].T, aspect='auto', origin='lower', 
                                   cmap='hot', interpolation='nearest', vmin=0, vmax=1)
        axes[1].set_title(f'Predicted: Frame (Active Notes)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Piano Keys (88)', fontsize=11)
        axes[1].set_xlabel('Time Frames', fontsize=11)
        plt.colorbar(im_frame, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✓ Visualization saved to {save_path}")


def main():
    """Main inference function."""
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                         'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    os.makedirs(INFERENCE_CONFIG['output_dir'], exist_ok=True)
    
    # Load model
    model, model_config = load_model(INFERENCE_CONFIG['model_path'], device)
    
    # Process audio
    print(f"\n{'='*80}")
    print(f"Processing audio: {INFERENCE_CONFIG['audio_path']}")
    print(f"{'='*80}")
    full_spectrogram = preprocess_audio_to_spectrogram(
        INFERENCE_CONFIG['audio_path'],
        model_config
    )
    
    # Process MIDI if available
    full_labels = None
    if INFERENCE_CONFIG['midi_path'] is not None and os.path.exists(INFERENCE_CONFIG['midi_path']):
        print(f"\nProcessing MIDI: {INFERENCE_CONFIG['midi_path']}")
        full_labels = create_piano_roll_with_onsets_durations(
            INFERENCE_CONFIG['midi_path'],
            fps=model_config['roll_fps'],
            onset_frames=model_config.get('onset_frames', 2)
        )
        print(f"  Labels shape: onset={full_labels['onset'].shape}")
    
    # Extract time range
    start_time = INFERENCE_CONFIG['start_time']
    end_time = INFERENCE_CONFIG['end_time']
    print(f"\n{'='*80}")
    print(f"Extracting time range: {start_time}s - {end_time}s")
    print(f"{'='*80}")
    
    spectrogram_snippet, labels_snippet = extract_time_range(
        full_spectrogram,
        full_labels,
        start_time,
        end_time,
        model_config['roll_fps'],
        model_config['hop_length'],
        model_config['sample_rate']
    )
    
    print(f"  Spectrogram snippet shape: {spectrogram_snippet.shape}")
    if labels_snippet is not None:
        print(f"  Labels snippet shape: onset={labels_snippet['onset'].shape}")
    
    # Run inference
    print(f"\n{'='*80}")
    predictions = run_inference(model, spectrogram_snippet, device, model_config)
    print(f"{'='*80}")
    
    # Visualize
    output_filename = os.path.join(
        INFERENCE_CONFIG['output_dir'],
        f"inference_{start_time:.1f}s_{end_time:.1f}s.png"
    )
    
    visualize_comparison(
        labels_snippet,
        predictions,
        output_filename,
        start_time,
        end_time,
        model_config['roll_fps']
    )
    
    print(f"\n✓ Inference complete!")


if __name__ == "__main__":
    main()
