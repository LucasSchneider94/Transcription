"""
Inference script for piano transcription model.
FIXED: Uses fresh 352-bin spectrograms and processes in 3-second chunks to match training.
"""

import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt
import os
import pretty_midi
import json

from model import PianoTranscriptionModel, PianoTranscriptionModelLegacy
from config import CONFIG, NUM_OUTPUTS, MIN_PITCH
from data_preparation import build_binary_piano_roll_with_pedals
from inference_config import INFERENCE_CONFIG


def preprocess_audio_to_spectrogram_inference(audio_path, n_mels, n_fft, hop_length, sample_rate):
    """
    Convert audio to Mel-spectrogram using the EXACT same parameters as training.
    Uses fresh computation with current config settings.
    
    Args:
        audio_path (str): Path to audio file
        n_mels (int): Number of mel bins
        n_fft (int): FFT window size
        hop_length (int): Hop length
        sample_rate (int): Sample rate
        
    Returns:
        np.ndarray: Mel-spectrogram of shape (n_mels, time_frames)
    """
    print(f"Computing fresh spectrogram with:")
    print(f"  n_mels: {n_mels}")
    print(f"  n_fft: {n_fft}")
    print(f"  hop_length: {hop_length}")
    print(f"  sample_rate: {sample_rate}")
    
    # Load audio
    audio, _ = librosa.load(audio_path, sr=sample_rate)
    
    # Compute Mel-spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    
    # Convert to log scale - SAME AS TRAINING
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    return log_mel_spectrogram


def extract_time_range(spectrogram, piano_roll, start_time, end_time, fps, hop_length, sample_rate):
    """Extract a time range from spectrogram and piano roll."""
    # Calculate frame indices for spectrogram (based on hop_length)
    spec_start_frame = int(start_time * sample_rate / hop_length)
    spec_end_frame = int(end_time * sample_rate / hop_length)
    
    # Extract spectrogram snippet
    spectrogram_snippet = spectrogram[:, spec_start_frame:spec_end_frame]
    
    # Extract piano roll snippet if available
    piano_roll_snippet = None
    if piano_roll is not None:
        roll_start_frame = int(start_time * fps)
        roll_end_frame = int(end_time * fps)
        piano_roll_snippet = piano_roll[roll_start_frame:roll_end_frame, :]
    
    return spectrogram_snippet, piano_roll_snippet


def run_inference_chunked(model, spectrogram, device, chunk_frames=300):
    """
    Run inference in chunks (like training) to avoid temporal degradation.
    
    Args:
        model: Trained model
        spectrogram (np.ndarray): Full spectrogram (n_mels, time_frames)
        device: torch device
        chunk_frames (int): Number of frames per chunk (should match training snippet_frames)
        
    Returns:
        np.ndarray: Model predictions of shape (time_frames, num_outputs)
    """
    n_mels, total_frames = spectrogram.shape
    predictions = []
    
    print(f"Running inference in {chunk_frames}-frame chunks...")
    
    # Process in chunks with overlap to avoid edge effects
    overlap = chunk_frames // 4  # 25% overlap
    stride = chunk_frames - overlap
    
    for start_idx in range(0, total_frames, stride):
        end_idx = min(start_idx + chunk_frames, total_frames)
        
        # Extract chunk
        chunk = spectrogram[:, start_idx:end_idx]
        
        # Pad if needed
        if chunk.shape[1] < chunk_frames:
            padding = chunk_frames - chunk.shape[1]
            chunk = np.pad(chunk, ((0, 0), (0, padding)), mode='constant', constant_values=0)
        
        # Normalize chunk individually (LIKE TRAINING)
        chunk_normalized = librosa.power_to_db(
            librosa.db_to_power(chunk), 
            ref=np.max(librosa.db_to_power(chunk))
        )
        
        # Convert to tensor
        chunk_tensor = torch.tensor(chunk_normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        chunk_tensor = chunk_tensor.to(device)
        
        # Run inference
        with torch.no_grad():
            chunk_pred = model(chunk_tensor)[0].cpu().numpy()  # (chunk_frames, num_outputs)
        
        # Take only non-overlapping part (or trim padding)
        if start_idx + stride >= total_frames:
            # Last chunk - take what we need
            take_frames = total_frames - start_idx
            predictions.append(chunk_pred[:take_frames])
        else:
            # Regular chunk - take non-overlapping part
            predictions.append(chunk_pred[:stride])
    
    return np.concatenate(predictions, axis=0)


def run_inference(model, spectrogram, device):
    """
    Run inference on a spectrogram snippet (single pass).
    
    Args:
        model: Trained model
        spectrogram (np.ndarray): Spectrogram of shape (n_mels, time_frames)
        device: torch device
        
    Returns:
        np.ndarray: Model predictions of shape (time_frames, num_outputs)
    """
    # Convert to tensor and add batch dimension
    spectrogram_tensor = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    spectrogram_tensor = spectrogram_tensor.to(device)
    
    # Run inference
    with torch.no_grad():
        predictions = model(spectrogram_tensor)[0].cpu().numpy()
    
    return predictions


def load_model(model_path, device):
    """Load model - automatically detects legacy vs new architecture."""
    model_dir = os.path.dirname(model_path)
    config_path = os.path.join(model_dir, "config.json")
    
    use_legacy = False
    model_config = CONFIG.copy()
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
            if 'preload_into_ram' not in saved_config and 'use_cnn_only' not in saved_config:
                use_legacy = True
                print("⚠️  Detected LEGACY model (training_run_001-008)")
            model_config.update(saved_config)
    else:
        print("⚠️  No config.json found, assuming legacy model")
        use_legacy = True
    
    # Create appropriate model
    if use_legacy:
        print("Loading with PianoTranscriptionModelLegacy...")
        model = PianoTranscriptionModelLegacy(
            n_mels=model_config.get('n_mels', 352),
            hidden_size=model_config.get('hidden_size', 256),
            num_heads=model_config.get('num_heads', 8),
            num_layers=model_config.get('num_layers', 4),
            num_outputs=NUM_OUTPUTS,
            dropout=model_config.get('dropout', 0.2)
        ).to(device)
    else:
        print("Loading with PianoTranscriptionModel (new architecture)...")
        model = PianoTranscriptionModel(
            n_mels=model_config.get('n_mels', 352),
            hidden_size=model_config.get('hidden_size', 256),
            num_heads=model_config.get('num_heads', 8),
            num_layers=model_config.get('num_layers', 4),
            num_outputs=NUM_OUTPUTS,
            dropout=model_config.get('dropout', 0.2)
        ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"✓ Model loaded from {model_path}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  n_mels: {model_config.get('n_mels', 352)}")
    
    return model, model_config


def visualize_results(predictions, ground_truth, save_path, start_time, end_time, plot_full_range=True):
    """
    Visualize model predictions vs ground truth.
    
    Args:
        predictions (np.ndarray): Model predictions (time_frames, num_outputs)
        ground_truth (np.ndarray): Ground truth piano roll (time_frames, num_outputs) or None
        save_path (str): Path to save the visualization
        start_time (float): Start time in seconds
        end_time (float): End time in seconds
        plot_full_range (bool): If True, plot all 91 outputs; if False, focus on active range
    """
    duration = end_time - start_time
    
    if ground_truth is not None:
        # Plot both ground truth and predictions
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        
        axes[0].imshow(ground_truth.T, aspect='auto', origin='lower', cmap='hot', interpolation='nearest')
        axes[0].set_title(f'Ground Truth ({start_time:.1f}s - {end_time:.1f}s)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Time Frames', fontsize=12)
        axes[0].set_ylabel('Piano Keys + Pedals', fontsize=12)
        
        # Add horizontal lines to separate keys from pedals
        axes[0].axhline(y=87.5, color='cyan', linestyle='--', linewidth=1, alpha=0.7, label='Keys/Pedals boundary')
        axes[0].legend(loc='upper right')
        
        axes[1].imshow(predictions.T, aspect='auto', origin='lower', cmap='hot', interpolation='nearest')
        axes[1].set_title(f'Model Predictions ({start_time:.1f}s - {end_time:.1f}s)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Time Frames', fontsize=12)
        axes[1].set_ylabel('Piano Keys + Pedals', fontsize=12)
        
        # Add horizontal lines to separate keys from pedals
        axes[1].axhline(y=87.5, color='cyan', linestyle='--', linewidth=1, alpha=0.7, label='Keys/Pedals boundary')
        axes[1].legend(loc='upper right')
        
    else:
        # Plot only predictions
        fig, ax = plt.subplots(1, 1, figsize=(16, 6))
        
        ax.imshow(predictions.T, aspect='auto', origin='lower', cmap='hot', interpolation='nearest')
        ax.set_title(f'Model Predictions ({start_time:.1f}s - {end_time:.1f}s)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time Frames', fontsize=12)
        ax.set_ylabel('Piano Keys + Pedals', fontsize=12)
        
        # Add horizontal lines to separate keys from pedals
        ax.axhline(y=87.5, color='cyan', linestyle='--', linewidth=1, alpha=0.7, label='Keys/Pedals boundary')
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Visualization saved to {save_path}")


def calculate_metrics(predictions, ground_truth, threshold=0.5):
    """
    Calculate precision, recall, and F1 score.
    
    Args:
        predictions (np.ndarray): Model predictions (time_frames, num_outputs)
        ground_truth (np.ndarray): Ground truth (time_frames, num_outputs)
        threshold (float): Threshold for binary classification
        
    Returns:
        dict: Metrics dictionary
    """
    pred_binary = (predictions > threshold).astype(np.float32)
    
    tp = np.sum(pred_binary * ground_truth)
    fp = np.sum(pred_binary * (1 - ground_truth))
    fn = np.sum((1 - pred_binary) * ground_truth)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def main():
    """Main inference function with fresh spectrogram computation."""
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                         'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    os.makedirs(INFERENCE_CONFIG['output_dir'], exist_ok=True)
    
    # Load model and get its config
    model, model_config = load_model(INFERENCE_CONFIG['model_path'], device)
    
    # Use model's config for spectrogram parameters
    n_mels = model_config.get('n_mels', 352)
    n_fft = model_config.get('n_fft', 4096)
    hop_length = model_config.get('hop_length', 480)
    sample_rate = model_config.get('sample_rate', 48000)
    snippet_frames = model_config.get('snippet_frames', 300)
    
    # Compute FRESH spectrogram with correct parameters
    print(f"\n{'='*80}")
    print(f"Processing audio: {INFERENCE_CONFIG['audio_path']}")
    print(f"{'='*80}")
    full_spectrogram = preprocess_audio_to_spectrogram_inference(
        INFERENCE_CONFIG['audio_path'],
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        sample_rate=sample_rate
    )
    print(f"Spectrogram shape: {full_spectrogram.shape}")
    
    # Process MIDI to piano roll if available
    full_piano_roll = None
    if INFERENCE_CONFIG['midi_path'] is not None and os.path.exists(INFERENCE_CONFIG['midi_path']):
        print(f"\nProcessing MIDI: {INFERENCE_CONFIG['midi_path']}")
        midi_data = pretty_midi.PrettyMIDI(INFERENCE_CONFIG['midi_path'])
        full_piano_roll = build_binary_piano_roll_with_pedals(midi_data, roll_fps=CONFIG['roll_fps'])
        print(f"Piano roll shape: {full_piano_roll.shape}")
    
    # Extract time range
    start_time = INFERENCE_CONFIG['start_time']
    end_time = INFERENCE_CONFIG['end_time']
    print(f"\n{'='*80}")
    print(f"Extracting time range: {start_time}s - {end_time}s")
    print(f"{'='*80}")
    
    spectrogram_snippet, piano_roll_snippet = extract_time_range(
        full_spectrogram,
        full_piano_roll,
        start_time,
        end_time,
        CONFIG['roll_fps'],
        hop_length,
        sample_rate
    )
    
    print(f"Spectrogram snippet shape: {spectrogram_snippet.shape}")
    if piano_roll_snippet is not None:
        print(f"Piano roll snippet shape: {piano_roll_snippet.shape}")
    
    # Run inference in chunks (like training) to avoid temporal degradation
    print(f"\n{'='*80}")
    print(f"Running inference...")
    print(f"{'='*80}")
    predictions = run_inference_chunked(
        model, 
        spectrogram_snippet, 
        device, 
        chunk_frames=snippet_frames
    )
    print(f"Predictions shape: {predictions.shape}")
    
    # Calculate metrics
    if piano_roll_snippet is not None:
        metrics = calculate_metrics(predictions, piano_roll_snippet)
        print(f"\n{'='*80}")
        print(f"METRICS")
        print(f"{'='*80}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"{'='*80}")
    
    # Visualize
    output_filename = os.path.join(
        INFERENCE_CONFIG['output_dir'],
        f"inference_{start_time:.1f}s_{end_time:.1f}s_chunked.png"
    )
    
    visualize_results(
        predictions,
        piano_roll_snippet,
        output_filename,
        start_time,
        end_time,
        plot_full_range=INFERENCE_CONFIG['plot_full_range']
    )
    
    print(f"\n✓ Inference complete!")


if __name__ == "__main__":
    main()
