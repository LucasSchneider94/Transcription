"""
Inference script for piano transcription model.
Loads a trained model and transcribes a specified segment of an audio file.
"""

import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt
import os
import pretty_midi

from model import PianoTranscriptionModel
from config import CONFIG, NUM_OUTPUTS, MIN_PITCH
from data_preparation import build_binary_piano_roll_with_pedals, preprocess_audio_to_spectrogram
from inference_config import INFERENCE_CONFIG


def load_model(model_path, device):
    """
    Load a trained model from a checkpoint.
    
    Args:
        model_path (str): Path to the model checkpoint (.pth file)
        device: torch device to load the model on
        
    Returns:
        model: Loaded model in eval mode
    """
    model = PianoTranscriptionModel(
        n_mels=CONFIG['n_mels'],
        hidden_size=CONFIG['hidden_size'],
        num_heads=CONFIG['num_heads'],
        num_layers=CONFIG['num_layers'],
        num_outputs=NUM_OUTPUTS,
        dropout=CONFIG['dropout']
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"Model loaded from {model_path}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def extract_time_range(spectrogram, piano_roll, start_time, end_time, fps, hop_length, sample_rate):
    """
    Extract a time range from spectrogram and piano roll.
    
    Args:
        spectrogram (np.ndarray): Full spectrogram (n_mels, time_frames)
        piano_roll (np.ndarray): Full piano roll (time_frames, num_outputs) or None
        start_time (float): Start time in seconds
        end_time (float): End time in seconds
        fps (float): Frames per second for piano roll
        hop_length (int): Hop length used for spectrogram
        sample_rate (int): Sample rate of audio
        
    Returns:
        tuple: (spectrogram_snippet, piano_roll_snippet or None)
    """
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


def run_inference(model, spectrogram, device):
    """
    Run inference on a spectrogram snippet.
    
    Args:
        model: Trained model
        spectrogram (np.ndarray): Spectrogram of shape (n_mels, time_frames)
        device: torch device
        
    Returns:
        np.ndarray: Model predictions of shape (time_frames, num_outputs)
    """
    # Convert to tensor and add batch dimension
    spectrogram_tensor = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, n_mels, time_frames)
    spectrogram_tensor = spectrogram_tensor.to(device)
    
    # Run inference
    with torch.no_grad():
        predictions = model(spectrogram_tensor)  # (1, time_frames, num_outputs)
    
    # Convert to numpy and remove batch dimension
    predictions = predictions.cpu().numpy()[0]  # (time_frames, num_outputs)
    
    return predictions


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
    """
    Main inference function.
    """
    # Setup device
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                         'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(INFERENCE_CONFIG['output_dir'], exist_ok=True)
    
    # Load model
    model = load_model(INFERENCE_CONFIG['model_path'], device)
    
    # Process audio to spectrogram
    print(f"\nProcessing audio: {INFERENCE_CONFIG['audio_path']}")
    full_spectrogram = preprocess_audio_to_spectrogram(
        INFERENCE_CONFIG['audio_path'],
        plot_path=None,
        save_path=None
    )
    print(f"Spectrogram shape: {full_spectrogram.shape}")
    
    # Process MIDI to piano roll if available
    full_piano_roll = None
    if INFERENCE_CONFIG['midi_path'] is not None and os.path.exists(INFERENCE_CONFIG['midi_path']):
        print(f"Processing MIDI: {INFERENCE_CONFIG['midi_path']}")
        midi_data = pretty_midi.PrettyMIDI(INFERENCE_CONFIG['midi_path'])
        full_piano_roll = build_binary_piano_roll_with_pedals(midi_data, roll_fps=CONFIG['roll_fps'])
        print(f"Piano roll shape: {full_piano_roll.shape}")
    else:
        print("No MIDI file provided - will only show predictions")
    
    # Extract time range
    start_time = INFERENCE_CONFIG['start_time']
    end_time = INFERENCE_CONFIG['end_time']
    print(f"\nExtracting time range: {start_time}s - {end_time}s")
    
    spectrogram_snippet, piano_roll_snippet = extract_time_range(
        full_spectrogram,
        full_piano_roll,
        start_time,
        end_time,
        CONFIG['roll_fps'],
        CONFIG['hop_length'],
        CONFIG['sample_rate']
    )
    
    print(f"Spectrogram snippet shape: {spectrogram_snippet.shape}")
    if piano_roll_snippet is not None:
        print(f"Piano roll snippet shape: {piano_roll_snippet.shape}")
    
    # Run inference
    print("\nRunning inference...")
    predictions = run_inference(model, spectrogram_snippet, device)
    print(f"Predictions shape: {predictions.shape}")
    
    # Calculate metrics if ground truth is available
    if piano_roll_snippet is not None:
        metrics = calculate_metrics(predictions, piano_roll_snippet)
        print(f"\nMetrics:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
    
    # Visualize results
    output_filename = os.path.join(
        INFERENCE_CONFIG['output_dir'],
        f"inference_{start_time:.1f}s_{end_time:.1f}s.png"
    )
    
    visualize_results(
        predictions,
        piano_roll_snippet,
        output_filename,
        start_time,
        end_time,
        plot_full_range=INFERENCE_CONFIG['plot_full_range']
    )
    
    print(f"\nInference complete!")


if __name__ == "__main__":
    main()
