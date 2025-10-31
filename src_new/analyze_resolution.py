"""
Diagnostic tool to evaluate spectrogram resolution quality.
Tests whether notes are visually distinguishable at different time/frequency settings.
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from pathlib import Path
import argparse


def create_spectrogram_comparison(audio_path, midi_path=None, save_path='resolution_comparison.png'):
    """
    Create spectrograms with different time/frequency resolutions for comparison.
    
    The goal: Can you clearly see individual note onsets and distinguish pitches?
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=48000)
    duration = len(y) / sr
    print(f"Audio duration: {duration:.2f}s, sample rate: {sr} Hz")
    
    # Different configurations to test
    configs = [
        # Current setup
        {'name': 'Current\n100fps, 264 bins', 'fps': 100, 'n_mels': 264, 'n_fft': 2048},
        
        # More time resolution
        {'name': 'High Time\n200fps, 264 bins', 'fps': 200, 'n_mels': 264, 'n_fft': 1024},
        
        # More frequency resolution
        {'name': 'High Freq\n100fps, 352 bins', 'fps': 100, 'n_mels': 352, 'n_fft': 4096},
        
        # Balanced high resolution
        {'name': 'Balanced High\n150fps, 352 bins', 'fps': 150, 'n_mels': 352, 'n_fft': 2048},
        
        # Lower resolution (faster)
        {'name': 'Low Res\n50fps, 176 bins', 'fps': 50, 'n_mels': 176, 'n_fft': 2048},
        
        # Very high resolution
        {'name': 'Very High\n200fps, 440 bins', 'fps': 200, 'n_mels': 440, 'n_fft': 4096},
    ]
    
    # Take a 5-second excerpt from the middle
    start_sec = duration / 2 - 2.5
    end_sec = duration / 2 + 2.5
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    y_excerpt = y[start_sample:end_sample]
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    for idx, config in enumerate(configs):
        hop_length = int(sr / config['fps'])
        
        # Compute mel spectrogram
        S = librosa.feature.melspectrogram(
            y=y_excerpt,
            sr=sr,
            n_fft=config['n_fft'],
            hop_length=hop_length,
            n_mels=config['n_mels'],
            fmin=librosa.midi_to_hz(21),  # A0
            fmax=librosa.midi_to_hz(108)  # C8
        )
        S_db = librosa.power_to_db(S, ref=np.max)
        
        # Plot
        ax = axes[idx]
        img = librosa.display.specshow(
            S_db,
            sr=sr,
            hop_length=hop_length,
            x_axis='time',
            y_axis='mel',
            ax=ax,
            cmap='viridis'
        )
        ax.set_title(f"{config['name']}\n"
                    f"Time res: {1000/config['fps']:.1f}ms, "
                    f"Bins/key: {config['n_mels']/88:.1f}",
                    fontsize=10)
        
        # Add frequency resolution info
        freq_res = sr / config['n_fft']
        ax.text(0.02, 0.98, f"Freq res: {freq_res:.1f} Hz", 
               transform=ax.transAxes, fontsize=8, va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.colorbar(img, ax=ax, format='%+2.0f dB')
    
    plt.suptitle(f'Spectrogram Resolution Comparison\n'
                f'Goal: Can you clearly see note onsets and distinguish pitches?',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison saved to: {save_path}")
    print("\nEvaluation criteria:")
    print("  ✓ Can you see sharp vertical lines at note onsets?")
    print("  ✓ Are different pitches clearly separated horizontally?")
    print("  ✓ Can you distinguish individual notes in chords?")
    print("  ✓ Is there minimal time smearing for fast passages?")
    
    return configs


def analyze_note_separation(audio_path, piano_roll_path=None):
    """
    Analyze if different resolutions can separate closely-spaced notes.
    This is the ultimate test - can the spectrogram resolve what we need to detect?
    """
    y, sr = librosa.load(audio_path, sr=48000)
    
    # Test different resolutions
    configs = [
        {'fps': 50, 'n_mels': 176, 'n_fft': 2048},
        {'fps': 100, 'n_mels': 264, 'n_fft': 2048},  # Current
        {'fps': 200, 'n_mels': 352, 'n_fft': 4096},
    ]
    
    print("\n" + "="*80)
    print("RESOLUTION ANALYSIS")
    print("="*80)
    
    for config in configs:
        hop_length = int(sr / config['fps'])
        time_res_ms = 1000 / config['fps']
        freq_res_hz = sr / config['n_fft']
        bins_per_key = config['n_mels'] / 88
        
        # Calculate minimum resolvable time difference (Nyquist)
        min_time_gap_ms = time_res_ms * 2
        
        # Calculate minimum resolvable frequency difference
        # For mel scale, resolution varies with frequency
        mel_res_low = librosa.hz_to_mel(110) - librosa.hz_to_mel(110 - freq_res_hz)  # A2
        mel_res_high = librosa.hz_to_mel(4186) - librosa.hz_to_mel(4186 - freq_res_hz)  # C8
        
        print(f"\nConfig: {config['fps']} fps, {config['n_mels']} mels, FFT={config['n_fft']}")
        print(f"  Time resolution: {time_res_ms:.1f}ms per frame")
        print(f"  Min detectable note gap: {min_time_gap_ms:.1f}ms")
        print(f"  Frequency resolution: {freq_res_hz:.1f} Hz")
        print(f"  Mel bins per piano key: {bins_per_key:.2f}")
        print(f"  Can resolve semitone? {'✓ YES' if bins_per_key >= 1 else '✗ NO (needs >1 bin/key)'}")
        
        # Real-world interpretation
        if min_time_gap_ms < 20:
            print(f"  ✓ Fast passages: Can resolve 32nd notes at 120 BPM")
        elif min_time_gap_ms < 50:
            print(f"  ⚠ Fast passages: Can resolve 16th notes at 120 BPM")
        else:
            print(f"  ✗ Fast passages: May miss rapid notes")
    
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze spectrogram resolution quality')
    parser.add_argument('--audio', type=str, help='Path to audio file to analyze')
    parser.add_argument('--data-dir', type=str, default='./processed_data',
                       help='Directory with processed data')
    args = parser.parse_args()
    
    # Find an audio file to test
    if args.audio:
        audio_path = args.audio
    else:
        # Use first file from processed data
        data_dir = Path(args.data_dir)
        spec_files = list(data_dir.glob("*_spectrogram.npy"))
        if spec_files:
            # Need to find corresponding audio - let's use a sample from data_subset
            sample_dir = Path('./data_subset')
            if sample_dir.exists():
                print("Using sample from data_subset for analysis...")
                # For now, just analyze without audio
                print("Note: This tool needs an audio file. Please provide one with --audio")
                print("\nExample usage:")
                print("  python analyze_resolution.py --audio /path/to/your/audio.wav")
                exit(1)
        else:
            print("No data found. Please provide an audio file with --audio")
            exit(1)
    
    # Run analysis
    create_spectrogram_comparison(audio_path)
    analyze_note_separation(audio_path)
