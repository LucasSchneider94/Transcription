"""
Data preparation for raw audio piano transcription.
Generates aligned raw audio frames and piano rolls (no spectrograms).
"""

import numpy as np
import librosa
import pretty_midi
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import CONFIG, MIN_PITCH, MAX_PITCH, NUM_PEDALS


def audio_to_frames(audio, sample_rate, fps, context_samples=2048):
    """
    Convert audio to overlapping frames for CNN input.
    
    Each frame gets a context window of `context_samples` centered on it.
    This preserves frequency information while allowing frame-rate prediction.
    
    Args:
        audio: Audio waveform (samples,)
        sample_rate: Audio sample rate
        fps: Target frame rate for predictions
        context_samples: Size of context window (e.g., 2048 for ~23Hz resolution)
    
    Returns:
        frames: (num_frames, context_samples) - overlapping audio windows
    """
    hop_length = int(sample_rate / fps)
    num_frames = int(len(audio) / hop_length)
    
    # Pad audio for edge cases
    half_context = context_samples // 2
    audio_padded = np.pad(audio, (half_context, half_context), mode='reflect')
    
    # Extract overlapping frames
    frames = []
    for i in range(num_frames):
        center = i * hop_length + half_context
        frame = audio_padded[center - half_context : center + half_context]
        
        if len(frame) == context_samples:
            frames.append(frame)
    
    return np.array(frames)  # (num_frames, context_samples)


def midi_to_piano_roll(midi_path, fps, total_frames):
    """
    Convert MIDI to piano roll representation.
    Same as before - this part doesn't change.
    """
    midi = pretty_midi.PrettyMIDI(midi_path)
    
    # Initialize piano roll (keys + pedals)
    piano_roll = np.zeros((total_frames, 88 + NUM_PEDALS), dtype=np.float32)
    
    # Process note events
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        
        for note in instrument.notes:
            start_frame = int(note.start * fps)
            end_frame = int(note.end * fps)
            pitch_idx = note.pitch - MIN_PITCH
            
            if 0 <= pitch_idx < 88:
                piano_roll[start_frame:end_frame, pitch_idx] = 1.0
    
    # Process pedal events
    for control_change in midi.instruments[0].control_changes if midi.instruments else []:
        frame = int(control_change.time * fps)
        
        if control_change.number == 64:  # Sustain
            value = 1.0 if control_change.value >= 64 else 0.0
            if frame < total_frames:
                piano_roll[frame, 88] = value
        elif control_change.number == 67:  # Soft
            value = 1.0 if control_change.value >= 64 else 0.0
            if frame < total_frames:
                piano_roll[frame, 89] = value
        elif control_change.number == 66:  # Sostenuto
            value = 1.0 if control_change.value >= 64 else 0.0
            if frame < total_frames:
                piano_roll[frame, 90] = value
    
    return piano_roll


def process_audio_file(audio_path, midi_path, output_dir, config):
    """Process a single audio file to raw frames + piano roll."""
    # Load audio
    audio, sr = librosa.load(audio_path, sr=config['sample_rate'])
    
    # Convert to frames with context windows
    audio_frames = audio_to_frames(
        audio, 
        sr, 
        config['roll_fps'],
        context_samples=config['n_fft']  # Use same context as old FFT window
    )
    
    num_frames = len(audio_frames)
    
    # Create piano roll
    piano_roll = midi_to_piano_roll(midi_path, config['roll_fps'], num_frames)
    
    # Ensure alignment
    min_frames = min(len(audio_frames), len(piano_roll))
    audio_frames = audio_frames[:min_frames]
    piano_roll = piano_roll[:min_frames]
    
    # Save
    base_name = Path(audio_path).stem
    
    # Save raw audio frames (replaces spectrogram)
    audio_frames_path = os.path.join(output_dir, f"{base_name}_audio_frames.npy")
    np.save(audio_frames_path, audio_frames.astype(np.float32))
    
    # Save piano roll
    piano_roll_path = os.path.join(output_dir, f"{base_name}_piano_roll_with_pedals.npz")
    np.savez_compressed(piano_roll_path, piano_roll=piano_roll)
    
    # Optional: plot for verification
    if config.get('plot_pngs', False):
        plot_audio_frames_and_piano_roll(
            audio_frames[:1000],  # First 10 seconds
            piano_roll[:1000],
            os.path.join(output_dir, f"{base_name}_verification.png")
        )
    
    return audio_frames.shape, piano_roll.shape


def plot_audio_frames_and_piano_roll(audio_frames, piano_roll, save_path):
    """Visualize raw audio frames and piano roll."""
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    
    # Plot audio waveform (first channel of frames)
    axes[0].imshow(audio_frames.T, aspect='auto', cmap='RdBu_r', 
                   origin='lower', interpolation='nearest')
    axes[0].set_title('Raw Audio Frames (Context Windows)')
    axes[0].set_xlabel('Frame Index (100 fps)')
    axes[0].set_ylabel('Sample Index in Window')
    
    # Plot piano roll
    axes[1].imshow(piano_roll.T, aspect='auto', cmap='hot', 
                   origin='lower', interpolation='nearest')
    axes[1].set_title('Piano Roll')
    axes[1].set_xlabel('Frame Index (100 fps)')
    axes[1].set_ylabel('Keys + Pedals')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def process_dataset(data_dir, output_dir, config):
    """Process entire dataset."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all audio/MIDI pairs
    audio_files = list(Path(data_dir).rglob("*.wav"))
    
    print(f"Found {len(audio_files)} audio files")
    print(f"Configuration:")
    print(f"  Sample rate: {config['sample_rate']} Hz")
    print(f"  Frame rate: {config['roll_fps']} fps")
    print(f"  Context window: {config['n_fft']} samples (~{config['sample_rate']/config['n_fft']:.1f} Hz resolution)")
    print(f"  Hop length: {config['hop_length']} samples")
    
    successful = 0
    failed = 0
    
    for audio_path in tqdm(audio_files, desc="Processing files"):
        # Find corresponding MIDI
        midi_path = audio_path.with_suffix('.midi')
        if not midi_path.exists():
            midi_path = audio_path.with_suffix('.mid')
        
        if not midi_path.exists():
            print(f"Warning: No MIDI found for {audio_path.name}")
            failed += 1
            continue
        
        try:
            audio_shape, piano_shape = process_audio_file(
                str(audio_path), 
                str(midi_path), 
                output_dir, 
                config
            )
            successful += 1
            
            if successful == 1:
                print(f"\nExample output shapes:")
                print(f"  Audio frames: {audio_shape}")
                print(f"  Piano roll: {piano_shape}")
        
        except Exception as e:
            print(f"Error processing {audio_path.name}: {e}")
            failed += 1
    
    print(f"\nProcessing complete:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    # Use same data directory structure
    maestro_dir = "../maestro-v3.0.0"
    output_dir = "./processed_data_raw"
    
    # Filter by year if specified
    if CONFIG.get('split_year_folder'):
        year_folder = CONFIG['split_year_folder']
        data_dir = os.path.join(maestro_dir, year_folder)
        print(f"Processing only year: {year_folder}")
    else:
        data_dir = maestro_dir
        print("Processing all years")
    
    process_dataset(data_dir, output_dir, CONFIG)
