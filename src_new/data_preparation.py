import os
import pretty_midi
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import scipy.sparse as sp
import matplotlib.pyplot as plt
from config import CONFIG, MIN_PITCH, MAX_PITCH, NUM_PEDALS
from pathlib import Path
import pickle
from typing import Tuple, Dict

# Use CONFIG values globally throughout the script
ROLL_FPS = CONFIG['roll_fps']
SAMPLE_RATE = CONFIG['sample_rate']
N_MELS = CONFIG['n_mels']
N_FFT = CONFIG['n_fft']
HOP_LENGTH = CONFIG['hop_length']
NUM_KEYS = CONFIG['num_keys']
PLOT_PNGS = CONFIG['plot_pngs']
SPLIT_YEAR = CONFIG['split_year_folder']
DATA_DIR = CONFIG['data_dir']

# Duration bin edges (in seconds) - logarithmic scale
# Bins: 0-0.05, 0.05-0.1, 0.1-0.2, 0.2-0.4, 0.4-0.8, 0.8-1.6, 1.6-3.2, 3.2+
DURATION_BINS = np.array([0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, np.inf])
NUM_DURATION_BINS = len(DURATION_BINS) - 1  # 8 bins

def get_duration_bin(duration_seconds):
    """
    Convert duration in seconds to bin index.
    
    Args:
        duration_seconds: Duration in seconds
        
    Returns:
        Bin index (0 to NUM_DURATION_BINS-1)
    """
    for i in range(len(DURATION_BINS) - 1):
        if DURATION_BINS[i] <= duration_seconds < DURATION_BINS[i + 1]:
            return i
    return NUM_DURATION_BINS - 1  # Last bin for very long notes

def get_duration_bin_label(bin_idx):
    """Get human-readable label for duration bin."""
    if bin_idx >= NUM_DURATION_BINS:
        return "Invalid"
    if bin_idx == NUM_DURATION_BINS - 1:
        return f"{DURATION_BINS[bin_idx]:.2f}s+"
    return f"{DURATION_BINS[bin_idx]:.2f}-{DURATION_BINS[bin_idx+1]:.2f}s"

def visualize_onset_duration_labels(onset, duration, frame, pedal, save_path, max_seconds=5, fps=100):
    """
    Visualize onset, duration, frame, and pedal data as separate subplots.
    
    Args:
        onset (np.ndarray): Onset roll (time, 88) - binary
        duration (np.ndarray): Duration roll (time, 88) - exact duration in seconds at onset frames, 0 elsewhere
        frame (np.ndarray): Frame roll (time, 88) - binary active notes
        pedal (np.ndarray): Pedal roll (time, 1) - binary
        save_path (str): Path to save the PNG image
        max_seconds (int): Maximum seconds to visualize
        fps (int): Frames per second
    """
    if not PLOT_PNGS:
        return
    
    # Limit to first N seconds
    max_frames = int(fps * max_seconds)
    onset = onset[:max_frames, :]
    duration = duration[:max_frames, :]
    frame = frame[:max_frames, :]
    pedal = pedal[:max_frames, :]
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    
    # Panel 1: Combined onset + duration heatmap
    # Convert exact durations to bins for visualization
    duration_binned = np.zeros_like(duration)
    for i in range(NUM_DURATION_BINS):
        mask = (duration >= DURATION_BINS[i]) & (duration < DURATION_BINS[i+1])
        duration_binned[mask] = i
    
    # 0 = silence/no onset, 1-8 = duration bins (mapped to warm colors)
    onset_duration_combined = duration_binned.astype(float).copy()
    # Where there's no onset, set to 0 (will appear as dark/cold color)
    onset_duration_combined[onset == 0] = 0
    # Add 1 to duration bins so they map to 1-8 instead of 0-7 (for better color separation from silence)
    onset_duration_combined[onset == 1] = duration_binned[onset == 1] + 1
    
    im1 = axes[0].imshow(onset_duration_combined.T, aspect='auto', origin='lower', 
                         cmap='hot', interpolation='nearest', vmin=0, vmax=NUM_DURATION_BINS+1)
    axes[0].set_title(f'Onsets with Duration (First {max_seconds} Seconds)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Piano Keys (88)', fontsize=10)
    
    # Add colorbar with bin labels
    cbar1 = plt.colorbar(im1, ax=axes[0], ticks=[0] + list(range(1, NUM_DURATION_BINS+1)))
    cbar1.set_label('Duration', fontsize=10)
    tick_labels = ['Silence'] + [get_duration_bin_label(i) for i in range(NUM_DURATION_BINS)]
    cbar1.ax.set_yticklabels(tick_labels, fontsize=8)
    
    # Panel 2: Frame (active notes)
    axes[1].imshow(frame.T, aspect='auto', origin='lower', cmap='hot', interpolation='nearest')
    axes[1].set_title(f'Frame (Active Notes) (First {max_seconds} Seconds)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Piano Keys (88)', fontsize=10)
    
    # Panel 3: Pedal
    axes[2].imshow(pedal.T, aspect='auto', origin='lower', cmap='hot', interpolation='nearest')
    axes[2].set_title(f'Sustain Pedal (First {max_seconds} Seconds)', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Pedal', fontsize=10)
    axes[2].set_xlabel('Time Frames', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    # Print statistics
    num_onsets = np.sum(onset)
    if num_onsets > 0:
        print(f"  Visualization stats:")
        print(f"    Total onsets: {int(num_onsets)}")
        
        # Get exact durations at onset frames
        exact_durations = duration[onset == 1]
        print(f"    Duration range: {exact_durations.min():.3f}s - {exact_durations.max():.3f}s")
        print(f"    Mean duration: {exact_durations.mean():.3f}s")
        print(f"    Median duration: {np.median(exact_durations):.3f}s")
        
        # Show distribution across bins
        duration_bins_count = [np.sum((exact_durations >= DURATION_BINS[i]) & 
                                      (exact_durations < DURATION_BINS[i+1])) 
                              for i in range(NUM_DURATION_BINS)]
        for i, count in enumerate(duration_bins_count):
            if count > 0:
                print(f"    {get_duration_bin_label(i)}: {int(count)} notes ({count/num_onsets*100:.1f}%)")

def process_maestro_split(maestro_dir, output_dir, split_year="2017"):
    """
    Process the MAESTRO dataset for a specific split year.

    Args:
        maestro_dir (str): Path to the MAESTRO dataset root directory.
        output_dir (str): Path to save processed data.
        split_year (str): Year of the MAESTRO split to process (e.g., "2017").
    """
    split_dir = os.path.join(maestro_dir, split_year)
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"Split directory {split_dir} does not exist.")

    os.makedirs(output_dir, exist_ok=True)

    for root, _, files in tqdm(os.walk(split_dir)):
        for file in files:
            if file.endswith(".midi") or file.endswith(".mid"):
                midi_path = os.path.join(root, file)
                audio_path = midi_path.replace(".midi", ".wav").replace(".mid", ".wav")

                if not os.path.exists(audio_path):
                    print(f"Audio file not found for {midi_path}, skipping.")
                    continue

                # Process MIDI and audio together
                process_midi(midi_path, output_dir)

def process_midi(mid_path, output_dir):
    """
    Process a single MIDI file: create piano roll and spectrogram.

    Args:
        mid_path (str): Path to the MIDI file.
        output_dir (str): Directory to save output files.
    """
    midi_data = pretty_midi.PrettyMIDI(mid_path)
    file_name = os.path.splitext(os.path.basename(mid_path))[0]

    # Build onset/duration/frame labels
    labels = create_piano_roll_with_onsets_durations(
        mid_path,
        fps=ROLL_FPS,
        onset_frames=CONFIG.get('onset_frames', 2)
    )

    # Process audio to spectrogram
    audio_path = mid_path.replace(".midi", ".wav").replace(".mid", ".wav")
    spectrogram_save_path = os.path.join(output_dir, f"{file_name}_spectrogram.npy")
    spectrogram_plot_path = os.path.join(output_dir, f"{file_name}_spectrogram.png") if PLOT_PNGS else None

    spectrogram = preprocess_audio_to_spectrogram(
        audio_path, 
        plot_path=spectrogram_plot_path, 
        save_path=spectrogram_save_path
    )

    # Ensure temporal alignment
    min_frames = min(spectrogram.shape[1], labels['frame'].shape[0])
    spectrogram = spectrogram[:, :min_frames]
    for key in labels:
        labels[key] = labels[key][:min_frames]

    # Save as .npz with onset/duration/frame
    output_filename = os.path.join(output_dir, f"{file_name}_piano_roll_with_pedals.npz")
    np.savez_compressed(
        output_filename,
        onset=labels['onset'],
        duration=labels['duration'],
        frame=labels['frame'],
        pedal=labels['pedal'],
        spectrogram=spectrogram
    )

    # Visualize onset/duration/frame labels if enabled
    if PLOT_PNGS:
        labels_plot_path = os.path.join(output_dir, f"{file_name}_labels.png")
        visualize_onset_duration_labels(
            labels['onset'],
            labels['duration'],
            labels['frame'],
            labels['pedal'],
            labels_plot_path,
            max_seconds=5,
            fps=ROLL_FPS
        )

    print(f"Processed {file_name}")

def preprocess_audio_to_spectrogram(audio_path, plot_path=None, save_path=None):
    """
    Convert an audio file to a Mel-spectrogram, optionally plot and save.

    Args:
        audio_path (str): Path to the audio file.
        plot_path (str): Path to save the spectrogram plot (optional).
        save_path (str): Path to save the spectrogram as a numpy array (optional).

    Returns:
        np.ndarray: Mel-spectrogram of shape (n_mels, time_frames).
    """
    # Load audio
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)

    # Compute Mel-spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )

    # Convert to log scale
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Plot the first 5 seconds if enabled
    if plot_path and PLOT_PNGS:
        plt.figure(figsize=(10, 4))
        max_time_frames = int(5 * SAMPLE_RATE / HOP_LENGTH)
        librosa.display.specshow(
            log_mel_spectrogram[:, :max_time_frames],
            sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel', cmap='hot'
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel-Spectrogram (First 5 Seconds)')
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    # Save the spectrogram as a numpy array
    if save_path:
        np.save(save_path, log_mel_spectrogram)

    return log_mel_spectrogram

def create_piano_roll_with_onsets_durations(midi_path: str, 
                                            fps: int = 100,
                                            pitches: int = 88,
                                            onset_frames: int = 2) -> Dict[str, np.ndarray]:
    """
    Create piano roll with onset, duration, and frame labels.
    
    New approach:
    - Onset: binary, marks note start (in N consecutive frames for alignment)
    - Duration: bin index (0 to NUM_DURATION_BINS-1) at ALL onset frames
    - Frame: binary, marks active notes with pedal extension (acoustic duration)
    
    IMPORTANT: Duration represents KEY PRESS TIME (note.end - note.start), NOT acoustic duration.
    This ensures:
    1. No conflict between duration and pedal (staccato with pedal down is valid)
    2. Sheet music can be reconstructed (durations match written notation)
    3. Acoustic rendering: predicted_duration + pedal_state = acoustic_duration
    
    Args:
        midi_path: Path to MIDI file
        fps: Frames per second (temporal resolution)
        pitches: Number of piano keys (88)
        onset_frames: Number of consecutive frames to mark for onsets
    
    Returns:
        Dictionary with keys 'onset', 'duration', 'frame', 'pedal'
    """
    pm = pretty_midi.PrettyMIDI(midi_path)
    
    # Get total duration
    total_time = pm.get_end_time()
    num_frames = int(np.ceil(total_time * fps))
    
    # Initialize arrays
    onset_roll = np.zeros((num_frames, pitches), dtype=np.float32)
    duration_roll = np.zeros((num_frames, pitches), dtype=np.float32)  # Exact durations in seconds
    frame_roll = np.zeros((num_frames, pitches), dtype=np.float32)
    pedal_roll = np.zeros((num_frames, 1), dtype=np.float32)
    
    # Process sustain pedal events (CC 64)
    pedal_intervals = []
    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        for cc in instrument.control_changes:
            if cc.number == 64:  # Sustain pedal
                pedal_frame = int(cc.time * fps)
                if pedal_frame < num_frames:
                    pedal_roll[pedal_frame:, 0] = 1.0 if cc.value >= 64 else 0.0
                    if cc.value >= 64:
                        pedal_intervals.append(('start', cc.time))
                    else:
                        pedal_intervals.append(('end', cc.time))
    
    # Build pedal active regions
    pedal_active = []
    pedal_start = None
    for event_type, time in sorted(pedal_intervals, key=lambda x: x[1]):
        if event_type == 'start' and pedal_start is None:
            pedal_start = time
        elif event_type == 'end' and pedal_start is not None:
            pedal_active.append((pedal_start, time))
            pedal_start = None
    if pedal_start is not None:
        pedal_active.append((pedal_start, total_time))
    
    # Helper function to check if pedal is active at time t
    def is_pedal_active(t):
        for start, end in pedal_active:
            if start <= t < end:
                return True
        return False
    
    # Process notes
    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
            
        # Sort notes by pitch and start time
        notes_by_pitch = {}
        for note in instrument.notes:
            pitch_idx = note.pitch - 21  # A0 = 21
            if 0 <= pitch_idx < pitches:
                if pitch_idx not in notes_by_pitch:
                    notes_by_pitch[pitch_idx] = []
                notes_by_pitch[pitch_idx].append(note)
        
        # Process each pitch separately to handle sustain and re-strikes
        for pitch_idx, notes in notes_by_pitch.items():
            notes = sorted(notes, key=lambda n: n.start)
            
            for i, note in enumerate(notes):
                start_frame = int(note.start * fps)
                end_frame = int(note.end * fps)
                
                # KEY CHANGE: Use note duration (key press time) ONLY
                # This is the duration we want the model to learn
                note_duration = note.end - note.start
                
                # Calculate acoustic duration (with pedal consideration) for frame roll
                acoustic_end_time = note.end
                
                # If sustain pedal is active at note end, extend until:
                # 1) Pedal is released, OR
                # 2) Same note is struck again (re-strike)
                if is_pedal_active(note.end):
                    # Find when pedal is released
                    pedal_release_time = total_time
                    for start, end in pedal_active:
                        if start <= note.end < end:
                            pedal_release_time = end
                            break
                    
                    # Check if there's a re-strike before pedal release
                    next_onset_time = pedal_release_time
                    if i + 1 < len(notes):
                        next_note = notes[i + 1]
                        if next_note.start < pedal_release_time:
                            next_onset_time = next_note.start
                    
                    # Acoustic duration extends to re-strike or pedal release
                    acoustic_end_time = min(next_onset_time, pedal_release_time)
                
                acoustic_end_frame = int(acoustic_end_time * fps)
                
                # Mark onset in N consecutive frames for better alignment
                onset_end = min(start_frame + onset_frames, num_frames)
                if start_frame < num_frames:
                    onset_roll[start_frame:onset_end, pitch_idx] = 1.0
                    
                    # FIX: Store exact duration in seconds at ALL onset frames
                    duration_roll[start_frame:onset_end, pitch_idx] = note_duration
                
                # Mark frames (active notes) - acoustic duration (what you hear)
                frame_end = min(acoustic_end_frame, num_frames)
                frame_roll[start_frame:frame_end, pitch_idx] = 1.0
    
    return {
        'onset': onset_roll,
        'duration': duration_roll,
        'frame': frame_roll,
        'pedal': pedal_roll
    }

if __name__ == "__main__":
    print("="*80)
    print("Piano Transcription Data Preparation - Onset + Duration Approach")
    print("="*80)
    print(f"\nDuration bins (logarithmic scale):")
    for i in range(NUM_DURATION_BINS):
        print(f"  Bin {i}: {get_duration_bin_label(i)}")
    print(f"\nTotal bins: {NUM_DURATION_BINS}")
    print(f"FPS: {ROLL_FPS}")
    print(f"Onset frames: {CONFIG.get('onset_frames', 2)}")
    print("="*80)
    
    maestro_dir = "../maestro-v3.0.0"
    output_dir = DATA_DIR
    process_maestro_split(maestro_dir, output_dir, split_year=SPLIT_YEAR)