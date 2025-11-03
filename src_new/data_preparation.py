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

def build_binary_piano_roll_with_pedals(midi_data, roll_fps=ROLL_FPS):
    """
    Build a binary piano roll with additional channels for pedal activations.

    Args:
        midi_data (pretty_midi.PrettyMIDI): MIDI data object.
        roll_fps (float): Frames per second for the piano roll.

    Returns:
        np.ndarray: Piano roll with 91 channels (88 keys + 3 pedals).
    """
    end_time = midi_data.get_end_time()
    n_frames = int(end_time * roll_fps) + 1

    # Initialize piano roll with 91 channels (88 keys + 3 pedals)
    piano_roll = np.zeros((n_frames, NUM_KEYS + NUM_PEDALS), dtype=np.uint8)

    # Process notes - filter and map to the range of 88 keys (MIDI pitches 21 to 108)
    for instrument in midi_data.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            if MIN_PITCH <= note.pitch <= MAX_PITCH:
                start_frame = int(note.start * roll_fps)
                end_frame = int(note.end * roll_fps)
                piano_roll[start_frame:end_frame, note.pitch - MIN_PITCH] = 1

    # Process pedals (control changes)
    pedal_controls = {
        64: NUM_KEYS,      # Sustain pedal
        67: NUM_KEYS + 1,  # Soft pedal
        66: NUM_KEYS + 2   # Sostenuto pedal
    }

    for instrument in midi_data.instruments:
        for cc in instrument.control_changes:
            if cc.number in pedal_controls:
                pedal_channel = pedal_controls[cc.number]
                frame = int(cc.time * roll_fps)
                piano_roll[frame:, pedal_channel] = 1 if cc.value >= 64 else 0

    return piano_roll

def visualize_piano_roll_with_onsets_offsets(onset, offset, frame, pedal, save_path, max_seconds=5, fps=100):
    """
    Visualize onset, offset, frame, and pedal data as separate subplots.
    
    Args:
        onset (np.ndarray): Onset roll (time, 88)
        offset (np.ndarray): Offset roll (time, 88)
        frame (np.ndarray): Frame roll (time, 88)
        pedal (np.ndarray): Pedal roll (time, 1)
        save_path (str): Path to save the PNG image
        max_seconds (int): Maximum seconds to visualize
        fps (int): Frames per second
    """
    if not PLOT_PNGS:
        return
    
    # Limit to first N seconds
    max_frames = int(fps * max_seconds)
    onset = onset[:max_frames, :]
    offset = offset[:max_frames, :]
    frame = frame[:max_frames, :]
    pedal = pedal[:max_frames, :]
    
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    
    # Plot onset
    axes[0].imshow(onset.T, aspect='auto', origin='lower', cmap='hot', interpolation='nearest')
    axes[0].set_title(f'Onset Predictions (First {max_seconds} Seconds)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Piano Keys (88)', fontsize=10)
    
    # Plot offset
    axes[1].imshow(offset.T, aspect='auto', origin='lower', cmap='hot', interpolation='nearest')
    axes[1].set_title(f'Offset Predictions (First {max_seconds} Seconds)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Piano Keys (88)', fontsize=10)
    
    # Plot frame (active notes)
    axes[2].imshow(frame.T, aspect='auto', origin='lower', cmap='hot', interpolation='nearest')
    axes[2].set_title(f'Frame (Active Notes) (First {max_seconds} Seconds)', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Piano Keys (88)', fontsize=10)
    
    # Plot pedal
    axes[3].imshow(pedal.T, aspect='auto', origin='lower', cmap='hot', interpolation='nearest')
    axes[3].set_title(f'Sustain Pedal (First {max_seconds} Seconds)', fontsize=12, fontweight='bold')
    axes[3].set_ylabel('Pedal', fontsize=10)
    axes[3].set_xlabel('Time Frames', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    #print(f"  Saved visualization: {save_path}")


def visualize_piano_roll(piano_roll, save_path):
    """
    Visualize and save the piano roll as a PNG image (first 5 seconds only).
    LEGACY: For old-style piano rolls with pedals.

    Args:
        piano_roll (np.ndarray): The piano roll to visualize.
        save_path (str): Path to save the PNG image.
    """
    if not PLOT_PNGS:
        return

    # Limit the piano roll to the first 5 seconds
    max_frames = int(ROLL_FPS * 5)
    piano_roll = piano_roll[:max_frames, :]

    plt.figure(figsize=(12, 6))
    plt.imshow(piano_roll.T, aspect='auto', origin='lower', cmap='hot', interpolation='nearest')
    plt.colorbar(label='Intensity')
    plt.title('Piano Roll Visualization (First 5 Seconds)')
    plt.xlabel('Time Frames')
    plt.ylabel('MIDI Note Number')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def process_midi(mid_path, output_dir):
    """
    Process a single MIDI file: create piano roll and spectrogram.

    Args:
        mid_path (str): Path to the MIDI file.
        output_dir (str): Directory to save output files.
    """
    midi_data = pretty_midi.PrettyMIDI(mid_path)
    file_name = os.path.splitext(os.path.basename(mid_path))[0]

    # Build onset/offset/frame labels instead of old piano roll
    labels = create_piano_roll_with_onsets_offsets(
        mid_path,
        fps=ROLL_FPS,
        onset_frames=CONFIG.get('onset_frames', 2),
        offset_frames=CONFIG.get('offset_frames', 1)
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

    # Save as .npz with onset/offset/frame (compatible format, can be loaded by old code too)
    output_filename = os.path.join(output_dir, f"{file_name}_piano_roll_with_pedals.npz")
    np.savez_compressed(
        output_filename,
        onset=labels['onset'],
        offset=labels['offset'],
        frame=labels['frame'],
        pedal=labels['pedal'],
        spectrogram=spectrogram
    )

    # Visualize onset/offset/frame labels if enabled
    if PLOT_PNGS:
        labels_plot_path = os.path.join(output_dir, f"{file_name}_labels.png")
        visualize_piano_roll_with_onsets_offsets(
            labels['onset'],
            labels['offset'],
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
            sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel', cmap='viridis'
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

def create_piano_roll_with_onsets_offsets(midi_path: str, 
                                          fps: int = 100,
                                          pitches: int = 88,
                                          onset_frames: int = 2,
                                          offset_frames: int = 1) -> Dict[str, np.ndarray]:
    """
    Create piano roll with onset, offset, and frame labels.
    
    Following the "Onsets & Frames" approach:
    - Onset: binary, marks note start (in N consecutive frames for alignment)
    - Offset: binary, marks note end (respecting sustain pedal)
    - Frame: binary, marks active notes
    
    Args:
        midi_path: Path to MIDI file
        fps: Frames per second (temporal resolution)
        pitches: Number of piano keys (88)
        onset_frames: Number of consecutive frames to mark for onsets
        offset_frames: Number of consecutive frames to mark for offsets
    
    Returns:
        Dictionary with keys 'onset', 'offset', 'frame', 'pedal'
    """
    pm = pretty_midi.PrettyMIDI(midi_path)
    
    # Get total duration
    total_time = pm.get_end_time()
    num_frames = int(np.ceil(total_time * fps))
    
    # Initialize arrays
    onset_roll = np.zeros((num_frames, pitches), dtype=np.float32)
    offset_roll = np.zeros((num_frames, pitches), dtype=np.float32)
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
                
                # Mark onset in N consecutive frames for better alignment
                onset_end = min(start_frame + onset_frames, num_frames)
                if start_frame < num_frames:
                    onset_roll[start_frame:onset_end, pitch_idx] = 1.0
                
                # Mark frames (active notes)
                frame_end = min(end_frame, num_frames)
                frame_roll[start_frame:frame_end, pitch_idx] = 1.0
                
                # Determine offset frame (with sustain pedal handling)
                offset_frame = end_frame
                
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
                    
                    # Extend frame until re-strike or pedal release
                    extended_end_frame = int(min(next_onset_time, pedal_release_time) * fps)
                    frame_roll[end_frame:min(extended_end_frame, num_frames), pitch_idx] = 1.0
                    offset_frame = extended_end_frame
                
                # Mark offset in N consecutive frames
                offset_end = min(offset_frame + offset_frames, num_frames)
                if offset_frame < num_frames:
                    offset_roll[offset_frame:offset_end, pitch_idx] = 1.0
    
    return {
        'onset': onset_roll,
        'offset': offset_roll,
        'frame': frame_roll,
        'pedal': pedal_roll
    }

def prepare_dataset(audio_dir: str, 
                   midi_dir: str, 
                   output_dir: str,
                   sr: int = 16000,
                   n_fft: int = 2048,
                   hop_length: int = 512,
                   fps: int = 100):
    """
    Prepare dataset with spectrograms and onset/offset/frame labels.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    audio_files = sorted(Path(audio_dir).glob('*.wav'))
    midi_files = sorted(Path(midi_dir).glob('*.mid'))
    
    for audio_file, midi_file in zip(audio_files, midi_files):
        print(f"Processing {audio_file.stem}...")
        
        # Load audio and compute spectrogram
        y, _ = librosa.load(str(audio_file), sr=sr)
        spec = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        spec_db = librosa.amplitude_to_db(np.abs(spec), ref=np.max)
        
        # Create piano roll with onsets and offsets
        labels = create_piano_roll_with_onsets_offsets(
            str(midi_file), 
            fps=fps,
            onset_frames=CONFIG.get('onset_frames', 2),
            offset_frames=CONFIG.get('offset_frames', 1)
        )
        
        # Ensure temporal alignment
        min_frames = min(spec_db.shape[1], labels['frame'].shape[0])
        spec_db = spec_db[:, :min_frames]
        
        for key in labels:
            labels[key] = labels[key][:min_frames]
        
        # Visualize onset/offset/frame labels
        if PLOT_PNGS:
            viz_path = Path(output_dir) / f"{audio_file.stem}_labels.png"
            visualize_piano_roll_with_onsets_offsets(
                labels['onset'],
                labels['offset'],
                labels['frame'],
                labels['pedal'],
                str(viz_path),
                max_seconds=5,
                fps=fps
            )
        
        # Save
        output_path = Path(output_dir) / f"{audio_file.stem}.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump({
                'spectrogram': spec_db,
                'onset': labels['onset'],
                'offset': labels['offset'],
                'frame': labels['frame'],
                'pedal': labels['pedal']
            }, f)
        
        print(f"  Saved: {output_path}")

if __name__ == "__main__":
    maestro_dir = "../maestro-v3.0.0"
    output_dir = DATA_DIR
    process_maestro_split(maestro_dir, output_dir, split_year=SPLIT_YEAR)