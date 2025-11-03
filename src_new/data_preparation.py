import os
import pretty_midi
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import scipy.sparse as sp
import matplotlib.pyplot as plt
from config import CONFIG, MIN_PITCH, MAX_PITCH, NUM_PEDALS

# Use CONFIG values globally throughout the script
ROLL_FPS = CONFIG['roll_fps']
SAMPLE_RATE = CONFIG['sample_rate']
N_MELS = CONFIG['n_mels']
N_FFT = CONFIG['n_fft']
HOP_LENGTH = CONFIG['hop_length']
NUM_KEYS = CONFIG['num_keys']
PLOT_PNGS = CONFIG['plot_pngs']
SPLIT_YEAR = CONFIG['split_year_folder']

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

def visualize_piano_roll(piano_roll, save_path):
    """
    Visualize and save the piano roll as a PNG image (first 5 seconds only).

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

    # Build binary piano roll with pedals
    piano_roll_with_pedals = build_binary_piano_roll_with_pedals(midi_data)

    # Process audio to spectrogram
    audio_path = mid_path.replace(".midi", ".wav").replace(".mid", ".wav")
    spectrogram_save_path = os.path.join(output_dir, f"{file_name}_spectrogram.npy")
    spectrogram_plot_path = os.path.join(output_dir, f"{file_name}_spectrogram.png") if PLOT_PNGS else None

    spectrogram = preprocess_audio_to_spectrogram(
        audio_path, 
        plot_path=spectrogram_plot_path, 
        save_path=spectrogram_save_path
    )

    # Save piano roll and spectrogram in the same .npz file
    output_filename = os.path.join(output_dir, f"{file_name}_piano_roll_with_pedals.npz")
    np.savez_compressed(output_filename, piano_roll=piano_roll_with_pedals, spectrogram=spectrogram)

    # Visualize piano roll if enabled
    if PLOT_PNGS:
        piano_roll_plot_path = os.path.join(output_dir, f"{file_name}_piano_roll.png")
        visualize_piano_roll(piano_roll_with_pedals, piano_roll_plot_path)

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

if __name__ == "__main__":
    maestro_dir = "../maestro-v3.0.0"
    output_dir = "./processed_data"
    process_maestro_split(maestro_dir, output_dir, split_year=SPLIT_YEAR)