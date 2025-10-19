import os
from midi2audio import FluidSynth
import pretty_midi
import numpy as np
import scipy.sparse as sp
from pydub import AudioSegment
from midiutil import MIDIFile
from pydub import AudioSegment
from scipy.io import wavfile
import soundfile as sf
from scipy.signal import spectrogram
from tqdm import tqdm

def build_clean_piano_roll(midi_data, roll_fps=250):
    n_keys = 128
    end_time = midi_data.get_end_time()
    n_frames = int(end_time * roll_fps) + 1  # total frames
    piano_roll = np.zeros((n_frames, n_keys), dtype=np.uint8)

    for instrument in midi_data.instruments:
        if instrument.is_drum:
            continue  # skip drums
        for note in instrument.notes:
            start_frame = int(note.start * roll_fps)
            end_frame = int(note.end * roll_fps)
            piano_roll[start_frame:end_frame, note.pitch] = 1
    #piano_roll=piano_roll[:,20:108]
    return piano_roll

def build_clean_onset_roll(midi_data, roll_fps=250):
    n_keys = 128
    end_time = midi_data.get_end_time()
    n_frames = int(end_time * roll_fps) + 1  # total frames
    piano_roll = np.zeros((n_frames, n_keys), dtype=np.uint8)

    for instrument in midi_data.instruments:
        if instrument.is_drum:
            continue  # skip drums
        for note in instrument.notes:
            start_frame = int(note.start * roll_fps)
            end_frame = int(note.end * roll_fps)
            piano_roll[start_frame:start_frame+1, note.pitch] = 1
    #piano_roll=piano_roll[:,20:108]
    return piano_roll

def build_sustain_aware_piano_roll(midi_data, roll_fps=250):
    n_keys = 128
    end_time = midi_data.get_end_time()
    n_frames = int(end_time * roll_fps) + 1
    piano_roll = np.zeros((n_frames, n_keys), dtype=np.uint8)

    for instrument in midi_data.instruments:
        if instrument.is_drum:
            continue  # skip drums

        # Build a list of sustain pedal events (control number 64)
        pedal_events = [cc for cc in instrument.control_changes if cc.number == 64]
        pedal_times = np.array([cc.time for cc in pedal_events])
        pedal_values = np.array([cc.value for cc in pedal_events])
        pedal_on = pedal_values >= 64  # 64 is the standard threshold for "pedal down"

        for note in instrument.notes:
            start_frame = int(note.start * roll_fps)

            # Default end is the note-off time
            note_end_time = note.end

            # Extend end time while sustain pedal is down
            # Look for the first pedal-off event after note.end
            if len(pedal_events) > 0:
                # Get all pedal-down intervals that overlap with this note
                i = np.searchsorted(pedal_times, note.start, side='right') - 1
                while i < len(pedal_times) - 1:
                    if not pedal_on[i]:
                        i += 1
                        continue
                    # pedal_on[i] is True
                    pedal_down_time = pedal_times[i]
                    # Search for corresponding pedal-up
                    j = i + 1
                    while j < len(pedal_times) and pedal_on[j]:
                        j += 1
                    if j < len(pedal_times):
                        pedal_up_time = pedal_times[j]
                    else:
                        pedal_up_time = end_time  # pedal held to end

                    if note.start < pedal_up_time and note.end <= pedal_up_time:
                        note_end_time = max(note_end_time, pedal_up_time)
                        break
                    i = j

            end_frame = int(note_end_time * roll_fps)
            piano_roll[start_frame:end_frame, note.pitch] = 1

    return piano_roll

def build_piano_roll_and_onsets(midi_data, roll_fps=250):
    n_keys = 128
    end_time = midi_data.get_end_time()
    n_frames = int(end_time * roll_fps) + 1

    piano_roll = np.zeros((n_frames, n_keys), dtype=np.uint8)
    onsets = np.zeros((n_frames, n_keys), dtype=np.uint8)

    for instrument in midi_data.instruments:
        if instrument.is_drum:
            continue

        # Sustain pedal (control change 64)
        pedal_events = [cc for cc in instrument.control_changes if cc.number == 64]
        pedal_times = np.array([cc.time for cc in pedal_events])
        pedal_values = np.array([cc.value for cc in pedal_events])
        pedal_on = pedal_values >= 64

        for note in instrument.notes:
            start_frame = int(note.start * roll_fps)
            end_frame = int(note.end * roll_fps)
            note_end_time = note.end

            # Extend note due to pedal if needed
            if len(pedal_events) > 0:
                i = np.searchsorted(pedal_times, note.start, side='right') - 1
                while i < len(pedal_times) - 1:
                    if not pedal_on[i]:
                        i += 1
                        continue
                    pedal_down_time = pedal_times[i]
                    j = i + 1
                    while j < len(pedal_times) and pedal_on[j]:
                        j += 1
                    pedal_up_time = pedal_times[j] if j < len(pedal_times) else end_time
                    if note.start < pedal_up_time and note.end <= pedal_up_time:
                        note_end_time = max(note_end_time, pedal_up_time)
                        break
                    i = j

            sustained_end_frame = int(note_end_time * roll_fps)

            # Fill sustain piano roll
            piano_roll[start_frame:sustained_end_frame, note.pitch] = 1

            # Mark onset
            onsets[start_frame, note.pitch] = 1

    return piano_roll, onsets


def process_midi(mid_path, output_dir, roll_fps=250):
    
    # Load MIDI and compute piano roll (shape: 128 x T)
    midi_data = pretty_midi.PrettyMIDI(mid_path)
    file_name = os.path.splitext(os.path.basename(mid_path))[0]
    #piano_roll = midi_data.get_piano_roll(fs=roll_fps)
    # Binarize (active note if >0) and transpose to shape: T x 128
    piano_roll_binary = build_clean_piano_roll(midi_data, roll_fps=roll_fps) #piano_roll_binary = (piano_roll > 20).astype(np.uint8).T
    onset_roll_binary = build_clean_onset_roll(midi_data, roll_fps=roll_fps)

    # Convert to sparse matrix (CSR format) and save as .npz
    sparse_piano_roll = sp.csr_matrix(piano_roll_binary)
    sparse_filename = os.path.join(output_dir, f"{file_name}_piano_roll_sparse.npz")
    sp.save_npz(sparse_filename, sparse_piano_roll)
    sparse_onset_roll = sp.csr_matrix(onset_roll_binary)
    sparse_filename = os.path.join(output_dir, f"{file_name}_onsets_sparse.npz")
    sp.save_npz(sparse_filename, sparse_onset_roll)
    print(f"NumPy files saved to {output_dir}")


def process_all_midis(root_folder, output_folder, roll_fps=250):

    for current_dir, _, files in tqdm(os.walk(root_folder)):
        try:
            for file_name in files:
                if file_name.lower().endswith(".midi"):
                    mid_path = os.path.join(current_dir, file_name)
                    process_midi(mid_path, output_dir=output_folder, roll_fps=roll_fps)
        except Exception as e: # Catch all exceptions for now
            print(f"An error occurred while processing {file_name}: {e}")

if __name__ == "__main__":
    root_folder = "../maestro-v3.0.0/2017" #"../adl-piano-midi-master/midi/adl-piano-midi/Ambient"
    print(f"Root folder being processed: {root_folder}")
    if not os.path.exists(root_folder):
        print(f"Error: Root folder does not exist: {root_folder}")
        pass # Exit if folder doesn't exist
    output_folder = root_folder
    roll_fps = 100
    process_all_midis(
        root_folder=root_folder,
        output_folder=output_folder,
        roll_fps=roll_fps
    )