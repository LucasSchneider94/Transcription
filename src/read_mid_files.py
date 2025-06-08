import os
import argparse
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

def process_midi(mid_path, output_dir, soundfont_path, samplerate=44000):
    os.makedirs(output_dir, exist_ok=True)
    wav_filename = os.path.join(output_dir, "output.wav")
    # npy_filename = os.path.join(output_dir, "piano_roll.npy")

    # Convert MIDI to WAV using FluidSynth
    fs = FluidSynth(sound_font=soundfont_path, sample_rate=samplerate)
    fs.midi_to_audio(mid_path, wav_filename)
    
    # Load MIDI and compute piano roll (shape: 128 x T)
    midi_data = pretty_midi.PrettyMIDI(mid_path)
    piano_roll = midi_data.get_piano_roll(fs=samplerate)
    # Binarize (active note if >0) and transpose to shape: T x 128
    piano_roll_binary = (piano_roll > 20).astype(np.uint8).T

    # Convert to sparse matrix (CSR format) and save as .npz
    sparse_piano_roll = sp.csr_matrix(piano_roll_binary)
    sparse_filename = os.path.join(output_dir, "piano_roll_sparse.npz")
    sp.save_npz(sparse_filename, sparse_piano_roll)

    # Trim WAV to expected duration
    expected_duration_ms = int(midi_data.get_end_time() * 1000)
    audio = AudioSegment.from_wav(wav_filename)
    if len(audio) > expected_duration_ms:
        audio = audio[:expected_duration_ms]
        audio.export(wav_filename, format="wav")
    
    # load wav file and compute spectrogram (time resolved FFT)
    audio_data, sample_rate = sf.read(os.path.join(output_dir, "output.wav"))
    f, t, Sxx = spectrogram(audio_data[:,0], fs=samplerate, nperseg=1024)
    Sxx = np.where(Sxx <= 1e-13, 1e-13, Sxx)  # Replace zeros with a small value
    np.save(os.path.join(output_dir,"Spectrogram.npy"),Sxx)
    # Optional: plot Spectrogram
    # plt.figure(figsize=(12, 6))
    # plt.pcolormesh(t, f, 10 * np.log10(Sxx))  # Convert power to dB
    # plt.title("Spectrogram")
    # plt.xlabel("Time [s]")
    # plt.ylabel("Frequency [Hz]")
    # plt.colorbar(label="Power [dB]")
    # plt.ylim(0, 8000)  # Optional: limit frequency range to 10 kHz
    # plt.show()
    # # Save the piano roll numpy file
    # np.save(npy_filename, piano_roll_binary)
    print(f"WAV and NumPy files saved to {output_dir}")


def process_all_midis(root_folder, output_folder, soundfont_path, samplerate=44000):
    for current_dir, _, files in tqdm(os.walk(root_folder)):
        try:
            for file_name in files:
                if file_name.lower().endswith(".mid"):
                    mid_path = os.path.join(current_dir, file_name)
                    relative_path = os.path.relpath(mid_path, root_folder)
                    flat_folder_name = os.path.splitext(relative_path)[0].replace(os.sep, "_")
                    sub_output_dir = os.path.join(output_folder, flat_folder_name)
                    process_midi(mid_path, sub_output_dir, soundfont_path, samplerate)
        except:
            pass


if __name__ == "__main__":
    root_folder = "/workspace/adl-piano-midi/midi/adl-piano-midi"
    output_folder = "with_spectogram_all"
    soundfont = "/usr/share/sounds/sf2/FluidR3_GM.sf2"
    samplerate = 22000
    process_all_midis(
        root_folder=root_folder,
        output_folder=output_folder,
        soundfont_path=soundfont,
        samplerate=samplerate
    )
