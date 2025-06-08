import os
import numpy as np
from midiutil import MIDIFile
from midi2audio import FluidSynth
from pydub import AudioSegment

def create_multi_note_midi(
    note_events,
    total_time,
    pathlocal="/Users/.../someFolder/",  # Adapt to your default
    base_folder="output",
    n_notes=120,            # Maximum MIDI note you’d like to record is < n_notes
    samplerate=44100
):
    """
    note_events: list of [pitch, start_time, duration]
                 e.g. [[88, 0, 2], [66, 0, 1.3], ...]
    total_time : float (seconds)  => total length of output files
    pathlocal  : path containing your soundfont file "FluidR3_GM.sf2"
    base_folder: top-level folder where new sub-folders will be created
    n_notes    : number of possible MIDI notes you'd like to encode (0..n_notes-1)
    samplerate : integer sample rate used for final WAV and the note-array 
    """

    # 1) Create an output subfolder
    os.makedirs(base_folder, exist_ok=True)
    existing_folders = [
        f for f in os.listdir(base_folder)
        if os.path.isdir(os.path.join(base_folder, f))
    ]
    folder_number = len(existing_folders) + 1
    folder_name = os.path.join(base_folder, f'folder_{folder_number}')
    os.makedirs(folder_name)

    # 2) Create MIDI file
    midi = MIDIFile(numTracks=1)
    track = 0
    midi.addTempo(track, time=0, tempo=60)  # or choose any BPM you like
    channel = 0
    volume = 100

    # Add each note event to the MIDI file
    for pitch, start_t, dur in note_events:
        midi.addNote(track, channel, pitch, start_t, dur, volume)

    midi_file_path = os.path.join(folder_name, "melody.mid")
    with open(midi_file_path, "wb") as output_file:
        midi.writeFile(output_file)

    # 3) Convert MIDI to WAV
    soundfont = os.path.join(pathlocal, "FluidR3_GM.sf2")
    fs = FluidSynth(sample_rate=samplerate, sound_font=soundfont)
    wav_file_path = os.path.join(folder_name, "melody.wav")
    fs.midi_to_audio(midi_file_path, wav_file_path)

    # 4) Trim or force total length
    #    if WAV is shorter than total_time, Pydub will pad with silence
    sound = AudioSegment.from_wav(wav_file_path)
    # forcibly slice or pad to `total_time`
    desired_length_ms = int(total_time * 1000)
    if len(sound) < desired_length_ms:
        # pad with silence
        silence = AudioSegment.silent(duration=desired_length_ms - len(sound))
        sound = sound + silence
    else:
        # trim to total_time
        sound = sound[:desired_length_ms]

    trimmed_wav_path = os.path.join(folder_name, "melody_trimmed.wav")
    sound.export(trimmed_wav_path, format="wav")

    # 5) Create time-based note-active array (samples x n_notes)
    total_samples = int(total_time * samplerate)
    note_array = np.zeros((total_samples, n_notes), dtype=np.uint8)

    # Fill it in by marking active notes for the time samples they cover
    # For each event, turn on pitch for range [start_t, start_t + dur)
    for pitch, start_t, dur in note_events:
        start_index = int(start_t * samplerate)
        end_index   = int((start_t + dur) * samplerate)
        # clamp to total_samples range
        start_index_clamped = max(0, min(total_samples, start_index))
        end_index_clamped   = max(0, min(total_samples, end_index))
        if 0 <= pitch < n_notes and end_index_clamped > start_index_clamped:
            note_array[start_index_clamped:end_index_clamped, pitch] = 1

    # 6) Save the array
    note_array_path = os.path.join(folder_name, "notes.npy")
    np.save(note_array_path, note_array)

    return folder_name

if __name__ == "__main__":
    events = [
        [88, 0, 2],
        [66, 0, 1.3],
        [60, 2.1, 0.7]
    ]
    pathlocal = "/usr/share/sounds/sf2/FluidR3_GM.sf2"
    folder = create_multi_note_midi(events, total_time=4.0, pathlocal=pathlocal, base_folder="output_multi")
    print("Results stored in:", folder)
    