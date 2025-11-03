import os
import numpy as np
# os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin"
from midiutil import MIDIFile
from midi2audio import FluidSynth
from pydub import AudioSegment
from itertools import product

def createMidi(notes, pathlocal="/Users/lucasschneider/Desktop/Privat/Transcription_Project/Transcription/"):
    #create folder name
    base_folder='output'
    # Ensure base folder exists
    os.makedirs(base_folder, exist_ok=True)
    
    # Count existing folders
    existing_folders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]
    folder_number = len(existing_folders) + 1
    folder_name = os.path.join(base_folder, f'folder_{folder_number}')
    
    # Create new folder
    os.makedirs(folder_name)
    
    # Step 1: Create MIDI
    midi = MIDIFile(1)
    track = 0
    time = 0
    midi.addTempo(track, time, 60)

    channel = 0
    volume = 100
    duration = 1
    samplerate=44000
    for i, note in enumerate(notes):
        midi.addNote(track, channel, note, time + i * duration, duration, volume)

    # Save MIDI file
    midi_file = "melody.mid"
    with open(midi_file, "wb") as output_file:
        midi.writeFile(output_file)

    # Step 2: Convert MIDI to audio
    soundfont = os.path.join(pathlocal,"FluidR3_GM.sf2")
    fs = FluidSynth(sample_rate=samplerate,sound_font=soundfont)
    fs.midi_to_audio(midi_file, "melody.wav")

    #print("Done: melody.wav created.")


    sound = AudioSegment.from_wav("melody.wav")
    trim_time_ms = 1000*len(notes) 
    sound = sound[:trim_time_ms]
    file_path1 = os.path.join(folder_name, "melody_trimmed.wav")    
    sound.export(file_path1, format="wav")


    # Repeat values and save
    file_path2 = os.path.join(folder_name, 'notes.npy')    
    np.save(file_path2,np.repeat(notes, samplerate))

if __name__ == "__main__":
    pathlocal = "/usr/share/sounds/sf2/FluidR3_GM.sf2"
    pairs = list(product(range(10, 101), repeat=2))
    for pair in pairs: 
        print(pair)
        createMidi(pair, pathlocal=pathlocal)
