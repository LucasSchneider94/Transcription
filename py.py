import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram
# Read an audio file
audio_data, sample_rate = sf.read("spectrogram test/Beethoven_ String Quartet #16 In F, Op. 135 - 1. Allegretto.mp3")

# Write to a WAV file
sf.write("spectrogram_file.wav", audio_data, sample_rate)

print("Audio conversion successful!")