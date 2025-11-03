import os
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram
from scipy.sparse import load_npz


def spectrogramOfTestData(
    nbinsX,
    start_time=0,
    end_time=2,
    folder_name_input="output/folder_1/",
    folder_name_output=("example_data/"),
):
    # Load the audio file
    audio_data, sample_rate = sf.read(os.path.join(folder_name_input, "melody_trimmed.wav"))

    # Create a time axis in seconds
    duration = len(audio_data) / sample_rate  # total duration of the audio
    time = np.linspace(0.0, duration, len(audio_data))

    # Find the indices that correspond to these times
    start_index = int(start_time * sample_rate)
    end_index = int(end_time * sample_rate)

    # Slice the audio data and time arrays
    zoomed_audio = audio_data[start_index:end_index]
    zoomed_time = time[start_index:end_index]

    # Load the notes file
    note_labels = np.load(os.path.join(folder_name_input, "notes.npy"))
    note_labels_downsampled = note_labels[:: round(len(note_labels) / (nbinsX - 1))]
    actual_notes = 440 * 2 ** ((note_labels_downsampled - 69) / 12)

    predicted_notes = np.load(os.path.join(folder_name_output, "example_output.npy"))[1]
    plt.figure(figsize=(6, 6))
    plt.pcolormesh(predicted_notes.T, cmap="gray")
    for i in range(0, nbinsX - 1):
        plt.axhline(
            y=note_labels_downsampled[i],
            color="red",
            alpha=0.2,
            linewidth=1,
            xmin=i / nbinsX,
            xmax=(i + 1) / nbinsX,
        )
    plt.xlabel("Time")
    plt.ylabel("MIDI number")
    plt.colorbar(label="probability")
    plt.show()
    # # Generate the spectrogram
    # f, t, Sxx = spectrogram(zoomed_audio[:,0], fs=sample_rate, nperseg=1024)

    # # Avoid taking log of zero by replacing zero values in Sxx with a small value
    # Sxx = np.where(Sxx <= 1e-13, 1e-13, Sxx)  # Replace zeros with a small value

    # # Plot the spectrogram (convert to dB scale)
    # plt.figure(figsize=(6, 6))
    # plt.pcolormesh(t, np.log10(f+1e-10), np.log10(Sxx),cmap='viridis', vmin=-10, vmax=-5)
    # for i in range(0,nbinsX-1):
    #     plt.axhline(y=np.log10(actual_notes[i]), color='red', linewidth=1, xmin=i/nbinsX,xmax=(i+1)/nbinsX)
    # plt.title("Spectrogram")
    # plt.xlabel("Time [s]")
    # plt.ylabel("log(Frequency [Hz])")
    # plt.colorbar(label="Power [dB]")
    # plt.ylim(np.log10(27.5), np.log10(4186))  # Optional: limit frequency range to 10 kHz
    # plt.show()


if __name__ == "__main__":

    piano_roll = load_npz(
        "/media/hendrik/data/workspace_2/Transcription/src/all_results/Sunburst (Album Version)/piano_roll_sparse.npz"
    )

    # Get coordinates of non-zero elements
    rows, cols = piano_roll.nonzero()

    plt.figure(figsize=(6, 6))
    plt.scatter(rows, cols, s=1, color="black")  # s controls point size
    plt.xlabel("Time")
    plt.ylabel("MIDI number")
    plt.show()
