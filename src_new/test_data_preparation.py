import os
import numpy as np
import matplotlib.pyplot as plt
from data_preparation import process_maestro_split

def visualize_piano_roll(piano_roll, output_path):
    """
    Visualize the piano roll and save it as a PNG file.

    Args:
        piano_roll (np.ndarray): The piano roll to visualize.
        output_path (str): Path to save the PNG file.
    """
    plt.figure(figsize=(12, 6))
    plt.imshow(piano_roll.T, aspect='auto', origin='lower', cmap='hot', interpolation='nearest')
    plt.colorbar(label='Intensity')
    plt.xlabel('Time Frames')
    plt.ylabel('MIDI Note Number')
    plt.title('Piano Roll Visualization')
    plt.savefig(output_path)
    plt.close()

def test_data_preparation():
    """
    Test the data preparation script and visualize the output for the first file.
    """
    maestro_dir = "../maestro-v3.0.0/training_subset"  # Updated to use a smaller subset
    output_dir = "./processed_data_test"

    # Process a small subset of the MAESTRO dataset
    process_maestro_split(maestro_dir, output_dir, split_year="2018_sub")

    # Find the first processed piano roll file
    piano_roll_files = [f for f in os.listdir(output_dir) if f.endswith("_piano_roll_with_pedals.npz")]
    if not piano_roll_files:
        print("No piano roll files found in the output directory.")
        return

    first_file = piano_roll_files[0]
    piano_roll_path = os.path.join(output_dir, first_file)

    # Load and visualize the piano roll
    data = np.load(piano_roll_path)
    piano_roll = data["arr_0"]  # Updated to load the sparse matrix

    # Only plot the first 20 seconds of the piano roll
    roll_fps = 250  # Frames per second
    max_frames = 20 * roll_fps
    piano_roll = piano_roll[:max_frames, :]

    visualization_path = os.path.join(output_dir, "piano_roll_with_pedals_visualization.png")
    visualize_piano_roll(piano_roll, visualization_path)
    print(f"Piano roll visualization saved to {visualization_path}")

if __name__ == "__main__":
    test_data_preparation()