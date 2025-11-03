import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import TranscriptionModel
from train import MaestroDataset


def evaluate_model(data_dir, model_path, output_dir):
    """
    Evaluate the trained model on the dataset and visualize results.

    Args:
        data_dir (str): Path to the processed dataset.
        model_path (str): Path to the trained model file.
        output_dir (str): Directory to save evaluation results.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset and model
    dataset = MaestroDataset(data_dir)
    model = TranscriptionModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Evaluate on a few samples
    for idx in range(min(5, len(dataset))):
        audio, piano_roll = dataset[idx]
        audio = audio.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

        with torch.no_grad():
            piano_roll_pred, onset_pred, decay_pred = model(audio)

        # Convert predictions to numpy
        piano_roll_pred = piano_roll_pred.squeeze(0).cpu().numpy()
        onset_pred = onset_pred.squeeze(0).cpu().numpy()
        decay_pred = decay_pred.squeeze(0).cpu().numpy()

        # Save visualizations
        plt.figure(figsize=(15, 5))

        plt.subplot(3, 1, 1)
        plt.title("Ground Truth Piano Roll")
        plt.imshow(piano_roll.T, aspect="auto", origin="lower", cmap="hot")
        plt.colorbar()

        plt.subplot(3, 1, 2)
        plt.title("Predicted Piano Roll")
        plt.imshow(piano_roll_pred.T, aspect="auto", origin="lower", cmap="hot")
        plt.colorbar()

        plt.subplot(3, 1, 3)
        plt.title("Predicted Onset and Decay")
        plt.plot(onset_pred.mean(axis=1), label="Onset")
        plt.plot(decay_pred.mean(axis=1), label="Decay")
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"sample_{idx}_evaluation.png"))
        plt.close()

        print(f"Saved evaluation visualization for sample {idx}")

if __name__ == "__main__":
    data_dir = "./processed_data"
    model_path = "./transcription_model.pth"
    output_dir = "./evaluation_results"
    evaluate_model(data_dir, model_path, output_dir)