import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from model_audio_transriber import AudioSegmentationModel

###############################################################################
# 1) Helper functions
###############################################################################
def downsample_labels(labels: np.ndarray, target_length: int) -> np.ndarray:
    orig_len = len(labels)
    if target_length == orig_len:
        return labels
    idxs = np.linspace(0, orig_len - 1, target_length, dtype=np.int32)
    return labels[idxs]

###############################################################################
# 2) Dataset
###############################################################################
class MelodyDataset(Dataset):
    def __init__(self, parent_folder):
        self.samples = []
        subfolders = glob.glob(os.path.join(parent_folder, "*"))
        for sf in subfolders:
            wav_path = os.path.join(sf, "melody_trimmed.wav")
            notes_path = os.path.join(sf, "notes.npy")
            if os.path.isfile(wav_path) and os.path.isfile(notes_path):
                self.samples.append((wav_path, notes_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wav_path, notes_path = self.samples[idx]
        audio, sr = torchaudio.load(wav_path)
        audio = audio.T
        notes = np.load(notes_path)
        audio_tensor = audio.to(dtype=torch.float32)
        notes_tensor = torch.tensor(notes, dtype=torch.long)
        return audio_tensor, notes_tensor

###############################################################################
# 3) Evaluation function
###############################################################################
def eval_model(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for audio_batch, notes_batch in dataloader:
            audio_batch = audio_batch.to(device)
            notes_batch = notes_batch.to(device)
            logits = model.forward(audio_batch)
            B, T_down, _ = logits.shape
            downsampled_labels = [
                downsample_labels(notes_batch[b].cpu().numpy(), T_down)
                for b in range(B)
            ]
            downsampled_labels = np.stack(downsampled_labels)
            downsampled_labels = torch.from_numpy(downsampled_labels).to(device)
            loss = criterion(logits.view(-1, logits.size(-1)), downsampled_labels.view(-1))
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"Eval Loss: {avg_loss:.4f}")
    model.train()

###############################################################################
# 4) Training function
###############################################################################
def train_model(
    model,
    device,
    parent_folder,
    batch_size=2,
    num_epochs=5,
    learning_rate=1e-4,
    eval_interval=50,
    train_split=0.8
):
    model.to(device)

    dataset = MelodyDataset(parent_folder)
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for audio_batch, notes_batch in train_loader:
            audio_batch = audio_batch.to(device)
            notes_batch = notes_batch.to(device)
            optimizer.zero_grad()
            logits = model.forward(audio_batch)
            B, T_down, _ = logits.shape
            downsampled_labels = [
                downsample_labels(notes_batch[b].cpu().numpy(), T_down)
                for b in range(B)
            ]
            downsampled_labels = np.stack(downsampled_labels)
            downsampled_labels = torch.from_numpy(downsampled_labels).to(device)
            loss = criterion(logits.view(-1, logits.size(-1)), downsampled_labels.view(-1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if (epoch + 1) % eval_interval == 0:
            avg_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        if (epoch + 1) % eval_interval == 0:
            eval_model(model, test_loader, device)

    print("Training complete!")

###############################################################################
# 5) Main
###############################################################################
if __name__ == "__main__":

    if torch.cuda.is_available():
        my_device = torch.device("cuda")
        print("Using CUDA")
    elif torch.backends.mps.is_available():
        my_device = torch.device("mps")
        print("Using MPS (Apple GPU)")
    else:
        my_device = torch.device("cpu")
        print("Using CPU")

    audio_model = AudioSegmentationModel(
        num_notes=120,
        cnn_kernel_size=24,
        cnn_num_layers=4,
        transformer_num_layers=2,
        transformer_d_model=256,
        transformer_nhead=4,
        transformer_dim_feedforward=256,
        cnn_stride=4
    )
    train_model(
        model=audio_model,
        device=my_device,
        parent_folder="/workspace/src/output",
        batch_size=8,
        num_epochs=10000,
        eval_interval=1,
        train_split=0.8
    )
