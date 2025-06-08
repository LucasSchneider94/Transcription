import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from multiscale_residual_conv import MultiScaleResidualCNN
from tqdm import tqdm
###############################################################################
# 1) Helper functions
###############################################################################
def downsample_labels(labels: np.ndarray, target_length: int) -> np.ndarray:
    orig_len = len(labels)
    if target_length == orig_len:
        return labels
    idxs = np.linspace(0, orig_len - 1, target_length, dtype=np.int32)
    return labels[idxs]

def downsample_along_dim(x: torch.Tensor, dim: int, stride: int) -> torch.Tensor:
    """
    Downsamples tensor x along dimension dim to length out_len
    by taking every n-th element. (No interpolation/antialias.)
    """
    # Generate indices
    indices = torch.arange(0, x.size(dim), stride, device=x.device)
    return x.index_select(dim, indices)

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

class MelodyDatasetCPU(Dataset):
    def __init__(self, parent_folder):
        self.data = []
        subfolders = glob.glob(os.path.join(parent_folder, "*"))
        for sf in tqdm(subfolders, desc="Loading MelodyDataset"):
            wav_path = os.path.join(sf, "melody_trimmed.wav")
            notes_path = os.path.join(sf, "notes.npy")
            if os.path.isfile(wav_path) and os.path.isfile(notes_path):
                audio, _ = torchaudio.load(wav_path)
                audio = audio.T.to(dtype=torch.float32)  # (frames, channels)
                notes = torch.tensor(np.load(notes_path), dtype=torch.long)
                self.data.append((audio, notes))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
###############################################################################
# 3) Evaluation function
###############################################################################
def eval_model(model, dataloader, device, stride):
    model.eval()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for audio_batch, notes_batch in dataloader:
            audio_batch = audio_batch.to(device)
            notes_batch = notes_batch.to(device)
            audio_batch = downsample_along_dim(audio_batch, dim=1, stride=stride)
            notes_batch = downsample_along_dim(notes_batch, dim=1, stride=stride)
            logits = model.forward(audio_batch)
            B, T_down, _ = logits.shape
            downsampled_labels = [
                downsample_labels(notes_batch[b].cpu().numpy(), T_down)
                for b in range(B)
            ]
            downsampled_labels = np.stack(downsampled_labels)
            downsampled_labels = torch.from_numpy(downsampled_labels).to(device)
            loss = criterion(logits.reshape(-1, logits.size(-1)), downsampled_labels.reshape(-1))
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

    dataset = MelodyDatasetCPU(parent_folder)
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    stride = 8

    for epoch in range(num_epochs):
        running_loss = 0.0
        for audio_batch, notes_batch in tqdm(train_loader):
            audio_batch = audio_batch.to(device)
            notes_batch = notes_batch.to(device)
            audio_batch = downsample_along_dim(audio_batch, dim=1, stride=stride)
            notes_batch = downsample_along_dim(notes_batch, dim=1, stride=stride)
            optimizer.zero_grad()
            logits = model.forward(audio_batch)
            # print(f"logits: {logits.shape}")
            B, T_down, _ = logits.shape
            downsampled_labels = [
                downsample_labels(notes_batch[b].cpu().numpy(), T_down)
                for b in range(B)
            ]
            downsampled_labels = np.stack(downsampled_labels)
            downsampled_labels = torch.from_numpy(downsampled_labels).to(device)
            loss = criterion(logits.reshape(-1, logits.size(-1)), downsampled_labels.reshape(-1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if (epoch + 1) % eval_interval == 0:
            avg_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        if (epoch + 1) % eval_interval == 0:
            eval_model(model, test_loader, device, stride=stride)

    print("Training complete!")

###############################################################################
# 5) Main
###############################################################################
if __name__ == "__main__":
    audio_model = MultiScaleResidualCNN(
        n_notes=120,
        base_channels=64,
        num_blocks=3,
        block_kernel_sizes=[10, 10, 10],
        block_strides=[4, 4, 4],
        block_dilations=[1, 1, 2],
        multiscale_kernel_sizes=[5, 7, 11],
        use_multiscale=True
    )
    train_model(
        model=audio_model,
        device="cuda",
        parent_folder="/workspace/src/output",
        batch_size=8,
        num_epochs=10000,
        eval_interval=1,
        train_split=0.8
    )
