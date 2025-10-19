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
import os
import torch
import numpy as np
import torchaudio
from torch.utils.data import Dataset
from scipy.sparse import load_npz
import matplotlib.pyplot as plt
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



class WavRollSparseDataset(Dataset):
    def __init__(self, root_dir, window_size):
        super().__init__()
        self.root_dir = root_dir
        self.window_size = window_size
        if not os.path.exists(root_dir):
            print(f"Error: Root folder does not exist: {root_dir}")
            pass

        # Collect subfolders
        self.subfolders = [
            os.path.join(root_dir, d)
            for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ]
        
        self.data = []
        for folder in self.subfolders:
            wav_path = os.path.join(folder, "output.wav")
            roll_path = os.path.join(folder, "piano_roll_sparse.npz")

            # 1) Load audio fully
            waveform, sr = torchaudio.load(wav_path)
            # # If multi-channel, reduce to mono (e.g. first channel):
            # if waveform.shape[0] > 1:
            #     waveform = waveform[0, :]
            # # Make shape (1, n_samples) for convenience:
            # waveform = waveform.unsqueeze(0)

            # 2) Load sparse piano roll
            #    (the file is expected to be a 2D sparse matrix of shape [T, 128])
            piano_roll_sp = load_npz(roll_path)
            # shape is (T, 128)
            T = piano_roll_sp.shape[0]

            self.data.append({
                'waveform': waveform,         # (1, n_samples)
                'sample_rate': sr,
                'piano_roll_sp': piano_roll_sp,  # sparse matrix
                'length': T                      # total timesteps
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        waveform = item['waveform']             # (1, n_samples)
        piano_roll_sp = item['piano_roll_sp']   # sparse matrix [T, 128]
        T = item['length']

        # Handle short sequences
        if T < self.window_size:
            # Option: return everything
            dense_roll = torch.from_numpy(piano_roll_sp.toarray()).float()
            return waveform, dense_roll

        # 1) Pick a random window
        start_idx = np.random.randint(0, T - self.window_size + 1)
        end_idx = start_idx + self.window_size

        # 2) Slice the sparse matrix => [window_size, 128], then convert to dense
        roll_window_sp = piano_roll_sp[start_idx:end_idx, :]  # no for-loop
        roll_window_np = roll_window_sp.toarray()             # shape (window_size, 128)
        roll_window = torch.from_numpy(roll_window_np).float()

        # 3) Slice the audio if you want approximate alignment
        waveform_window = waveform[:, start_idx:end_idx]

        return waveform_window, roll_window

    
###############################################################################
# 3) Evaluation function
###############################################################################
def eval_model(model, dataloader, device, stride):
    model.eval()
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for audio_batch, notes_batch in dataloader:
            audio_batch = audio_batch.to(device)
            notes_batch = notes_batch.to(device)
            audio_batch = downsample_along_dim(audio_batch, dim=2, stride=stride)
            notes_batch = downsample_along_dim(notes_batch, dim=1, stride=stride)
            logits = model.forward(audio_batch)
            B, T_down, _ = logits.shape
            downsampled_labels = [
                downsample_labels(notes_batch[b].cpu().numpy(), T_down)
                for b in range(B)
            ]
            downsampled_labels = np.stack(downsampled_labels)
            downsampled_labels = torch.from_numpy(downsampled_labels).to(device)
            loss = criterion(logits, downsampled_labels)
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    model.addEvalLosses(avg_loss)
    print(f"Eval Loss: {avg_loss:.4f}")
    model.train()

###############################################################################
# 4) Training function
###############################################################################
def train_model(
    model,
    device,
    batch_size=2,
    num_epochs=5,
    learning_rate=1e-4,
    eval_interval=50,
    train_split=0.8,
    root_dir="with_spectogram_all"
):
    model.to(device)

    dataset = WavRollSparseDataset(root_dir=root_dir, window_size=2*88000)
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    stride = 10

    for epoch in range(num_epochs):
        running_loss = 0.0
        for audio_batch, notes_batch in train_loader:
            audio_batch = audio_batch.to(device)
            notes_batch = notes_batch.to(device)
            audio_batch = downsample_along_dim(audio_batch, dim=2, stride=stride)
            notes_batch = downsample_along_dim(notes_batch, dim=1, stride=stride)
            optimizer.zero_grad()
            logits = model.forward(audio_batch)
            B, T_down, _ = logits.shape
            downsampled_labels = [
                downsample_labels(notes_batch[b].cpu().numpy(), T_down)
                for b in range(B)
            ]
            downsampled_labels = np.stack(downsampled_labels)
            downsampled_labels = torch.from_numpy(downsampled_labels).to(device)
            loss = criterion(logits, downsampled_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if (epoch + 1) % eval_interval == 0:
            avg_loss = running_loss / len(train_loader)
            model.addLosses(avg_loss)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        if (epoch + 1) % eval_interval == 0:
            eval_model(model, test_loader, device, stride=stride)

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


    audio_model = MultiScaleResidualCNN(
        n_notes=128,
        base_channels=64,
        num_blocks=3,
        block_kernel_sizes=[11, 11, 11,11,11],
        block_strides=[2, 2, 2,2,2],
        block_dilations=[1, 2, 2,1,1],
        multiscale_kernel_sizes=[5, 5, 5,5,5],
        use_multiscale=True
    )

    epochs=20

    train_model(
        model=audio_model,
        device=my_device,
        batch_size=8,
        num_epochs=epochs, #10000
        eval_interval=1,
        train_split=0.8,
    )

    plt.plot(range(epochs),audio_model.losses, label="Loss")
    plt.plot(range(epochs),audio_model.eval_losses, label="Eval Loss")
    plt.show()