import os
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
from torch.utils.data import Dataset
from scipy.sparse import load_npz
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR

###############################################################################
# 1) Helper functions
###############################################################################
# def downsample_labels(labels: np.ndarray, target_length: int) -> np.ndarray:
#     orig_len = len(labels)
#     if target_length == orig_len:
#         return labels
#     idxs = np.linspace(0, orig_len - 1, target_length, dtype=np.int32)
#     return labels[idxs]

# def downsample_along_dim(x: torch.Tensor, dim: int, stride: int) -> torch.Tensor:
#     """
#     Downsamples tensor x along dimension dim to length out_len
#     by taking every n-th element. (No interpolation/antialias.)
#     """
#     # Generate indices
#     indices = torch.arange(0, x.size(dim), stride, device=x.device)
#     return x.index_select(dim, indices)

# def actualsize(input_obj):
#     memory_size = 0
#     ids = set()
#     objects = [input_obj]
#     while objects:
#         new = []
#         for obj in objects:
#             if id(obj) not in ids:
#                 ids.add(id(obj))
#                 memory_size += sys.getsizeof(obj)
#                 new.append(obj)
#         objects = gc.get_referents(*new)
#     return memory_size

###############################################################################
# 2) Dataset
###############################################################################

class WavRollEfficientDataset(Dataset):
    def __init__(self, root_dir, window_size_frames=100, roll_fps=100):
        super().__init__()
        self.root_dir = root_dir
        self.window_size_frames = window_size_frames
        self.roll_fps = roll_fps
        self.data = []

        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Root folder does not exist: {root_dir}")

        # Scan and store only metadata (paths, lengths)
        for current_dir, _, files in tqdm(os.walk(root_dir)):
            for file_name in files:
                if file_name.lower().endswith(".wav"):
                    try:
                        wav_path = os.path.join(current_dir, file_name)
                        roll_path = os.path.join(
                            current_dir,
                            f"{os.path.splitext(file_name)[0]}_piano_roll_sparse.npz"
                        )

                        if not os.path.exists(roll_path):
                            print(f"Skipping {wav_path}, missing piano roll file.")
                            continue

                        # Optionally preload duration (useful for sanity checking)
                        # but here we skip that to keep it fast.
                        self.data.append({
                            "wav_path": wav_path,
                            "roll_path": roll_path
                        })

                    except Exception as e:
                        print(f"Error processing {file_name}: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        item = self.data[idx]
        wav_path = item["wav_path"]
        roll_path = item["roll_path"]

        # Load sparse piano roll (shape: T x 128)
        piano_roll_sp = load_npz(roll_path)
        piano_roll = torch.from_numpy(piano_roll_sp.toarray()).float()

        # Load audio
        waveform, sample_rate = torchaudio.load(wav_path) 
        waveform = waveform.squeeze(0) 

        # Convert stereo to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # shape (1, N)
        
        if waveform.shape[0] == 0 or waveform.shape[1] == 0:
            raise RuntimeError(f"Invalid waveform shape {waveform.shape} in {wav_path}")
        
       
        samples_per_frame = sample_rate / self.roll_fps
        total_frames = piano_roll.shape[0]
        total_samples = int(samples_per_frame * total_frames)

        # Ensure waveform is at least long enough
        if waveform.shape[1] < total_samples:
            pad_size = total_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))
        else:
            # Truncate or round waveform to match exactly
            waveform = waveform[:, :total_samples]
        

        # Select a random aligned window
        max_start_frame = total_frames - self.window_size_frames
        if max_start_frame <= 0:
            # Sequence too short — return full
            start_frame = 0
            end_frame = total_frames
        else:
            start_frame = np.random.randint(0, max_start_frame + 1)
            end_frame = start_frame + self.window_size_frames

        start_sample = int(round(start_frame * samples_per_frame))
        end_sample = int(round(end_frame * samples_per_frame))

        audio_window = waveform[0][start_sample:end_sample] 
        roll_window = piano_roll[start_frame:end_frame]  

        # Most recent version:
        # audio_chunks = audio_window.unfold(dimension=0, size=int(samples_per_frame), step=int(samples_per_frame))  # shape: [T, 192]
        
        # print(f'Samples per frame = {samples_per_frame}')
        # print(f'T = {total_frames}')
        # print(f'waveform duration total: {total_samples/sample_rate/60}mins')

        # print(f'audio chunks shape: {audio_chunks.shape}')
        # print(f'piano roll shape: {roll_window.shape}')
        chunk_size = 1024
        hop_size = int(samples_per_frame)  # step size is still 192 if roll_fps = 100 and sample_rate = 48k

        # Pad audio to handle edges
        pad = chunk_size // 2
        padded_audio = torch.nn.functional.pad(audio_window.unsqueeze(0), (pad, pad), mode='reflect').squeeze(0)
        audio_chunks = torch.stack([
            padded_audio[i * hop_size : i * hop_size + chunk_size]
            for i in range(self.window_size_frames)
        ], dim=0)  # shape: [T, 1024]
        
        return audio_chunks, roll_window
    
###############################################################################
# 3) Evaluation function
###############################################################################
def eval_model(model, dataloader, device, stride):
    model.eval()
    #criterion = nn.BCEWithLogitsLoss()
    pos_weight_value=40
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value).to(device))
    total_loss = 0.0
    with torch.no_grad():
        for audio_batch, notes_batch in dataloader:
            audio_batch = audio_batch.to(device)  # shape: [B, T_samples]
            audio_batch = audio_batch.permute(0, 2, 1)  # -> [B, 192, T]
            notes_batch = notes_batch.to(device)  # shape: [B, T_frames, 128]

            logits = model(audio_batch)  # expected shape: [B, T_down, 128]
            # Check shapes
            # notes_batch: [B, T, 128]
            B, T, N = notes_batch.shape
            pad_len = (8 - (T % 8)) % 8  # how many to pad at the end
 


            # Pad only if needed
            if pad_len > 0:
                notes_batch = F.pad(notes_batch, (0, 0, 0, pad_len))  # pad time dim

            # Now apply avg_pool1d
            notes_batch_down = F.avg_pool1d(notes_batch.permute(0, 2, 1).float(), kernel_size=8, stride=8).permute(0, 2, 1)
            
            loss = criterion(logits, notes_batch_down)
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
    stride=10,
    roll_fps=100,
    root_dir="../maestro-v3.0.0/2018"
):
    model.to(device)

    dataset = WavRollEfficientDataset(root_dir=root_dir, window_size_frames=30*100, roll_fps=roll_fps)

    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch

    scheduler = OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=total_steps,
        pct_start=0.3,        # 30% ramp up, 70% ramp down
        anneal_strategy='cos', # cosine annealing
        div_factor=25.0,       # initial LR = max_lr / div_factor
        final_div_factor=1e4,  # final LR = max_lr / final_div_factor
    )

    #criterion = nn.BCEWithLogitsLoss()
    pos_weight_value=40
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value).to(device))
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for audio_batch, notes_batch in train_loader:
            audio_batch = audio_batch.to(device)  # shape: [B, T_samples]
            audio_batch = audio_batch.permute(0, 2, 1)  # -> [B, 192, T]
            notes_batch = notes_batch.to(device)  # shape: [B, T_frames, 128]

            logits = model(audio_batch)  # expected shape: [B, T_down, 128]
            # Check shapes
            # notes_batch: [B, T, 128]
            B, T, N = notes_batch.shape
            pad_len = (8 - (T % 8)) % 8  # how many to pad at the end

            # Pad only if needed
            if pad_len > 0:
                notes_batch = F.pad(notes_batch, (0, 0, 0, pad_len))  # pad time dim

            # Now apply avg_pool1d
            notes_batch_down = F.avg_pool1d(notes_batch.permute(0, 2, 1).float(), kernel_size=8, stride=8).permute(0, 2, 1)
            
            #print("Logits mean:", logits.mean().item(), "std:", logits.std().item())
 
            probs = torch.sigmoid(logits)  # [B, T, 128]
            preds = (probs > 0.5).float()

            # Diagnostics:
            # print("Predicted notes per batch:", preds.sum().item())
            # #print("Sigmoid mean:", probs.mean().item())
            # #print(f'notesbatchdown={notes_batch_down.shape}')
            # print("Ground truth avg active notes per timestep:",notes_batch.float().mean(dim=(0, 1)).sum().item())  # Sum over notes
            # # print(f"Logits shape: {logits.shape}")

            # #assert notes_batch.shape[0] == B and notes_batch.shape[1] == T_down
            # plt.imshow(audio_batch[0].squeeze().cpu(), aspect='auto', origin='lower')
            # plt.title("Input audio vector (reshaped)")
            # plt.show()

            # plt.imshow(notes_batch[0].cpu().T, aspect='auto', origin='lower')
            # plt.title("Ground truth piano roll")
            # plt.show()
            loss = criterion(logits, notes_batch_down)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        if (epoch + 1) % eval_interval == 0:
            avg_loss = running_loss / len(train_loader)
            model.addLosses(avg_loss)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
            current_lr = scheduler.get_last_lr()[0]
            print(f"LR: {current_lr:.6f}")
            model.addLR(current_lr)


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
        in_channels=1024,
        num_blocks=3,
        block_kernel_sizes=[11, 11, 11,11,11],
        block_strides=[2, 2, 2,2,2],
        block_dilations=[1, 2, 2,1,1],
        multiscale_kernel_sizes=[5, 5, 5,5,5],
        use_multiscale=True
    )

    epochs=1200

    train_model(
        model=audio_model,
        device=my_device,
        batch_size=8,
        num_epochs=epochs, #10000
        eval_interval=1,
        learning_rate=2e-4,
        train_split=0.8,
    )
    # Save the model
    torch.save(audio_model.state_dict(),'audio_model_MAESTRO_posweight40_1500epochs.pt')
    
    plt.plot(range(epochs),audio_model.losses, label="Loss")
    plt.plot(range(epochs),audio_model.eval_losses, label="Eval Loss")
    plt.legend(['Loss','Eval Loss'])
    plt.xlabel('epochs')
    plt.ylabel('BCE with logits loss')
    plt.yscale('log')
    plt.show()
