"""
Dataset class for piano transcription training.
Optimized version with RAM pre-loading for faster training.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import os
from tqdm import tqdm


class SnippetDataset(Dataset):
    """
    Dataset that randomly samples snippets from spectrograms and piano rolls.
    
    Features:
    - RAM pre-loading for 10-20x faster data access
    - Random snippet sampling for data augmentation
    - File-based train/val splitting to prevent data leakage
    """
    
    def __init__(self, data_dir, snippet_frames, snippets_per_file=20, 
                 file_indices=None, file_list=None, seed=42, fixed_snippets=False,
                 preload_into_ram=True):
        """
        Args:
            data_dir: Directory containing processed .npz and .npy files
            snippet_frames: Number of frames per snippet
            snippets_per_file: Number of snippets to sample per file per epoch
            file_indices: List of file indices for train/val split
            file_list: Pre-filtered list of files (optional)
            seed: Random seed for reproducible file selection
            fixed_snippets: If True, use fixed positions (for overfitting tests)
            preload_into_ram: If True, load all data into RAM for faster access
        """
        self.data_dir = data_dir
        self.snippet_frames = snippet_frames
        self.snippets_per_file = snippets_per_file
        self.fixed_snippets = fixed_snippets
        self.preload_into_ram = preload_into_ram
        self.file_list = []
        
        # Storage for pre-loaded data
        self.spectrograms = []
        self.piano_rolls = []
        
        # Collect files
        np.random.seed(seed)
        if file_list is not None:
            all_files = file_list
        else:
            all_files = sorted([f for f in os.listdir(data_dir) 
                              if f.endswith("_piano_roll_with_pedals.npz")])
        
        # Filter by indices if provided
        if file_indices is not None:
            files = [all_files[i] for i in file_indices if i < len(all_files)]
        else:
            files = all_files
        
        # Build file list with metadata (and optionally pre-load)
        print(f"Loading dataset from {data_dir}...")
        if preload_into_ram:
            print("⚡ Pre-loading all files into RAM for faster training...")
        
        total_size_mb = 0
        
        for file_name in tqdm(files, desc="Loading files"):
            piano_roll_path = os.path.join(data_dir, file_name)
            spectrogram_path = piano_roll_path.replace("_piano_roll_with_pedals.npz", 
                                                       "_spectrogram.npy")
            
            if not os.path.exists(spectrogram_path):
                continue
            
            if preload_into_ram:
                # Load entire file into RAM
                spectrogram = np.load(spectrogram_path)
                piano_roll_data = np.load(piano_roll_path)
                piano_roll = piano_roll_data["piano_roll"]
                
                # Calculate memory usage
                total_size_mb += (spectrogram.nbytes + piano_roll.nbytes) / (1024 * 1024)
                
                min_frames = min(spectrogram.shape[1], piano_roll.shape[0])
                
                if min_frames >= snippet_frames:
                    # Store in RAM
                    self.spectrograms.append(spectrogram[:, :min_frames])
                    self.piano_rolls.append(piano_roll[:min_frames, :])
                    
                    file_info = {
                        'file_idx': len(self.spectrograms) - 1,  # Index into pre-loaded arrays
                        'max_frames': min_frames,
                        'max_start': min_frames - snippet_frames
                    }
                    self.file_list.append(file_info)
                
                # Cleanup
                if hasattr(piano_roll_data, 'close'):
                    piano_roll_data.close()
            else:
                # Memory-mapped loading (old method)
                spectrogram = np.load(spectrogram_path, mmap_mode='r')
                piano_roll_data = np.load(piano_roll_path, mmap_mode='r')
                piano_roll = piano_roll_data["piano_roll"]
                
                min_frames = min(spectrogram.shape[1], piano_roll.shape[0])
                
                if min_frames >= snippet_frames:
                    file_info = {
                        'spectrogram_path': spectrogram_path,
                        'piano_roll_path': piano_roll_path,
                        'max_frames': min_frames,
                        'max_start': min_frames - snippet_frames
                    }
                    self.file_list.append(file_info)
                
                # Cleanup
                del spectrogram
                if hasattr(piano_roll_data, 'close'):
                    piano_roll_data.close()
                del piano_roll_data
        
        np.random.seed(None)
        
        print(f"\nDataset loaded: {len(self.file_list)} files, "
              f"{len(self)} snippets ({self.snippets_per_file} per file)")
        if preload_into_ram:
            print(f"💾 RAM usage: {total_size_mb:.1f} MB")
        if fixed_snippets:
            print(f"🔒 FIXED snippets - same data every epoch (overfitting mode)")
        else:
            print(f"🎲 RANDOM snippets - different every epoch (generalization mode)")
    
    def __len__(self):
        return len(self.file_list) * self.snippets_per_file
    
    def __getitem__(self, idx):
        """Returns a snippet from a file (fixed or random position)."""
        # Determine which file
        file_idx = idx // self.snippets_per_file
        file_info = self.file_list[file_idx]
        
        if self.fixed_snippets:
            # FIXED: Use evenly-spaced positions (deterministic)
            snippet_idx = idx % self.snippets_per_file
            stride = file_info['max_start'] // max(self.snippets_per_file, 1)
            start_frame = snippet_idx * stride
        else:
            # RANDOM: Different position each epoch
            start_frame = np.random.randint(0, file_info['max_start'] + 1)
        
        end_frame = start_frame + self.snippet_frames
        
        if self.preload_into_ram:
            # Fast access from RAM
            data_idx = file_info['file_idx']
            spectrogram_snippet = self.spectrograms[data_idx][:, start_frame:end_frame]
            piano_roll_snippet = self.piano_rolls[data_idx][start_frame:end_frame, :]
        else:
            # Load from disk (old method)
            spectrogram_mmap = np.load(file_info['spectrogram_path'], mmap_mode='r')
            spectrogram_snippet = np.array(spectrogram_mmap[:, start_frame:end_frame])
            del spectrogram_mmap
            
            piano_roll_data = np.load(file_info['piano_roll_path'], mmap_mode='r')
            piano_roll = piano_roll_data["piano_roll"]
            piano_roll_snippet = np.array(piano_roll[start_frame:end_frame, :])
            if hasattr(piano_roll_data, 'close'):
                piano_roll_data.close()
            del piano_roll_data
        
        # Convert to tensors
        spectrogram_tensor = torch.tensor(spectrogram_snippet, dtype=torch.float32).unsqueeze(0)
        piano_roll_tensor = torch.tensor(piano_roll_snippet, dtype=torch.float32)
        
        return spectrogram_tensor, piano_roll_tensor
