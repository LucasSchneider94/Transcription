"""
Dataset class for piano transcription training.
Handles memory-efficient loading and random snippet sampling.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import os


class SnippetDataset(Dataset):
    """
    Dataset that randomly samples snippets from spectrograms and piano rolls.
    
    Features:
    - Memory-efficient loading using mmap
    - Random snippet sampling for data augmentation
    - File-based train/val splitting to prevent data leakage
    """
    
    def __init__(self, data_dir, snippet_frames, snippets_per_file=20, 
                 file_indices=None, file_list=None, seed=42):
        """
        Args:
            data_dir: Directory containing processed .npz and .npy files
            snippet_frames: Number of frames per snippet
            snippets_per_file: Number of snippets to sample per file per epoch
            file_indices: List of file indices for train/val split
            file_list: Pre-filtered list of files (optional)
            seed: Random seed for reproducible file selection
        """
        self.data_dir = data_dir
        self.snippet_frames = snippet_frames
        self.snippets_per_file = snippets_per_file
        self.file_list = []
        
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
        
        # Build file list with metadata
        for file_name in files:
            piano_roll_path = os.path.join(data_dir, file_name)
            spectrogram_path = piano_roll_path.replace("_piano_roll_with_pedals.npz", 
                                                       "_spectrogram.npy")
            
            if not os.path.exists(spectrogram_path):
                continue
            
            # Load metadata only (memory-efficient)
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
        
        print(f"Dataset: {len(self.file_list)} files, "
              f"{len(self)} snippets ({self.snippets_per_file} per file)")
    
    def __len__(self):
        return len(self.file_list) * self.snippets_per_file
    
    def __getitem__(self, idx):
        """Returns a random snippet from a file."""
        # Determine which file
        file_idx = idx // self.snippets_per_file
        file_info = self.file_list[file_idx]
        
        # Random snippet position
        start_frame = np.random.randint(0, file_info['max_start'] + 1)
        end_frame = start_frame + self.snippet_frames
        
        # Load snippet only (memory-efficient)
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
