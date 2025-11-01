"""
Dataset class for raw audio piano transcription.
Loads raw audio frames instead of spectrograms.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import os


class RawAudioSnippetDataset(Dataset):
    """
    Dataset that loads raw audio frame snippets and piano rolls.
    
    Instead of spectrograms, loads overlapping audio windows.
    """
    
    def __init__(self, data_dir, snippet_frames, snippets_per_file=20, 
                 file_indices=None, file_list=None, seed=42):
        """
        Args:
            data_dir: Directory containing processed audio_frames.npy and piano_roll files
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
            # Look for audio_frames instead of spectrogram
            audio_frames_path = piano_roll_path.replace("_piano_roll_with_pedals.npz", 
                                                        "_audio_frames.npy")
            
            if not os.path.exists(audio_frames_path):
                continue
            
            # Load metadata only (memory-efficient)
            audio_frames = np.load(audio_frames_path, mmap_mode='r')
            piano_roll_data = np.load(piano_roll_path, mmap_mode='r')
            piano_roll = piano_roll_data["piano_roll"]
            
            min_frames = min(audio_frames.shape[0], piano_roll.shape[0])
            
            if min_frames >= snippet_frames:
                file_info = {
                    'audio_frames_path': audio_frames_path,
                    'piano_roll_path': piano_roll_path,
                    'max_frames': min_frames,
                    'max_start': min_frames - snippet_frames
                }
                self.file_list.append(file_info)
            
            # Cleanup
            del audio_frames
            if hasattr(piano_roll_data, 'close'):
                piano_roll_data.close()
            del piano_roll_data
        
        np.random.seed(None)
        
        print(f"Dataset: {len(self.file_list)} files, "
              f"{len(self)} snippets ({self.snippets_per_file} per file)")
    
    def __len__(self):
        return len(self.file_list) * self.snippets_per_file
    
    def __getitem__(self, idx):
        """Returns a random snippet of raw audio frames and piano roll."""
        # Determine which file
        file_idx = idx // self.snippets_per_file
        file_info = self.file_list[file_idx]
        
        # Random snippet position
        start_frame = np.random.randint(0, file_info['max_start'] + 1)
        end_frame = start_frame + self.snippet_frames
        
        # Load snippet only (memory-efficient)
        audio_frames_mmap = np.load(file_info['audio_frames_path'], mmap_mode='r')
        audio_snippet = np.array(audio_frames_mmap[start_frame:end_frame, :])
        del audio_frames_mmap
        
        piano_roll_data = np.load(file_info['piano_roll_path'], mmap_mode='r')
        piano_roll = piano_roll_data["piano_roll"]
        piano_roll_snippet = np.array(piano_roll[start_frame:end_frame, :])
        if hasattr(piano_roll_data, 'close'):
            piano_roll_data.close()
        del piano_roll_data
        
        # Convert to tensors
        # audio_snippet shape: (snippet_frames, context_samples)
        audio_tensor = torch.tensor(audio_snippet, dtype=torch.float32)
        piano_roll_tensor = torch.tensor(piano_roll_snippet, dtype=torch.float32)
        
        return audio_tensor, piano_roll_tensor
