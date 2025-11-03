"""
Dataset class for piano transcription training.
Optimized version with RAM pre-loading for faster training.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import os
from tqdm import tqdm
import pickle
from pathlib import Path


class PianoTranscriptionDataset(Dataset):
    """
    Dataset for piano transcription with onset, offset, and frame labels.
    Loads from .npz files and generates random snippets.
    """
    def __init__(self, data_dir, snippet_frames, snippets_per_file=20, 
                 file_indices=None, file_list=None, seed=42, 
                 fixed_snippets=False, preload_into_ram=False):
        """
        Args:
            data_dir: Directory containing .npz files
            snippet_frames: Number of frames per snippet
            snippets_per_file: Number of snippets to extract per file
            file_indices: List of file indices to use (for train/val split)
            file_list: Pre-sorted list of all files
            seed: Random seed for reproducibility
            fixed_snippets: If True, use fixed snippet positions
            preload_into_ram: If True, load all data into RAM
        """
        self.data_dir = data_dir
        self.snippet_frames = snippet_frames
        self.snippets_per_file = snippets_per_file
        self.fixed_snippets = fixed_snippets
        self.preload_into_ram = preload_into_ram
        
        # Get file list
        if file_list is not None and file_indices is not None:
            self.files = [os.path.join(data_dir, file_list[i]) for i in file_indices]
        else:
            self.files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                               if f.endswith('_piano_roll_with_pedals.npz')])
        
        self.num_files = len(self.files)
        self.rng = np.random.RandomState(seed)
        
        # Preload data if requested
        self.data_cache = {}
        if self.preload_into_ram:
            print(f"Preloading {self.num_files} files into RAM...")
            for file_path in tqdm(self.files):
                data = np.load(file_path)
                self.data_cache[file_path] = {
                    'spectrogram': data['spectrogram'],
                    'onset': data['onset'],
                    'offset': data['offset'],
                    'frame': data['frame'],
                    'pedal': data['pedal']
                }
            print(f"✓ Preloaded {len(self.data_cache)} files")
    
    def __len__(self):
        return self.num_files * self.snippets_per_file
    
    def __getitem__(self, idx):
        """
        Returns:
            Dictionary with:
                'spectrogram': (n_mels, snippet_frames)
                'onset': (snippet_frames, 88)
                'offset': (snippet_frames, 88)
                'frame': (snippet_frames, 88)
                'pedal': (snippet_frames, 1)
        """
        # Determine which file and snippet
        file_idx = idx // self.snippets_per_file
        snippet_idx = idx % self.snippets_per_file
        
        file_path = self.files[file_idx]
        
        # Load data from cache or disk
        if self.preload_into_ram:
            data = self.data_cache[file_path]
            spectrogram = data['spectrogram']
            onset = data['onset']
            offset = data['offset']
            frame = data['frame']
            pedal = data['pedal']
        else:
            data = np.load(file_path)
            spectrogram = data['spectrogram']
            onset = data['onset']
            offset = data['offset']
            frame = data['frame']
            pedal = data['pedal']
        
        # Extract snippet
        total_frames = spectrogram.shape[1]
        
        if total_frames <= self.snippet_frames:
            # Pad if too short
            pad_frames = self.snippet_frames - total_frames
            spectrogram = np.pad(spectrogram, ((0, 0), (0, pad_frames)), mode='constant')
            onset = np.pad(onset, ((0, pad_frames), (0, 0)), mode='constant')
            offset = np.pad(offset, ((0, pad_frames), (0, 0)), mode='constant')
            frame = np.pad(frame, ((0, pad_frames), (0, 0)), mode='constant')
            pedal = np.pad(pedal, ((0, pad_frames), (0, 0)), mode='constant')
            start_frame = 0
        else:
            # Random or fixed snippet extraction
            if self.fixed_snippets:
                # Evenly spaced snippets
                step = (total_frames - self.snippet_frames) // max(1, self.snippets_per_file - 1)
                start_frame = min(snippet_idx * step, total_frames - self.snippet_frames)
            else:
                # Random snippet
                start_frame = self.rng.randint(0, total_frames - self.snippet_frames + 1)
            
            end_frame = start_frame + self.snippet_frames
            spectrogram = spectrogram[:, start_frame:end_frame]
            onset = onset[start_frame:end_frame, :]
            offset = offset[start_frame:end_frame, :]
            frame = frame[start_frame:end_frame, :]
            pedal = pedal[start_frame:end_frame, :]
        
        return {
            'spectrogram': torch.FloatTensor(spectrogram),
            'onset': torch.FloatTensor(onset),
            'offset': torch.FloatTensor(offset),
            'frame': torch.FloatTensor(frame),
            'pedal': torch.FloatTensor(pedal)
        }
