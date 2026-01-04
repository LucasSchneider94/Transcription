"""
Dataset class for piano transcription training.
Optimized version with RAM pre-loading for faster training.
Uses onset + duration approach instead of onset + offset.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import os
from tqdm import tqdm
import pickle
from pathlib import Path

# Duration bin edges (in seconds) - logarithmic scale
# Must match data_preparation.py
DURATION_BINS = np.array([0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, np.inf])
NUM_DURATION_BINS = len(DURATION_BINS) - 1  # 8 bins

def duration_to_bin(duration):
    """
    Convert duration in seconds to bin index.
    
    Args:
        duration: Duration in seconds (scalar or array)
        
    Returns:
        Bin index (0 to NUM_DURATION_BINS-1)
    """
    duration = np.asarray(duration)
    bins = np.digitize(duration, DURATION_BINS[1:-1])  # Returns 0 to NUM_DURATION_BINS-1
    return bins

def duration_to_log_duration(duration, eps=1e-6):
    """
    Convert duration to log-duration for regression.
    Uses log(duration + eps) for numerical stability.
    
    Args:
        duration: Duration in seconds (scalar or array)
        eps: Small epsilon to avoid log(0)
        
    Returns:
        Log-duration
    """
    return np.log(np.maximum(duration, eps))

def log_duration_to_duration(log_duration):
    """
    Convert log-duration back to duration.
    
    Args:
        log_duration: Log-duration values
        
    Returns:
        Duration in seconds
    """
    return np.exp(log_duration)


class PianoTranscriptionDataset(Dataset):
    """
    Dataset for piano transcription with onset, duration, and frame labels.
    Loads from .npz files and generates random snippets.
    """
    def __init__(self, data_dir, snippet_frames, snippets_per_file=20, 
                 file_indices=None, file_list=None, seed=42, 
                 fixed_snippets=False, preload_into_ram=False,
                 duration_mode='bins', clip_duration=True):
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
            duration_mode: 'bins' for classification, 'log' for log-duration regression, 'linear' for linear regression
            clip_duration: If True, clip target durations to snippet boundaries
        """
        self.data_dir = data_dir
        self.snippet_frames = snippet_frames
        self.snippets_per_file = snippets_per_file
        self.fixed_snippets = fixed_snippets
        self.preload_into_ram = preload_into_ram
        self.duration_mode = duration_mode
        self.clip_duration = clip_duration
        
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
                    'duration': data['duration'],
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
                'duration': (snippet_frames, 88) - format depends on duration_mode
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
            duration = data['duration']
            frame = data['frame']
            pedal = data['pedal']
        else:
            data = np.load(file_path)
            spectrogram = data['spectrogram']
            onset = data['onset']
            duration = data['duration']
            frame = data['frame']
            pedal = data['pedal']
        
        # Extract snippet
        total_frames = spectrogram.shape[1]
        
        if total_frames <= self.snippet_frames:
            # Pad if too short
            pad_frames = self.snippet_frames - total_frames
            spectrogram = np.pad(spectrogram, ((0, 0), (0, pad_frames)), mode='constant')
            onset = np.pad(onset, ((0, pad_frames), (0, 0)), mode='constant')
            duration = np.pad(duration, ((0, pad_frames), (0, 0)), mode='constant')
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
            duration = duration[start_frame:end_frame, :]
            frame = frame[start_frame:end_frame, :]
            pedal = pedal[start_frame:end_frame, :]
        
        # Clip duration to snippet boundaries if requested
        if self.clip_duration:
            # Create time indices [0, 1, ..., T-1]
            time_indices = np.arange(self.snippet_frames).reshape(-1, 1) # (T, 1)
            # Max duration in seconds (assuming 100 fps)
            max_seconds_left = (self.snippet_frames - time_indices) / 100.0
            
            # Clip duration
            duration = np.minimum(duration, max_seconds_left)
            duration = np.maximum(duration, 0)

        # Process duration based on mode
        if self.duration_mode == 'bins':
            # Convert exact durations to bin indices for classification
            duration_processed = duration_to_bin(duration).astype(np.int64)
        elif self.duration_mode == 'log':
            # Convert to log-duration for regression
            duration_processed = duration_to_log_duration(duration).astype(np.float32)
        elif self.duration_mode == 'linear':
            # Keep as-is for linear regression
            duration_processed = duration.astype(np.float32)
        else:
            raise ValueError(f"Unknown duration_mode: {self.duration_mode}")
        
        return {
            'spectrogram': torch.FloatTensor(spectrogram),
            'onset': torch.FloatTensor(onset),
            'duration': torch.from_numpy(duration_processed),
            'frame': torch.FloatTensor(frame),
            'pedal': torch.FloatTensor(pedal)
        }
