"""
Dataset class for piano transcription training.
Supports onset-aware snippet sampling and optional spectrogram normalization.
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# Duration bin edges (in seconds) - logarithmic scale.
DURATION_BINS = np.array([0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, np.inf])
NUM_DURATION_BINS = len(DURATION_BINS) - 1


def duration_to_bin(duration):
    duration = np.asarray(duration)
    return np.digitize(duration, DURATION_BINS[1:-1])


def duration_to_log_duration(duration, eps=1e-6):
    return np.log(np.maximum(duration, eps))


def log_duration_to_duration(log_duration):
    return np.exp(log_duration)


class PianoTranscriptionDataset(Dataset):
    """
    Dataset for piano transcription with onset, duration, and frame labels.
    """

    def __init__(
        self,
        data_dir,
        snippet_frames,
        snippets_per_file=20,
        file_indices=None,
        file_list=None,
        seed=42,
        fixed_snippets=False,
        preload_into_ram=False,
        duration_mode="bins",
        clip_duration=True,
        sampling_mode="random",
        onset_sampling_min_active_ratio=0.5,
        normalization_stats=None,
    ):
        self.data_dir = data_dir
        self.snippet_frames = int(snippet_frames)
        self.snippets_per_file = int(snippets_per_file)
        self.fixed_snippets = fixed_snippets
        self.preload_into_ram = preload_into_ram
        self.duration_mode = duration_mode
        self.clip_duration = clip_duration
        self.sampling_mode = sampling_mode
        self.onset_sampling_min_active_ratio = float(onset_sampling_min_active_ratio)
        self.normalization_stats = normalization_stats or {}

        if file_list is not None and file_indices is not None:
            self.files = [os.path.join(data_dir, file_list[i]) for i in file_indices]
        else:
            self.files = sorted(
                [
                    os.path.join(data_dir, f)
                    for f in os.listdir(data_dir)
                    if f.endswith("_piano_roll_with_pedals.npz")
                ]
            )

        self.num_files = len(self.files)
        self.rng = np.random.RandomState(seed)

        self.data_cache = {}
        if self.preload_into_ram:
            print(f"Preloading {self.num_files} files into RAM...")
            for file_path in tqdm(self.files):
                with np.load(file_path) as data:
                    self.data_cache[file_path] = {
                        "spectrogram": data["spectrogram"],
                        "onset": data["onset"],
                        "duration": data["duration"],
                        "frame": data["frame"],
                        "pedal": data["pedal"],
                    }
            print(f"Loaded {len(self.data_cache)} files into RAM")

    def __len__(self):
        return self.num_files * self.snippets_per_file

    def _load_file(self, file_path):
        if self.preload_into_ram:
            return self.data_cache[file_path]

        with np.load(file_path) as data:
            return {
                "spectrogram": data["spectrogram"],
                "onset": data["onset"],
                "duration": data["duration"],
                "frame": data["frame"],
                "pedal": data["pedal"],
            }

    def _get_start_random(self, total_frames):
        return self.rng.randint(0, total_frames - self.snippet_frames + 1)

    def _get_start_fixed(self, total_frames, snippet_idx):
        step = (total_frames - self.snippet_frames) // max(1, self.snippets_per_file - 1)
        return min(snippet_idx * step, total_frames - self.snippet_frames)

    def _get_start_onset_aware(self, onset, total_frames, snippet_idx):
        if self.fixed_snippets:
            return self._get_start_fixed(total_frames, snippet_idx)

        if self.sampling_mode != "onset_aware":
            return self._get_start_random(total_frames)

        choose_onset_window = self.rng.rand() < self.onset_sampling_min_active_ratio
        if not choose_onset_window:
            return self._get_start_random(total_frames)

        onset_sum = onset.sum(axis=1)
        onset_frames = np.where(onset_sum > 0)[0]
        if onset_frames.size == 0:
            return self._get_start_random(total_frames)

        pivot = int(onset_frames[self.rng.randint(0, onset_frames.size)])
        left = max(0, pivot - self.snippet_frames + 1)
        right = min(pivot, total_frames - self.snippet_frames)
        if left > right:
            return self._get_start_random(total_frames)

        return self.rng.randint(left, right + 1)

    def _apply_normalization(self, spectrogram):
        mode = self.normalization_stats.get("mode", "none")
        if mode != "global":
            return spectrogram.astype(np.float32)

        mean = float(self.normalization_stats.get("mean", 0.0))
        std = float(self.normalization_stats.get("std", 1.0))
        std = std if std > 1e-8 else 1.0
        return ((spectrogram - mean) / std).astype(np.float32)

    def __getitem__(self, idx):
        file_idx = idx // self.snippets_per_file
        snippet_idx = idx % self.snippets_per_file
        file_path = self.files[file_idx]

        data = self._load_file(file_path)
        spectrogram = data["spectrogram"]
        onset = data["onset"]
        duration = data["duration"]
        frame = data["frame"]
        pedal = data["pedal"]

        total_frames = spectrogram.shape[1]
        if total_frames <= self.snippet_frames:
            pad_frames = self.snippet_frames - total_frames
            spectrogram = np.pad(spectrogram, ((0, 0), (0, pad_frames)), mode="constant")
            onset = np.pad(onset, ((0, pad_frames), (0, 0)), mode="constant")
            duration = np.pad(duration, ((0, pad_frames), (0, 0)), mode="constant")
            frame = np.pad(frame, ((0, pad_frames), (0, 0)), mode="constant")
            pedal = np.pad(pedal, ((0, pad_frames), (0, 0)), mode="constant")
        else:
            start_frame = self._get_start_onset_aware(onset, total_frames, snippet_idx)
            end_frame = start_frame + self.snippet_frames
            spectrogram = spectrogram[:, start_frame:end_frame]
            onset = onset[start_frame:end_frame, :]
            duration = duration[start_frame:end_frame, :]
            frame = frame[start_frame:end_frame, :]
            pedal = pedal[start_frame:end_frame, :]

        if self.clip_duration:
            time_indices = np.arange(self.snippet_frames).reshape(-1, 1)
            max_seconds_left = (self.snippet_frames - time_indices) / 100.0
            duration = np.minimum(duration, max_seconds_left)
            duration = np.maximum(duration, 0.0)

        if self.duration_mode == "bins":
            duration_processed = duration_to_bin(duration).astype(np.int64)
        elif self.duration_mode == "log":
            duration_processed = duration_to_log_duration(duration).astype(np.float32)
        elif self.duration_mode == "linear":
            duration_processed = duration.astype(np.float32)
        else:
            raise ValueError(f"Unknown duration_mode: {self.duration_mode}")

        spectrogram = self._apply_normalization(spectrogram)

        return {
            "spectrogram": torch.from_numpy(spectrogram).float(),
            "onset": torch.from_numpy(onset).float(),
            "duration": torch.from_numpy(duration_processed),
            "frame": torch.from_numpy(frame).float(),
            "pedal": torch.from_numpy(pedal).float(),
        }
