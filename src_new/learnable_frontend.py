"""
Learnable audio frontend models - alternative to fixed mel spectrograms.
These models learn their own time-frequency representation from raw audio.
"""

import torch
import torch.nn as nn
import math


class LearnableSTFT(nn.Module):
    """
    Learnable Short-Time Fourier Transform.
    Instead of fixed mel filterbanks, learn optimal filters for piano transcription.
    
    Inspired by: "Learning filterbanks for spectral feature extraction" and
    "SincNet: Speaker Recognition from Raw Waveform"
    """
    def __init__(self, n_filters=264, kernel_size=2048, stride=480, sample_rate=48000):
        super().__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.stride = stride
        
        # Learnable convolutional filters (replacing FFT + Mel filterbank)
        # Each filter learns to extract a specific frequency pattern
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=n_filters,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False
        )
        
        # Initialize with mel-scale filters (warm start)
        self._init_as_mel_filterbank(sample_rate)
        
    def _init_as_mel_filterbank(self, sr):
        """Initialize filters to approximate mel filterbank."""
        import numpy as np
        
        # Create mel filterbank as initialization
        mel_points = np.linspace(
            self._hz_to_mel(27.5),  # A0
            self._hz_to_mel(4186),   # C8
            self.n_filters + 2
        )
        hz_points = self._mel_to_hz(mel_points)
        
        # Create triangular filters in frequency domain
        fft_freqs = np.fft.rfftfreq(self.kernel_size, 1/sr)
        filterbank = np.zeros((self.n_filters, len(fft_freqs)))
        
        for i in range(self.n_filters):
            left = hz_points[i]
            center = hz_points[i + 1]
            right = hz_points[i + 2]
            
            # Triangular filter
            for j, freq in enumerate(fft_freqs):
                if left <= freq <= center:
                    filterbank[i, j] = (freq - left) / (center - left)
                elif center <= freq <= right:
                    filterbank[i, j] = (right - freq) / (right - center)
        
        # Convert to time domain filters
        time_filters = np.fft.irfft(filterbank, n=self.kernel_size)
        
        # Apply window
        window = np.hanning(self.kernel_size)
        time_filters = time_filters * window
        
        # Set as initial weights
        with torch.no_grad():
            self.conv.weight.data = torch.from_numpy(time_filters).float().unsqueeze(1)
    
    @staticmethod
    def _hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)
    
    @staticmethod
    def _mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)
    
    def forward(self, x):
        """
        Args:
            x: Raw audio (batch, 1, time_samples)
        Returns:
            Learned spectrogram-like representation (batch, n_filters, time_frames)
        """
        # Apply learnable filters
        x = self.conv(x)  # (batch, n_filters, time_frames)
        
        # Apply non-linearity (like log compression in spectrograms)
        x = torch.log(torch.abs(x) + 1e-8)
        
        return x


class RawAudioTranscriptionModel(nn.Module):
    """
    Piano transcription model that learns its own representation from raw audio.
    No fixed FFT or mel filterbank - everything is learned.
    """
    def __init__(self, n_filters=264, kernel_size=2048, stride=480, 
                 hidden_size=256, num_heads=8, num_layers=4, 
                 num_outputs=91, dropout=0.1, sample_rate=48000):
        super().__init__()
        
        # Learnable frontend (replaces fixed mel spectrogram)
        self.frontend = LearnableSTFT(
            n_filters=n_filters,
            kernel_size=kernel_size,
            stride=stride,
            sample_rate=sample_rate
        )
        
        # Same architecture as before, but works on learned features
        # Additional CNN layers to refine learned features
        self.feature_refiner = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )
        
        # Calculate feature size after conv layers
        self.feature_size = 128 * (n_filters // 8)
        self.projection = nn.Linear(self.feature_size, hidden_size)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_size, dropout=dropout)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output
        self.output_projection = nn.Linear(hidden_size, num_outputs)
    
    def forward(self, x):
        """
        Args:
            x: Raw audio waveform (batch, 1, time_samples)
        Returns:
            Piano roll predictions (batch, time_frames, num_outputs)
        """
        # Learn features from raw audio
        x = self.frontend(x)  # (batch, n_filters, time_frames)
        
        # Add channel dimension for CNN
        x = x.unsqueeze(1)  # (batch, 1, n_filters, time_frames)
        
        # Refine features with CNN
        x = self.feature_refiner(x)  # (batch, 128, n_filters//8, time_frames)
        
        # Reshape for transformer
        batch_size, channels, freq, time = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()  # (batch, time, channels, freq)
        x = x.view(batch_size, time, -1)  # (batch, time, channels * freq)
        
        # Project to hidden size
        x = self.projection(x)  # (batch, time, hidden_size)
        
        # Transformer
        x = self.pos_encoder(x)
        x = self.transformer(x)
        
        # Output
        x = self.output_projection(x)
        x = torch.sigmoid(x)
        
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    def __init__(self, hidden_size, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


if __name__ == "__main__":
    # Test the models
    from config import CONFIG, NUM_OUTPUTS
    
    print("\n" + "="*80)
    print("COMPARING FIXED vs LEARNABLE AUDIO FRONTENDS")
    print("="*80)
    
    # Calculate input sizes
    sample_rate = 48000
    snippet_duration = 3.0
    audio_samples = int(sample_rate * snippet_duration)  # 144000 samples
    
    batch_size = 4
    
    # Test learnable model
    print("\n1. Learnable STFT Model:")
    learnable_model = RawAudioTranscriptionModel(
        n_filters=CONFIG['n_mels'],
        kernel_size=CONFIG['n_fft'],
        stride=CONFIG['hop_length'],
        hidden_size=CONFIG['hidden_size'],
        num_heads=CONFIG['num_heads'],
        num_layers=CONFIG['num_layers'],
        num_outputs=NUM_OUTPUTS,
        dropout=CONFIG['dropout'],
        sample_rate=sample_rate
    )
    
    # Create dummy raw audio input
    dummy_audio = torch.randn(batch_size, 1, audio_samples)
    output = learnable_model(dummy_audio)
    
    print(f"  Input:  {dummy_audio.shape} (raw audio)")
    print(f"  Output: {output.shape}")
    print(f"  Parameters: {sum(p.numel() for p in learnable_model.parameters()):,}")
    
    # Compare to fixed spectrogram model
    from model import PianoTranscriptionModel
    
    print("\n2. Fixed Mel Spectrogram Model:")
    fixed_model = PianoTranscriptionModel(
        n_mels=CONFIG['n_mels'],
        hidden_size=CONFIG['hidden_size'],
        num_heads=CONFIG['num_heads'],
        num_layers=CONFIG['num_layers'],
        num_outputs=NUM_OUTPUTS,
        dropout=CONFIG['dropout']
    )
    
    # This expects pre-computed spectrograms
    dummy_spec = torch.randn(batch_size, 1, CONFIG['n_mels'], CONFIG['snippet_frames'])
    output_fixed = fixed_model(dummy_spec)
    
    print(f"  Input:  {dummy_spec.shape} (pre-computed mel spectrogram)")
    print(f"  Output: {output_fixed.shape}")
    print(f"  Parameters: {sum(p.numel() for p in fixed_model.parameters()):,}")
    
    print("\n" + "="*80)
    print("ANALYSIS:")
    print("="*80)
    print("\nLearnable approach:")
    print("  ✓ Can learn optimal filters for piano (not human hearing)")
    print("  ✓ End-to-end differentiable")
    print("  ✓ Might find better time/frequency tradeoffs")
    print("  ✗ Requires more data to learn good features")
    print("  ✗ Harder to interpret what it learned")
    print("  ✗ ~25% more parameters (learnable filters)")
    print("\nFixed mel spectrogram:")
    print("  ✓ Proven to work, based on decades of research")
    print("  ✓ Works with smaller datasets")
    print("  ✓ Interpretable (can visualize spectrograms)")
    print("  ✗ Fixed time/frequency tradeoff")
    print("  ✗ Designed for human hearing, not piano")
    
    print("\n" + "="*80)
    print("RECOMMENDATION:")
    print("="*80)
    print("Start with fixed mel spectrograms UNLESS you have:")
    print("  1. Large dataset (10x more data)")
    print("  2. Strong compute resources")
    print("  3. Evidence that mel spectrograms are the bottleneck")
    print("\nBut it's worth experimenting once you have a solid baseline!")
    print("="*80 + "\n")
