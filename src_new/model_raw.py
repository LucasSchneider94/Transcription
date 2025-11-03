"""
Raw audio piano transcription model.
Uses 1D convolutions on raw waveform context windows instead of spectrograms.
"""

import torch
import torch.nn as nn
import math


class RawAudio1DCNN(nn.Module):
    """
    1D CNN that processes raw audio context windows.
    Extracts frequency-like features directly from waveform.
    """
    def __init__(self, context_samples=2048, hidden_size=256):
        super().__init__()
        
        # 1D convolutions on raw audio windows
        # These learn to extract frequency features (like a learnable filterbank)
        self.conv_layers = nn.Sequential(
            # First layer: learn basic patterns
            nn.Conv1d(1, 32, kernel_size=64, stride=4, padding=32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            
            # Second layer: combine patterns
            nn.Conv1d(32, 64, kernel_size=32, stride=2, padding=16),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            
            # Third layer: high-level features
            nn.Conv1d(64, 128, kernel_size=16, stride=2, padding=8),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        
        # Calculate output size after convolutions
        # Starting with context_samples (e.g., 2048)
        # After conv1: (2048 + 2*32 - 64) / 4 + 1 = 512
        # After pool1: 512 / 4 = 128
        # After conv2: (128 + 2*16 - 32) / 2 + 1 = 65
        # After pool2: 65 / 4 = 16
        # After conv3: (16 + 2*8 - 16) / 2 + 1 = 9
        # After pool3: 9 / 2 = 4
        self.feature_size = 128 * 4  # 512 features
        
        self.projection = nn.Linear(self.feature_size, hidden_size)
        
    def forward(self, x):
        """
        Args:
            x: Raw audio windows (batch, time_frames, context_samples)
        Returns:
            Features for each frame (batch, time_frames, hidden_size)
        """
        batch_size, time_frames, context_samples = x.shape
        
        # Reshape to process all frames
        x = x.view(batch_size * time_frames, 1, context_samples)  # (batch*time, 1, context)
        
        # Extract features with 1D CNN
        x = self.conv_layers(x)  # (batch*time, 128, 4)
        
        # Flatten
        x = x.view(batch_size * time_frames, -1)  # (batch*time, 512)
        
        # Project to hidden size
        x = self.projection(x)  # (batch*time, hidden_size)
        
        # Reshape back to sequence
        x = x.view(batch_size, time_frames, -1)  # (batch, time, hidden_size)
        
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


class RawAudioPianoTranscriptionModel(nn.Module):
    """
    Piano transcription from raw audio.
    Architecture: 1D CNN (feature extraction) + Transformer (temporal modeling)
    """
    def __init__(self, context_samples=2048, hidden_size=256, num_heads=8, 
                 num_layers=4, num_outputs=91, dropout=0.1):
        super().__init__()
        
        # 1D CNN for raw audio feature extraction
        self.feature_extractor = RawAudio1DCNN(context_samples, hidden_size)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_size, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size, num_outputs)
        
    def forward(self, x):
        """
        Args:
            x: Raw audio context windows (batch, time_frames, context_samples)
        Returns:
            Piano roll predictions (batch, time_frames, num_outputs)
        """
        # Extract features from raw audio
        x = self.feature_extractor(x)  # (batch, time_frames, hidden_size)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Temporal modeling with transformer
        x = self.transformer(x)
        
        # Output projection
        x = self.output_projection(x)
        x = torch.sigmoid(x)
        
        return x


if __name__ == "__main__":
    from config import CONFIG, NUM_OUTPUTS
    
    print("\n" + "="*80)
    print("RAW AUDIO MODEL TEST")
    print("="*80)
    
    # Create model
    model = RawAudioPianoTranscriptionModel(
        context_samples=CONFIG['n_fft'],
        hidden_size=CONFIG['hidden_size'],
        num_heads=CONFIG['num_heads'],
        num_layers=CONFIG['num_layers'],
        num_outputs=NUM_OUTPUTS,
        dropout=CONFIG['dropout']
    )
    
    # Test input
    batch_size = 4
    time_frames = CONFIG['snippet_frames']  # 200 frames (2 seconds at 100fps)
    context_samples = CONFIG['n_fft']  # 2048 samples per frame
    
    dummy_input = torch.randn(batch_size, time_frames, context_samples)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"  Batch size: {batch_size}")
    print(f"  Time frames: {time_frames} (at {CONFIG['roll_fps']} fps)")
    print(f"  Context window: {context_samples} samples")
    
    # Forward pass
    output = model(dummy_input)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"  Piano roll predictions: {time_frames} frames × {NUM_OUTPUTS} outputs")
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n" + "="*80)
    print("Model ready for training on raw audio!")
    print("="*80 + "\n")
