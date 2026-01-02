import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from config import CONFIG

class ConvolutionalFeatureExtractorLegacy(nn.Module):
    """
    LEGACY: Old CNN architecture for loading checkpoints from training_run_001-008.
    Use this for inference on old models.
    """
    def __init__(self, n_mels, hidden_size):
        super(ConvolutionalFeatureExtractorLegacy, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # First conv block - OLD SIZE
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            
            # Second conv block - OLD SIZE
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            
            # Third conv block - OLD SIZE
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )
        
        # Calculate output feature size after conv layers
        self.feature_size = 128 * (n_mels // 8)  # After 3 pooling layers
        self.projection = nn.Linear(self.feature_size, hidden_size)
        
    def forward(self, x):
        # x shape: (batch_size, 1, n_mels, time_frames)
        x = self.conv_layers(x)  # (batch_size, 128, n_mels//8, time_frames)
        
        # Reshape: (batch_size, time_frames, 128 * n_mels//8)
        batch_size, channels, freq, time = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(batch_size, time, -1)
        
        # Project to hidden_size
        x = self.projection(x)
        return x


class ConvolutionalFeatureExtractor(nn.Module):
    """
    Convolutional layers to extract local features from spectrograms.
    Increased capacity for better feature learning.
    """
    def __init__(self, n_mels, hidden_size):
        super(ConvolutionalFeatureExtractor, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # First conv block - INCREASED channels
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=(1, 1)),  # 32 → 64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            
            # Second conv block - INCREASED channels
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),  # 64 → 128
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            
            # Third conv block - INCREASED channels
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1)),  # 128 → 256
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            
            # NEW: Fourth conv block for more depth
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )
        
        # Calculate output feature size after conv layers
        self.feature_size = 256 * (n_mels // 16)  # After 4 pooling layers (was // 8)
        self.projection = nn.Linear(self.feature_size, hidden_size)
        
    def forward(self, x):
        # x shape: (batch_size, 1, n_mels, time_frames)
        x = self.conv_layers(x)  # (batch_size, 256, n_mels//16, time_frames)
        
        # Reshape: (batch_size, time_frames, 256 * n_mels//16)
        batch_size, channels, freq, time = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(batch_size, time, -1)
        
        # Project to hidden_size
        x = self.projection(x)
        return x


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer to capture temporal position information.
    """
    def __init__(self, hidden_size, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, hidden_size)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_size)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PianoTranscriptionModelLegacy(nn.Module):
    """
    LEGACY: Model for loading old checkpoints (training_run_001-008).
    Uses old CNN architecture (32→64→128, 3 layers).
    """
    def __init__(self, n_mels=128, hidden_size=256, num_heads=8, num_layers=4, 
                 num_outputs=91, dropout=0.1):
        super(PianoTranscriptionModelLegacy, self).__init__()
        
        # OLD Convolutional feature extractor
        self.feature_extractor = ConvolutionalFeatureExtractorLegacy(n_mels, hidden_size)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_size, dropout=dropout)
        
        # Transformer encoder layers
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
        """Forward pass."""
        x = self.feature_extractor(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = self.output_projection(x)
        x = torch.sigmoid(x)
        return x


class PianoTranscriptionModel(nn.Module):
    """
    Piano transcription model with CNN-Transformer architecture.
    Outputs two heads: onset and duration predictions.
    Duration can be either classification (bins) or regression (log/linear).
    """
    def __init__(self, 
                 input_features=CONFIG['n_mels'],
                 num_keys=CONFIG['num_keys'],
                 cnn_channels=[32, 64, 128],
                 transformer_dim=256,
                 num_heads=8,
                 num_layers=6,
                 dropout=0.1,
                 duration_mode='bins',
                 num_duration_bins=8):
        super(PianoTranscriptionModel, self).__init__()
        
        self.duration_mode = duration_mode
        self.num_duration_bins = num_duration_bins
        
        # Convolutional feature extractor
        self.feature_extractor = ConvolutionalFeatureExtractor(input_features, transformer_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(transformer_dim, dropout=dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dim_feedforward=transformer_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Two output heads: onset and duration
        self.onset_head = nn.Linear(transformer_dim, num_keys)
        
        if duration_mode == 'bins':
            # Classification: predict duration bin for each key
            self.duration_head = nn.Linear(transformer_dim, num_keys * num_duration_bins)
        else:
            # Regression: predict continuous duration (log or linear)
            self.duration_head = nn.Linear(transformer_dim, num_keys)
        
        # Optional: frame head for consistency loss (predicting active notes)
        self.frame_head = nn.Linear(transformer_dim, num_keys)
        
    def forward(self, x):
        """
        Args:
            x: Input spectrogram (batch, 1, freq, time)
            
        Returns:
            Dictionary with 'onset', 'duration', 'frame' predictions
            - onset: (batch, time, num_keys) - logits
            - duration: (batch, time, num_keys, num_bins) for classification 
                       or (batch, time, num_keys) for regression - logits/values
            - frame: (batch, time, num_keys) - logits
        """
        # Extract features with CNN
        x = self.feature_extractor(x)  # (batch_size, time_frames, transformer_dim)
        
        # Add positional encoding
        x = self.pos_encoder(x)  # (batch_size, time_frames, transformer_dim)
        
        # Transformer encoding
        x = self.transformer(x)  # (batch_size, time_frames, transformer_dim)
        
        # Apply onset head
        onset_logits = self.onset_head(x)  # (batch, time, num_keys)
        
        # Apply duration head
        duration_output = self.duration_head(x)
        if self.duration_mode == 'bins':
            # Reshape to (batch, time, num_keys, num_bins) for classification
            batch_size, time_frames, _ = x.shape
            duration_logits = duration_output.view(batch_size, time_frames, -1, self.num_duration_bins)
        else:
            # (batch, time, num_keys) for regression
            duration_logits = duration_output
        
        # Apply frame head for consistency
        frame_logits = self.frame_head(x)  # (batch, time, num_keys)
        
        return {
            'onset': onset_logits,
            'duration': duration_logits,
            'frame': frame_logits
        }


class PianoTranscriptionModelCNNOnly(nn.Module):
    """
    CNN-only model (no Transformer) for ablation study.
    Frame-by-frame prediction without temporal context.
    """
    def __init__(self, n_mels=128, hidden_size=256, num_outputs=91, dropout=0.1):
        super(PianoTranscriptionModelCNNOnly, self).__init__()
        
        # Convolutional feature extractor
        self.feature_extractor = ConvolutionalFeatureExtractor(n_mels, hidden_size)
        
        # Direct output projection (no transformer)
        self.output_projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_outputs)
        )
        
    def forward(self, x):
        """
        Forward pass - frame-by-frame prediction.
        
        Args:
            x: Input spectrogram (batch_size, 1, n_mels, time_frames)
            
        Returns:
            Output piano roll predictions (batch_size, time_frames, num_outputs)
        """
        # Extract features with CNN
        x = self.feature_extractor(x)  # (batch_size, time_frames, hidden_size)
        
        # Direct projection to outputs (no temporal modeling)
        x = self.output_projection(x)  # (batch_size, time_frames, num_outputs)
        
        # Apply sigmoid for binary predictions
        x = torch.sigmoid(x)
        
        return x


if __name__ == "__main__":
    from config import CONFIG, NUM_OUTPUTS
    
    # Test onset + duration model
    print("="*80)
    print("ONSET + DURATION MODEL (CNN + Transformer)")
    print("="*80)
    print(f"Duration mode: {CONFIG['duration_mode']}")
    
    model = PianoTranscriptionModel(
        input_features=CONFIG['n_mels'],
        num_keys=CONFIG['num_keys'],
        cnn_channels=[32, 64, 128],
        transformer_dim=CONFIG['hidden_size'],
        num_heads=CONFIG['num_heads'],
        num_layers=CONFIG['num_layers'],
        dropout=CONFIG['dropout'],
        duration_mode=CONFIG['duration_mode'],
        num_duration_bins=CONFIG['num_duration_bins']
    )
    
    # Create dummy input
    batch_size = 4
    time_frames = CONFIG['snippet_frames']
    dummy_input = torch.randn(batch_size, 1, CONFIG['n_mels'], time_frames)
    
    # Forward pass
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shapes:")
    print(f"  - onset: {output['onset'].shape}")
    print(f"  - duration: {output['duration'].shape}")
    print(f"  - frame: {output['frame'].shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with different duration modes
    print("\n" + "="*80)
    print("TESTING DIFFERENT DURATION MODES")
    print("="*80)
    
    for mode in ['bins', 'log', 'linear']:
        model_test = PianoTranscriptionModel(
            input_features=CONFIG['n_mels'],
            num_keys=CONFIG['num_keys'],
            transformer_dim=CONFIG['hidden_size'],
            num_heads=CONFIG['num_heads'],
            num_layers=CONFIG['num_layers'],
            dropout=CONFIG['dropout'],
            duration_mode=mode,
            num_duration_bins=CONFIG['num_duration_bins']
        )
        out = model_test(dummy_input)
        print(f"{mode:10s} mode - duration shape: {out['duration'].shape}")
    
    print("\n" + "="*80)
    print("CNN-ONLY MODEL (Ablation Study)")
    print("="*80)
    model_cnn = PianoTranscriptionModelCNNOnly(
        n_mels=CONFIG['n_mels'],
        hidden_size=CONFIG['hidden_size'],
        num_outputs=NUM_OUTPUTS,
        dropout=CONFIG['dropout']
    )
    
    output_cnn = model_cnn(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output_cnn.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model_cnn.parameters()):,}")
    
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    full_params = sum(p.numel() for p in model.parameters())
    cnn_params = sum(p.numel() for p in model_cnn.parameters())
    print(f"Onset+Duration model: {full_params:,} parameters")
    print(f"CNN-only: {cnn_params:,} parameters")
    print(f"Transformer overhead: {full_params - cnn_params:,} parameters ({(full_params - cnn_params) / full_params * 100:.1f}%)")
    print("="*80)
