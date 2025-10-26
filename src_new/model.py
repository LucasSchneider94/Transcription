import torch
import torch.nn as nn
import math

class ConvolutionalFeatureExtractor(nn.Module):
    """
    Convolutional layers to extract local features from spectrograms.
    """
    def __init__(self, n_mels, hidden_size):
        super(ConvolutionalFeatureExtractor, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # Reduce frequency dimension only
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # Reduce frequency dimension only
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # Reduce frequency dimension only
        )
        
        # Calculate output feature size after conv layers
        self.feature_size = 128 * (n_mels // 8)  # After 3 pooling layers with stride 2
        self.projection = nn.Linear(self.feature_size, hidden_size)
        
    def forward(self, x):
        # x shape: (batch_size, 1, n_mels, time_frames)
        x = self.conv_layers(x)  # (batch_size, 128, n_mels//8, time_frames)
        
        # Reshape: (batch_size, time_frames, 128 * n_mels//8)
        batch_size, channels, freq, time = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()  # (batch_size, time_frames, channels, freq)
        x = x.view(batch_size, time, -1)  # (batch_size, time_frames, channels * freq)
        
        # Project to hidden_size
        x = self.projection(x)  # (batch_size, time_frames, hidden_size)
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


class PianoTranscriptionModel(nn.Module):
    """
    Hybrid model: CNN for local feature extraction + Transformer for temporal modeling.
    """
    def __init__(self, n_mels=128, hidden_size=256, num_heads=8, num_layers=4, 
                 num_outputs=91, dropout=0.1):
        super(PianoTranscriptionModel, self).__init__()
        
        # Convolutional feature extractor
        self.feature_extractor = ConvolutionalFeatureExtractor(n_mels, hidden_size)
        
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
        """
        Forward pass.
        
        Args:
            x: Input spectrogram (batch_size, 1, n_mels, time_frames)
            
        Returns:
            Output piano roll predictions (batch_size, time_frames, num_outputs)
        """
        # Extract features with CNN
        x = self.feature_extractor(x)  # (batch_size, time_frames, hidden_size)
        
        # Add positional encoding
        x = self.pos_encoder(x)  # (batch_size, time_frames, hidden_size)
        
        # Transformer encoding
        x = self.transformer(x)  # (batch_size, time_frames, hidden_size)
        
        # Output projection
        x = self.output_projection(x)  # (batch_size, time_frames, num_outputs)
        
        # Apply sigmoid for binary predictions
        x = torch.sigmoid(x)
        
        return x


if __name__ == "__main__":
    from config import CONFIG, NUM_OUTPUTS
    
    # Test the model
    model = PianoTranscriptionModel(
        n_mels=CONFIG['n_mels'],
        hidden_size=CONFIG['hidden_size'],
        num_heads=CONFIG['num_heads'],
        num_layers=CONFIG['num_layers'],
        num_outputs=NUM_OUTPUTS,
        dropout=CONFIG['dropout']
    )
    
    # Create dummy input
    batch_size = 4
    time_frames = CONFIG['snippet_frames']
    dummy_input = torch.randn(batch_size, 1, CONFIG['n_mels'], time_frames)
    
    # Forward pass
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
