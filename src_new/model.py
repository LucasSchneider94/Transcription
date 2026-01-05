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


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            
            # Handle both 2D (B, C, H, W) and 1D (B, C, T) spatial/temporal dims
            if x.ndim == 4:
                x = self.weight[:, None, None] * x + self.bias[:, None, None]
            elif x.ndim == 3:
                x = self.weight[:, None] * x + self.bias[:, None]
            return x


class ConvNeXtBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we usually have PyTorch layers that expect channels_first, but Linear/LN are faster on channels_last.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = nn.Identity() # Placeholder, usually DropPath is used here

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 1) # (N, C, L) -> (N, L, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 2, 1) # (N, L, C) -> (N, C, L)

        x = input + self.drop_path(x)
        return x


class ConvolutionalFeatureExtractor(nn.Module):
    """
    Convolutional layers to extract local features from spectrograms.
    Uses ConvNeXt blocks (modern best practice) instead of standard CNN.
    """
    def __init__(self, n_mels, hidden_size):
        super(ConvolutionalFeatureExtractor, self).__init__()
        
        # Config for ConvNeXt-like structure
        # We want to maintain the (Freq/16, Time/1) downsampling of the original
        dims = [64, 128, 256, 256]
        
        # Stem: Standard Conv to get into feature space
        self.stem = nn.Sequential(
            nn.Conv1d(n_mels, dims[0], kernel_size=3, padding=1),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        
        self.stages = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        
        # 4 stages to match original depth
        # Stage 0
        self.stages.append(ConvNeXtBlock(dims[0]))
        # Downsample 0: 64 -> 128, Time/1
        self.downsample_layers.append(nn.Sequential(
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
            nn.Conv1d(dims[0], dims[1], kernel_size=2, stride=2)
        ))
        
        # Stage 1
        self.stages.append(ConvNeXtBlock(dims[1]))
        # Downsample 1: 128 -> 256, Time/1
        self.downsample_layers.append(nn.Sequential(
            LayerNorm(dims[1], eps=1e-6, data_format="channels_first"),
            nn.Conv1d(dims[1], dims[2], kernel_size=2, stride=2)
        ))
        
        # Stage 2
        self.stages.append(ConvNeXtBlock(dims[2]))
        # Downsample 2: 256 -> 256, Time/1
        self.downsample_layers.append(nn.Sequential(
            LayerNorm(dims[2], eps=1e-6, data_format="channels_first"),
            nn.Conv1d(dims[2], dims[3], kernel_size=2, stride=2)
        ))
        
        # Stage 3
        self.stages.append(ConvNeXtBlock(dims[3]))
        # Downsample 3: 256 -> 256, Time/1
        self.downsample_layers.append(nn.Sequential(
            LayerNorm(dims[3], eps=1e-6, data_format="channels_first"),
            nn.Conv1d(dims[3], dims[3], kernel_size=2, stride=2)
        ))
        
        # Calculate output feature size
        self.feature_size = dims[3]
        self.projection = nn.Linear(self.feature_size, hidden_size)
        
    def forward(self, x):
        x = self.stem(x)
        
        for i in range(4):
            x = self.stages[i](x)
            x = self.downsample_layers[i](x)
            
        # Reshape: (batch_size, channels, time)
        # We want (batch_size, time, channels) for projection
        x = x.permute(0, 2, 1) 
        
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
        
        # Initialize onset head bias to encourage onset detection from the start
        # Without this, model predicts ~0 everywhere initially due to extreme sparsity (0.23% positive)
        # Initialize bias such that sigmoid(bias) ≈ 0.01 (i.e., bias ≈ -4.6)
        # This gives the model a "head start" on detecting onsets
        nn.init.constant_(self.onset_head.bias, -4.6)  # sigmoid(-4.6) ≈ 0.01
        
    def forward(self, x):
        """
        Args:
            x: Input spectrogram (batch, freq, time)
            
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
    def __init__(self, n_mels=128, hidden_size=256, num_keys=88, dropout=0.1, duration_mode='bins', num_duration_bins=8):
        super(PianoTranscriptionModelCNNOnly, self).__init__()
        
        self.duration_mode = duration_mode
        self.num_duration_bins = num_duration_bins
        
        # Convolutional feature extractor
        self.feature_extractor = ConvolutionalFeatureExtractor(n_mels, hidden_size)
        
        # Direct output projection (no transformer)
        self.projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Heads
        self.onset_head = nn.Linear(hidden_size, num_keys)
        self.frame_head = nn.Linear(hidden_size, num_keys)
        
        if duration_mode == 'bins':
            self.duration_head = nn.Linear(hidden_size, num_keys * num_duration_bins)
        else:
            self.duration_head = nn.Linear(hidden_size, num_keys)
            
        nn.init.constant_(self.onset_head.bias, -4.6)
        
    def forward(self, x):
        """
        Forward pass - frame-by-frame prediction.
        
        Args:
            x: Input spectrogram (batch_size, n_mels, time_frames)
            
        Returns:
            Dictionary with 'onset', 'duration', 'frame' predictions
        """
        # Extract features with CNN
        x = self.feature_extractor(x)  # (batch_size, time_frames, hidden_size)
        
        # Projection
        x = self.projection(x)
        
        # Heads
        onset = self.onset_head(x)
        frame = self.frame_head(x)
        
        duration = self.duration_head(x)
        if self.duration_mode == 'bins':
            batch_size, time_frames, _ = x.shape
            duration = duration.view(batch_size, time_frames, -1, self.num_duration_bins)
        
        return {
            'onset': onset,
            'duration': duration,
            'frame': frame
        }


class UNetEncoder(nn.Module):
    """
    Encoder path with time+frequency downsampling.
    Returns features at each layer for skip connections.
    """
    def __init__(self, input_channels, encoder_channels, downsample_factor, num_layers):
        super(UNetEncoder, self).__init__()
        self.layers = nn.ModuleList()
        self.downsample_factor = downsample_factor
        
        in_c = input_channels
        for out_c in encoder_channels:
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(downsample_factor, downsample_factor), 
                           stride=(downsample_factor, downsample_factor))
            ))
            in_c = out_c
            
    def forward(self, x):
        # Returns: (encoder_features_list, bottleneck_features)
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return features[:-1], features[-1]


class UNetDecoder(nn.Module):
    """
    Decoder path with transposed convolutions for upsampling.
    Uses skip connections from encoder.
    """
    def __init__(self, decoder_channels, upsample_factor, num_layers, encoder_channels):
        super(UNetDecoder, self).__init__()
        self.layers = nn.ModuleList()
        self.upsample_factor = upsample_factor
        
        in_c = encoder_channels[-1]  # Start with bottleneck channels
        skip_channels = encoder_channels[:-1][::-1]  # Reverse encoder channels for skip connections
        
        for i, out_c in enumerate(decoder_channels):
            skip_c = skip_channels[i] if i < len(skip_channels) else 0
            self.layers.append(nn.ModuleDict({
                'up': nn.ConvTranspose2d(in_c, out_c, kernel_size=upsample_factor, stride=upsample_factor),
                'conv': nn.Sequential(
                    nn.Conv2d(out_c + skip_c, out_c, kernel_size=3, padding=1),
                    nn.ReLU()
                )
            }))
            in_c = out_c

    def forward(self, x, skip_connections):
        skips = skip_connections[::-1]
        for i, layer in enumerate(self.layers):
            x = layer['up'](x)
            if i < len(skips):
                skip = skips[i]
                if x.shape != skip.shape:
                    x = F.interpolate(x, size=skip.shape[2:], mode=bilinear, align_corners=False)
                x = torch.cat([x, skip], dim=1)
            x = layer['conv'](x)
        return x


class TransformerBottleneck(nn.Module):
    """
    Transformer at bottleneck (sees T/64 frames).
    Operates on flattened features.
    """
    def __init__(self, in_channels, transformer_dim, num_heads, num_layers, ff_dim, dropout=0.1):
        super(TransformerBottleneck, self).__init__()
        
        self.input_dim = in_channels
        self.transformer_dim = transformer_dim
        
        # Projections
        self.input_proj = nn.Linear(self.input_dim, transformer_dim)
        self.output_proj = nn.Linear(transformer_dim, self.input_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(transformer_dim, dropout=dropout)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x, print_shapes=False):
        # Input: (B, C, T)
        B, C, T = x.shape
        
        if print_shapes:
            print(f"    Bottleneck Input: {x.shape}")
        
        # Reshape to (B, T, C)
        x = x.permute(0, 2, 1)
        
        if print_shapes:
            print(f"    Bottleneck Reshaped (B, T, C): {x.shape}")
        
        # Project and Transformer
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        
        if print_shapes:
            print(f"    Transformer Input (Proj+Pos): {x.shape}")
        
        x = self.transformer(x)
            
        x = self.output_proj(x)
        
        # Reshape back to (B, C, T)
        x = x.permute(0, 2, 1)
        
        if print_shapes:
            print(f"    Bottleneck Output: {x.shape}")
        
        return x


class PianoTranscriptionModelUNet(nn.Module):
    """
    1D U-Net architecture for piano transcription.
    Treats frequency bins as input channels.
    """
    def __init__(self, config):
        super(PianoTranscriptionModelUNet, self).__init__()
        
        self.duration_mode = config['duration_mode']
        self.num_duration_bins = config.get('num_duration_bins', 8)
        
        # Get downsample factor from config
        time_downsample = config.get('unet_downsample_factor', 2)
        
        # Encoder
        self.encoder = nn.ModuleList()
        in_c = config['n_mels'] # Input channels = Frequency bins
        
        for out_c in config['unet_encoder_channels']:
            self.encoder.append(nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(),
                # Time downsample only
                nn.MaxPool1d(kernel_size=time_downsample, stride=time_downsample)
            ))
            in_c = out_c
            
        # Bottleneck dims
        bottleneck_channels = config['unet_encoder_channels'][-1]
        
        self.transformer = TransformerBottleneck(
            in_channels=bottleneck_channels,
            transformer_dim=config['transformer_dim'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            ff_dim=config['transformer_ff_dim'],
            dropout=config['dropout']
        )
        
        # Decoder
        self.decoder = nn.ModuleList()
        skip_channels = config['unet_encoder_channels'][:-1][::-1]
        decoder_channels = config['unet_decoder_channels']
        
        in_c = bottleneck_channels
        
        for i, out_c in enumerate(decoder_channels):
            skip_c = skip_channels[i] if i < len(skip_channels) else config['n_mels'] # Last skip is input
            
            self.decoder.append(nn.ModuleDict({
                # Upsample time
                'up': nn.ConvTranspose1d(in_c, out_c, kernel_size=time_downsample, stride=time_downsample),
                'conv': nn.Sequential(
                    nn.Conv1d(out_c + skip_c, out_c, kernel_size=3, padding=1),
                    nn.ReLU()
                )
            }))
            in_c = out_c
            
        # Final projection to 88 keys + heads
        # We use a shared projection layer to get to a hidden representation before heads
        self.head_projection = nn.Sequential(
            nn.Conv1d(decoder_channels[-1], 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.onset_head = nn.Linear(64, 88)
        self.frame_head = nn.Linear(64, 88)
        
        if self.duration_mode == 'bins':
            self.duration_head = nn.Linear(64, 88 * self.num_duration_bins)
        else:
            self.duration_head = nn.Linear(64, 88)
            
        nn.init.constant_(self.onset_head.bias, -4.6)

    def forward(self, x, print_shapes=False):
        # Input: (B, n_mels, T)
        if print_shapes:
            print(f"Input: {x.shape}")

        # Encoder
        skips = []
        # Save input as the last skip connection (for the last decoder layer)
        skips.append(x)
        
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            skips.append(x)
            if print_shapes:
                print(f"Encoder Layer {i}: {x.shape}")
            
        bottleneck = skips.pop() # This is the output of the last encoder layer
        
        # Transformer
        if print_shapes:
            print("--- Transformer Bottleneck ---")
        bottleneck = self.transformer(bottleneck, print_shapes=print_shapes)
        if print_shapes:
            print("----------------------------")
        
        # Decoder
        x = bottleneck
        # skips now contains [input, enc1, enc2, ...]
        # We want to pop from the end: enc2, enc1, input
        
        for i, layer in enumerate(self.decoder):
            x = layer['up'](x)
            
            skip = skips.pop()
            
            # Handle potential size mismatch due to odd dimensions
            if x.shape[2] != skip.shape[2]:
                x = F.interpolate(x, size=skip.shape[2:], mode='linear', align_corners=False)
            
            x = torch.cat([x, skip], dim=1)
            x = layer['conv'](x)
            if print_shapes:
                print(f"Decoder Layer {i}: {x.shape}")
            
        # Heads
        x = self.head_projection(x) # (B, 64, T)
        
        # Permute for Linear layers: (B, T, 64)
        x = x.permute(0, 2, 1)
        
        onset = self.onset_head(x)
        frame = self.frame_head(x)
        
        duration = self.duration_head(x)
        if self.duration_mode == 'bins':
            batch_size, time_frames, _ = x.shape
            duration = duration.view(batch_size, time_frames, 88, self.num_duration_bins)
            
        if print_shapes:
            print(f"Output Onset: {onset.shape}")
            print(f"Output Frame: {frame.shape}")
            print(f"Output Duration: {duration.shape}")

        return {
            'onset': onset,
            'duration': duration,
            'frame': frame
        }


if __name__ == "__main__":
    from config import CONFIG, NUM_OUTPUTS
    
    if CONFIG.get('use_unet', False):
        print("="*80)
        print("TESTING 1D U-NET ARCHITECTURE")
        print("="*80)
        
        model = PianoTranscriptionModelUNet(CONFIG)
        
        # Create dummy input
        batch_size = 2
        # Input is (Batch, n_mels, Time)
        dummy_input = torch.randn(batch_size, CONFIG['n_mels'], CONFIG['snippet_bins'])
        
        print(f"Input shape: {dummy_input.shape}")
        
        # Forward pass
        output = model(dummy_input, print_shapes=True)
        
        print(f"Output shapes:")
        print(f"  - onset: {output['onset'].shape}")
        print(f"  - duration: {output['duration'].shape}")
        print(f"  - frame: {output['frame'].shape}")
        
        # Validate shapes
        assert output['onset'].shape == (batch_size, CONFIG['snippet_bins'], 88)
        assert output['frame'].shape == (batch_size, CONFIG['snippet_bins'], 88)
        print(f"✓ Output shapes correct")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"TOTAL PARAMS: {total_params:,}")
        print("="*80)
        exit()

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
        num_keys=CONFIG['num_keys'],
        dropout=CONFIG['dropout'],
        duration_mode=CONFIG['duration_mode'],
        num_duration_bins=CONFIG['num_duration_bins']
    )
    
    output_cnn = model_cnn(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shapes:")
    print(f"  - onset: {output_cnn['onset'].shape}")
    print(f"  - duration: {output_cnn['duration'].shape}")
    print(f"  - frame: {output_cnn['frame'].shape}")
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
