import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)


class AudioSegmentationModel(nn.Module):
    def __init__(self, 
                 num_notes: int,
                 cnn_kernel_size: int,
                 cnn_num_layers: int,
                 cnn_stride: int,
                 transformer_num_layers: int,
                 transformer_d_model: int,
                 transformer_nhead: int,
                 transformer_dim_feedforward: int = 2048,
                 transformer_dropout: float = 0.1):
        super().__init__()

        # CNN layers for downsampling
        cnn_layers = []
        in_channels = 2  # stereo input
        out_channels = transformer_d_model  # prepare for transformer input

        for _ in range(cnn_num_layers):
            cnn_layers.append(nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=cnn_kernel_size,
                stride=cnn_stride,
                padding=cnn_kernel_size // 2
            ))
            cnn_layers.append(nn.ReLU())
            in_channels = out_channels

        self.cnn = nn.Sequential(*cnn_layers)

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model=transformer_d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            dim_feedforward=transformer_dim_feedforward,
            dropout=transformer_dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_num_layers)

        # Final classification layer
        self.classifier = nn.Linear(transformer_d_model, num_notes)

    def forward(self, x):
        # x: (batch, time_steps, 2)
        x = x.transpose(1, 2)  # -> (batch, channels=2, time_steps)
        x = self.cnn(x)  # -> (batch, channels=transformer_d_model, time_steps_downsampled)
        x = x.transpose(1, 2)  # -> (batch, time_steps_downsampled, transformer_d_model)

        x = self.positional_encoding(x)  # add positional info
        x = self.transformer(x)  # -> (batch, time_steps_downsampled, transformer_d_model)
        logits = self.classifier(x)  # -> (batch, time_steps_downsampled, num_notes)

        return logits
