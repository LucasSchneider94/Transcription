import torch.nn as nn
import torch.nn.functional as F

class ChordPredictionCNN(nn.Module):
    def __init__(self, kernel_size=3, depth=64, stride=2, n_notes=88, num_conv_layers=3):
        super(ChordPredictionCNN, self).__init__()
        self.conv_layers = nn.ModuleList()
        in_channels = 2  # Stereo input

        # Downsampling and feature extraction layers
        for i in range(num_conv_layers):
            out_channels = depth * (2 ** i)
            self.conv_layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
            )
            self.conv_layers.append(nn.BatchNorm1d(out_channels))
            self.conv_layers.append(nn.ReLU())
            in_channels = out_channels

        # Prediction layer
        self.prediction_layer = nn.Conv1d(in_channels, n_notes, kernel_size=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, 2, n_time_steps)
        for layer in self.conv_layers:
            x = layer(x)
        x = self.prediction_layer(x)
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, downsampled_seq_len, n_notes)
        return x
