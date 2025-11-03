import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=1, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class MultiScaleResidualCNN(nn.Module):
    def __init__(self,
                 n_notes=88,
                 in_channels=2,
                 base_channels=64,
                 num_blocks=3,
                 block_kernel_sizes=None,
                 block_strides=None,
                 block_dilations=None,
                 multiscale_kernel_sizes=None,
                 use_multiscale=True):
        super().__init__()

        # Defaults
        if block_kernel_sizes is None:
            block_kernel_sizes = [3] * num_blocks
        if block_strides is None:
            block_strides = [1] * num_blocks
        if block_dilations is None:
            block_dilations = [1] * num_blocks
        if multiscale_kernel_sizes is None:
            multiscale_kernel_sizes = [3, 5, 7]

        # Initial projection
        self.initial_conv = nn.Conv1d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)

        # Residual blocks
        channels = base_channels
        blocks = []
        for i in range(num_blocks):
            next_channels = channels * (2 if block_strides[i] > 1 else 1)
            blocks.append(
                ResidualBlock(
                    in_channels=channels,
                    out_channels=next_channels,
                    kernel_size=block_kernel_sizes[i],
                    stride=block_strides[i],
                    dilation=block_dilations[i],
                )
            )
            channels = next_channels
        self.res_blocks = nn.ModuleList(blocks)

        # Multi-scale branches
        self.use_multiscale = use_multiscale
        if use_multiscale:
            self.branches = nn.ModuleList([
                nn.Conv1d(channels, channels, kernel_size=k, padding=k // 2)
                for k in multiscale_kernel_sizes
            ])
            self.combine_conv = nn.Conv1d(channels * len(multiscale_kernel_sizes), channels, kernel_size=1)

        # Prediction head
        self.prediction_layer = nn.Conv1d(channels, n_notes, kernel_size=1)

        # Losses
        self.losses=[]
        self.eval_losses=[]
        self.learning_rate=[]
    
        self.F1=[]
        self.R=[]
        self.P=[]
        self.F1EVAL=[]
        self.REVAL=[]
        self.PEVAL=[]
    
    def addLosses(self, val):
        self.losses.append(val)

    def addEvalLosses(self, val):
        self.eval_losses.append(val)

    def addLR(self, val):
        self.learning_rate.append(val)

    def addF1(self, val):
        self.F1.append(val)

    def addR(self, val):
        self.R.append(val)

    def addP(self, val):
        self.P.append(val)

    def addF1EVAL(self, val):
        self.F1EVAL.append(val)

    def addREVAL(self, val):
        self.REVAL.append(val)

    def addPEVAL(self, val):
        self.PEVAL.append(val)

    def forward(self, x):
        # Input shape: (batch, time, channels)
        # x = x.permute(0, 2, 1)  # -> (batch, channels, time)

        x = self.initial_conv(x)

        for block in self.res_blocks:
            x = block(x)

        if self.use_multiscale:
            multiscale_outs = [branch(x) for branch in self.branches]
            x = torch.cat(multiscale_outs, dim=1)
            x = F.relu(self.combine_conv(x))

        x = self.prediction_layer(x)

        return x.permute(0, 2, 1)  # -> (batch, time, n_notes)
