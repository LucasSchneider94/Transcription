"""Piano Transcription Model: Asymmetric Separable CNN + Temporal Transformer

Architecture:
  CNNEncoder          — Three ConvBlocks with elongated frequency-axis kernels that
                        span harmonic/overtone structure across many mel bins, paired
                        with moderate time-axis kernels for local dynamics.  Average-
                        pooling between blocks progressively reduces the frequency
                        dimension.  For n_mels=352 (=88×4) the default pool schedule
                        (×4, ×2, ×4) yields a flat dim of 128 × 11 = 1 408 which is
                        projected to transformer_dim.
  TemporalTransformer — Pre-norm Transformer encoder over the time axis.  Self-
                        attention gives every frame access to every other frame,
                        enabling global temporal context at modest cost (sequences
                        are only a few hundred frames long).
  Two linear heads    — onset (sparse, fired at note-attack) and frame (dense,
                        active throughout acoustic duration).  Both output raw
                        logits over 88 piano keys × T time frames.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CONFIG



# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Separable frequency-then-time convolution block with residual connection.

    Applies a tall frequency-axis kernel first — to capture harmonic/overtone
    relationships that span many mel bins — then a wide time-axis kernel for
    local temporal dynamics.  A 1×1 shortcut allows gradients to flow freely
    and lets the block learn residual corrections rather than full mappings.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        freq_kernel: int,
        time_kernel: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert freq_kernel % 2 == 1, "freq_kernel must be odd (same-padding)"
        assert time_kernel % 2 == 1, "time_kernel must be odd (same-padding)"

        self.freq_conv = nn.Conv2d(
            in_channels, out_channels, (freq_kernel, 1),
            padding=(freq_kernel // 2, 0), bias=False,
        )
        self.freq_norm = nn.GroupNorm(min(8, out_channels), out_channels)

        self.time_conv = nn.Conv2d(
            out_channels, out_channels, (1, time_kernel),
            padding=(0, time_kernel // 2), bias=False,
        )
        self.time_norm = nn.GroupNorm(min(8, out_channels), out_channels)

        self.residual_proj = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.drop = nn.Dropout2d(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.residual_proj(x)
        x = F.gelu(self.freq_norm(self.freq_conv(x)))
        x = self.drop(F.gelu(self.time_norm(self.time_conv(x))))
        return x + identity


class CNNEncoder(nn.Module):
    """Multi-stage separable CNN.

    Input:  (B, 1, F, T)   — F = n_mels (default 352 = 88 × 4)
    Output: (B, T, D)

    Each stage applies a ConvBlock (freq then time conv) then averages over a
    small frequency window.  The default schedule (pool ×4, ×2, ×4) brings
    352 freq bins down to 11, giving a flat dim of 128 × 11 = 1 408 before the
    linear projection to transformer_dim.

    The frequency kernels grow smaller at each stage (87 → 31 → 15) because
    the effective receptive field grows through the pooling chain.
    """

    def __init__(
        self,
        n_mels: int = 352,
        channels: tuple = (32, 64, 128),
        freq_kernels: tuple = (87, 31, 15),
        time_kernel: int = 9,
        freq_pool: tuple = (4, 2, 4),
        transformer_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert len(channels) == len(freq_kernels) == len(freq_pool)
        total_pool = 1
        for p in freq_pool:
            total_pool *= p
        assert n_mels % total_pool == 0, (
            f"n_mels={n_mels} must be divisible by cumulative freq-pool factor {total_pool}"
        )

        in_c = 1
        self.blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        for out_c, fk, pool in zip(channels, freq_kernels, freq_pool):
            self.blocks.append(ConvBlock(in_c, out_c, fk, time_kernel, dropout))
            self.pools.append(nn.AvgPool2d((pool, 1)))
            in_c = out_c

        flat_dim = channels[-1] * (n_mels // total_pool)
        self.project = nn.Sequential(
            nn.Linear(flat_dim, transformer_dim, bias=False),
            nn.LayerNorm(transformer_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block, pool in zip(self.blocks, self.pools):
            x = pool(block(x))                              # (B, Ci, Fi, T)
        B, C, F_out, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * F_out)  # (B, T, C×F')
        return self.project(x)                              # (B, T, D)


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 8000, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(x + self.pe[:, : x.size(1)])


class TemporalTransformer(nn.Module):
    """Pre-norm Transformer encoder over the time axis.

    Input:  (B, T, D)
    Output: (B, T, D)

    norm_first=True (pre-norm) is used throughout — it gives more stable
    gradients and generally converges faster than post-norm at moderate scale.
    """

    def __init__(self, d_model: int, num_heads: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.pos_enc(x))


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class PianoTranscriptionModel(nn.Module):
    """Separable CNN + Temporal Transformer for piano transcription.

    Predicts per-key logits for:
      onset — sparse, fired at note-attack frames
      frame — dense, active throughout acoustic note duration (incl. pedal)
    """

    def __init__(
        self,
        n_mels: int = CONFIG["n_mels"],
        num_keys: int = CONFIG["num_keys"],
        cnn_channels: tuple = (32, 64, 128),
        cnn_freq_kernels: tuple = (87, 31, 15),
        cnn_time_kernel: int = 9,
        cnn_freq_pool: tuple = (4, 2, 4),
        transformer_dim: int = 256,
        transformer_heads: int = 8,
        transformer_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cnn = CNNEncoder(
            n_mels=n_mels,
            channels=tuple(cnn_channels),
            freq_kernels=tuple(cnn_freq_kernels),
            time_kernel=cnn_time_kernel,
            freq_pool=tuple(cnn_freq_pool),
            transformer_dim=transformer_dim,
            dropout=dropout,
        )
        self.transformer = TemporalTransformer(
            d_model=transformer_dim,
            num_heads=transformer_heads,
            num_layers=transformer_layers,
            dropout=dropout,
        )
        self.onset_head = nn.Linear(transformer_dim, num_keys)
        self.frame_head = nn.Linear(transformer_dim, num_keys)
        self._init_heads()

    def _init_heads(self):
        """Initialise head biases to reflect the sparsity of each label type.

        Onsets are very sparse (~2% of frames have an onset at any key).
        Frame activations are denser (~12%).  Starting the sigmoid close to
        the true base rate avoids large initial focal-loss gradients.
        """
        nn.init.constant_(self.onset_head.bias, -4.0)   # sigmoid ≈ 0.018
        nn.init.constant_(self.frame_head.bias, -2.0)   # sigmoid ≈ 0.119

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: (B, 1, F, T)  log-mel spectrogram
        Returns:
            dict with:
              "onset": (B, T, num_keys)  raw logits
              "frame": (B, T, num_keys)  raw logits
        """
        f = self.cnn(x)          # (B, T, D)
        f = self.transformer(f)  # (B, T, D)
        return {
            "onset": self.onset_head(f),  # (B, T, 88)
            "frame": self.frame_head(f),  # (B, T, 88)
        }


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = PianoTranscriptionModel()
    n = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n:,}")
    x = torch.randn(2, 1, CONFIG["n_mels"], 300)
    out = model(x)
    print(f"Input: {tuple(x.shape)}")
    print(f"Onset: {tuple(out['onset'].shape)}")
    print(f"Frame: {tuple(out['frame'].shape)}")
