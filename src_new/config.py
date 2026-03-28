CONFIG = {
    'sample_rate': 48000,
    'roll_fps': 100,
    'num_keys': 88,
    'n_mels': 88 * 4,
    'n_fft': 4096,
    'plot_pngs': True,

    # Data paths
    'data_dir': './processed_data_17_18',
    'maestro_json': './maestro-v3.0.0.json',
    'split_year_folder': '2013',

    # Data/label semantics
    'onset_frames': 2,
    'duration_mode': 'log',
    'num_duration_bins': 8,
    'min_note_duration': 0.05,

    # Normalization
    'normalization_mode': 'global',         # one of: none, global

    # Training data slicing
    'snippet_duration': 3.0,
    'snippets_per_file': 10,   # × 444 files / B=16 → ~277 batches/epoch (~8-9h overnight)
    'data_fraction': 1.0,
    'fixed_snippets': False,

    # Onset-aware sampling
    'sampling_mode': 'onset_aware',         # one of: random, onset_aware
    'onset_sampling_min_active_ratio': 0.5,

    # Optimization
    'batch_size': 16,
    'learning_rate': 1e-4,
    'num_epochs': 200,
    'weight_decay': 1e-4,
    'grad_clip_norm': 1.0,
    'deterministic_seed': 42,
    'use_amp': True,

    # Model architecture (separable CNN + Transformer)
    # CNN stage: three ConvBlocks with alternating freq + time convolutions.
    # freq_kernels shrink at each stage because the effective receptive field
    # grows through the pooling chain; pool factors must divide n_mels (352).
    'cnn_channels':     [32, 64, 128],
    'cnn_freq_kernels': [87, 31, 15],   # odd, for same-padding
    'cnn_time_kernel':  9,
    'cnn_freq_pool':    [4, 2, 4],      # 352→88→44→11 (total ÷32)
    # Transformer stage
    'transformer_dim':     256,
    'transformer_heads':   8,
    'transformer_layers':  4,
    'dropout': 0.1,

    # Losses (onset + frame only; duration head removed)
    'onset_weight': 3.0,
    'frame_weight': 1.0,

    # Imbalance handling
    'onset_loss_type': 'focal',
    'frame_loss_type': 'focal',
    'use_computed_pos_weight': True,
    'max_pos_weight': 1000.0,
    'onset_focal_alpha': 0.9,
    'onset_focal_gamma': 2.0,
    'frame_focal_alpha': 0.5,
    'frame_focal_gamma': 2.0,

    # Threshold sweep & metric tolerance
    'threshold_sweep_values': [0.2, 0.3, 0.4, 0.5, 0.6],
    'onset_tolerance_frames': 2,           # ±20 ms at 100 fps — key must be exact
    'max_collect_batches': 64,             # batches to collect for metrics per epoch
    'default_inference_onset_threshold': 0.5,
    'default_inference_frame_threshold': 0.4,
    'inference_apply_onset_frame_gating': True,
    'inference_temporal_median_kernel': 3,
    'inference_min_active_frames': 1,

    # Visualization/runtime behavior
    'visualize_every_n_epochs': 10,
    'enable_visualization': True,

    # Learning rate scheduler settings
    'use_lr_scheduler': True,
    'scheduler_type': 'reduce_on_plateau',
    'scheduler_factor': 0.5,
    'scheduler_patience': 12,
    'warmup_epochs': 10,
    'scheduler_min_lr': 5e-7,

    # DataLoader settings
    'num_workers': 4,
    'pin_memory': False,
    'prefetch_factor': 2,
    'persistent_workers': True,
    'preload_into_ram': False,

    # Optional overfit switch for CLI
    'overfit_mode': False,
    'overfit_num_files': 2,

    # Training behavior
    'shuffle_train': True,
}

# Derived constants
CONFIG['hop_length'] = int(CONFIG['sample_rate'] / CONFIG['roll_fps'])  # Hop length for STFT
CONFIG['snippet_frames'] = int(CONFIG['snippet_duration'] * CONFIG['roll_fps'])  # Frames per snippet

MIN_PITCH = 21  # Lowest MIDI pitch for 88-key piano (A0)
MAX_PITCH = 108  # Highest MIDI pitch for 88-key piano (C8)
NUM_PEDALS = 3  # Sustain, soft, sostenuto
NUM_OUTPUTS = CONFIG['num_keys'] + NUM_PEDALS  # 88 keys + 3 pedals = 91