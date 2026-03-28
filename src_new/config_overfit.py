CONFIG_OVERFIT = {
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
    'normalization_mode': 'global',

    # Overfit setup
    'snippet_duration': 1.0,
    'snippets_per_file': 4,
    'data_fraction': 0.01,
    'fixed_snippets': True,
    'sampling_mode': 'onset_aware',
    'onset_sampling_min_active_ratio': 0.8,

    # Optimization (aggressive to force memorization)
    'batch_size': 1,
    'learning_rate': 1e-4,
    'num_epochs': 500,
    'weight_decay': 0.0,
    'grad_clip_norm': 1.0,
    'deterministic_seed': 42,
    'use_amp': False,

    # Architecture (smaller CNN + Transformer for fast overfit testing)
    'cnn_channels':     [16, 32, 64],
    'cnn_freq_kernels': [87, 31, 15],
    'cnn_time_kernel':  9,
    'cnn_freq_pool':    [4, 2, 4],      # same pool schedule as full model
    'transformer_dim':     128,
    'transformer_heads':   4,
    'transformer_layers':  2,
    'dropout': 0.0,

    # Losses (onset + frame only)
    'onset_weight': 3.0,
    'frame_weight': 1.0,
    'onset_loss_type': 'bce',           # plain BCE with pos_weight to force memorisation
    'frame_loss_type': 'bce',
    'use_computed_pos_weight': True,
    'max_pos_weight': 1000.0,
    'onset_focal_alpha': 0.9,
    'onset_focal_gamma': 2.0,
    'frame_focal_alpha': 0.5,
    'frame_focal_gamma': 2.0,

    # Threshold sweep & metric tolerance
    'threshold_sweep_values': [0.2, 0.3, 0.4, 0.5],
    'onset_tolerance_frames': 2,
    'max_collect_batches': 4,           # tiny — only a handful of overfit batches
    'default_inference_onset_threshold': 0.3,
    'default_inference_frame_threshold': 0.4,
    # Gating off: if onset head is suppressed, gating will black out frame too.
    # Keep off until both heads are confirmed firing.
    'inference_apply_onset_frame_gating': False,
    'inference_temporal_median_kernel': 1,
    'inference_min_active_frames': 1,

    # Runtime
    'visualize_every_n_epochs': 5,
    'enable_visualization': True,

    # Scheduler disabled for overfit
    'use_lr_scheduler': False,
    'scheduler_type': 'cosine_warmup',
    'warmup_epochs': 0,
    'scheduler_min_lr': 1e-6,

    # DataLoader
    'num_workers': 0,
    'pin_memory': False,
    'prefetch_factor': 2,
    'persistent_workers': False,
    'preload_into_ram': False,

    # Overfit mode controls
    'overfit_mode': True,
    'overfit_num_files': 2,
    'shuffle_train': False,
}

# Derived constants
CONFIG_OVERFIT['hop_length'] = int(CONFIG_OVERFIT['sample_rate'] / CONFIG_OVERFIT['roll_fps'])
CONFIG_OVERFIT['snippet_frames'] = int(CONFIG_OVERFIT['snippet_duration'] * CONFIG_OVERFIT['roll_fps'])

MIN_PITCH = 21
MAX_PITCH = 108
NUM_PEDALS = 3
NUM_OUTPUTS = CONFIG_OVERFIT['num_keys'] + NUM_PEDALS  # 88 keys + 3 pedals = 91
