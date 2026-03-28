CONFIG_OVERFIT = {
    'sample_rate': 48000,
    'roll_fps': 100,
    'num_keys': 88,
    'n_mels': 88 * 4,                       # 4 bins per key (352 total)
    'n_fft': 4096,
    'plot_pngs': True,
    
    # Data paths
    'data_dir': './processed_data_17_18',
    'split_year_folder': "2013",
    'maestro_json': './maestro-v3.0.0.json',
    
    # Onset/Duration parameters
    'onset_frames': 2,
    'duration_mode': 'log',
    'num_duration_bins': 8,
    'min_note_duration': 0.05,
    
    # Training parameters - OPTIMIZED FOR OVERFITTING (SAME MODEL SIZE AS NORMAL)
    'snippet_duration': 3.0,
    'batch_size': 8,                        # OVERFIT: Small batch for better memorization
    'learning_rate': 5e-4,                  # OVERFIT: Higher LR to learn faster
    'num_epochs': 1000,                      # OVERFIT: More epochs to ensure full memorization
    'hidden_size': 256,                     # SAME as normal config
    'num_heads': 8,                         # SAME as normal config
    'num_layers': 4,                        # SAME as normal config
    'dropout': 0.0,                         # OVERFIT: No dropout - we WANT to overfit
    'weight_decay': 0.0,                    # OVERFIT: No regularization
    
    # Multi-task loss weights - REBALANCED for overfitting
    'onset_weight': 1.0,                   # OVERFIT: Focus heavily on onsets
    'duration_weight': 5.0,                 # OVERFIT: Need to learn durations better
    'frame_weight': 1.0,                    # OVERFIT: Low weight - don't let it dominate!
    'consistency_weight': 0.5,              # OVERFIT: Disable consistency loss
    
    # Gaussian onset smoothing (sigma annealing)
    'initial_sigma': 1.5,                   # Fixed Gaussian sigma (bins) - no annealing
    'final_sigma': 1.5,                     # Fixed Gaussian sigma (bins) - no annealing
    'sigma_anneal_threshold': 0.4,          # Start annealing when onset F1 > this threshold
    'sigma_anneal_epochs': 50,              # Duration of sigma annealing in epochs
    
    # Focal Loss parameters (CRITICAL for overfitting test)
    'use_focal_loss': True,                 # OVERFIT: Use Focal Loss to force onset detection
    'onset_focal_alpha': 0.9,               # OVERFIT: Very high alpha (onsets are 0.23% positive)
    'onset_focal_gamma': 2.0,               # OVERFIT: Standard gamma
    'frame_focal_alpha': 0.5,               # OVERFIT: Moderate alpha for frames
    'frame_focal_gamma': 2.0,               # OVERFIT: Standard gamma
    
    # Data settings - OVERFIT MODE
    'snippets_per_file': 5,                # OVERFIT: 10 fixed snippets per file
    'data_fraction': 0.01,                  # OVERFIT: Use only 1% of data (~12 files)
    'fixed_snippets': True,                 # OVERFIT: Use FIXED snippets - see same data every epoch!
    
    # Model architecture
    'use_cnn_only': False,
    
    # Learning rate scheduler - DISABLED FOR OVERFITTING
    'use_lr_scheduler': False,              # OVERFIT: Keep LR constant for easier debugging
    'scheduler_type': 'cosine_warmup',
    'warmup_epochs': 5,
    'scheduler_min_lr': 1e-6,
    
    # DataLoader optimization settings
    'num_workers': 0,                       # OVERFIT: Single worker for debugging
    'pin_memory': False,
    'prefetch_factor': 2,
    'persistent_workers': False,
    'preload_into_ram': True,               # OVERFIT: Load all into RAM (small dataset)
    
    # Training mode settings
    'shuffle_train': True,                  # Keep shuffling the ORDER, but snippets are fixed
}

# Derived constants
CONFIG_OVERFIT['hop_length'] = int(CONFIG_OVERFIT['sample_rate'] / CONFIG_OVERFIT['roll_fps'])
CONFIG_OVERFIT['snippet_frames'] = int(CONFIG_OVERFIT['snippet_duration'] * CONFIG_OVERFIT['roll_fps'])

MIN_PITCH = 21
MAX_PITCH = 108
NUM_PEDALS = 3
NUM_OUTPUTS = CONFIG_OVERFIT['num_keys'] + NUM_PEDALS  # 88 keys + 3 pedals = 91
