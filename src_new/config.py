CONFIG = {
    'sample_rate': 48000,
    'roll_fps': 100,
    'num_keys': 88,
    'n_mels': 88*3,
    'n_fft': 2048,
    'plot_pngs': True,
    
    # Data paths
    'data_dir': './processed_data',
    'data_fraction': 1.0,      # FULL DATASET for proper training
    
    # Training parameters
    'snippet_duration': 2.0,
    'batch_size': 8,          # DOUBLED: More stable gradients, faster convergence
    'learning_rate': 2e-4,     # REDUCED: Stepwise improvements show high LR was overshooting (was 5e-4)
    'num_epochs': 1000,         # Enough for full convergence
    'hidden_size': 256,        # INCREASED: Model may be capacity-limited (was 256)
    'num_heads': 8,
    'num_layers': 4,           # INCREASED: More depth for complex patterns (was 4)
    'dropout': 0.1,            # REDUCED: Less regularization since we're not overfitting (was 0.3)
    'weight_decay': 5e-5,      # REDUCED: Less L2 penalty (was 1e-4)
    'split_year_folder': "2008",
    
    # Data augmentation settings
    'snippets_per_file': 20,   # REDUCED: Less randomness per epoch, more stability (was 30)
    'use_time_masking': False, # Disabled - signals too delicate
    'time_mask_param': 30,
    'use_freq_masking': False, # Disabled - only ~3 bins per key
    'freq_mask_param': 20,
    
    # Learning rate scheduler settings
    'use_lr_scheduler': True,
    'scheduler_type': 'cosine_warmup',  # NEW: Warmup then smooth decay
    'warmup_epochs': 5,        # REDUCED: Shorter warmup since starting LR is already conservative (was 10)
    'scheduler_patience': 25,  # Patient - don't reduce too early
    'scheduler_factor': 0.5,
    'scheduler_min_lr': 1e-6,
    'scheduler_threshold': 5e-3,  # Tolerant of small fluctuations
    
    # Loss function settings
    'use_focal_loss': False,   # BCE works better based on your tests
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,
    
    # DataLoader optimization settings
    'num_workers': 8,
    'pin_memory': False,
    'prefetch_factor': 2,
    'persistent_workers': True,
    
    # Training mode settings (for overfitting tests - DISABLED for normal training)
    'fixed_snippets': False,   # DISABLED: Use random sampling for generalization
    'shuffle_train': True,     # ENABLED: Shuffle for better training
}

# Derived constants - hop_length is calculated to ensure alignment
CONFIG['hop_length'] = int(CONFIG['sample_rate'] / CONFIG['roll_fps'])  # Hop length for STFT (calculated)
CONFIG['snippet_frames'] = int(CONFIG['snippet_duration'] * CONFIG['roll_fps'])  # Number of frames per snippet

MIN_PITCH = 21  # Lowest MIDI pitch for 88-key piano (A0)
MAX_PITCH = 108  # Highest MIDI pitch for 88-key piano (C8)
NUM_PEDALS = 3  # Sustain, soft, sostenuto
NUM_OUTPUTS = CONFIG['num_keys'] + NUM_PEDALS  # Total output channels (88 keys + 3 pedals = 91)