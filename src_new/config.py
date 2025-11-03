CONFIG = {
    'sample_rate': 48000,
    'roll_fps': 100,
    'num_keys': 88,
    'n_mels': 88 * 4,                       # 4 bins per key (352 total)
    'n_fft': 4096,                          # Larger FFT for better freq resolution
    'plot_pngs': True,
    
    # Data paths
    'data_dir': './processed_data_17_18',
    'data_fraction': 1.0,
    
    # Training parameters - BALANCED FOR MEMORY
    'snippet_duration': 3.0,
    'batch_size': 16,
    'learning_rate': 1e-4,
    'num_epochs': 200,
    'hidden_size': 256,
    'num_heads': 8,
    'num_layers': 4,
    'dropout': 0.2,
    'weight_decay': 1e-4,
    'split_year_folder': "2018",
    
    # Data augmentation settings - INCREASED BUT REASONABLE
    'snippets_per_file': 50,
    'use_time_masking': False,
    'time_mask_param': 30,
    'use_freq_masking': False,
    'freq_mask_param': 20,
    
    # Learning rate scheduler settings
    'use_lr_scheduler': True,
    'scheduler_type': 'cosine_warmup',
    'warmup_epochs': 5,
    'scheduler_patience': 5,
    'scheduler_factor': 0.5,
    'scheduler_min_lr': 1e-6,
    'scheduler_threshold': 5e-3,
    
    # Loss function settings
    'use_focal_loss': False,
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,
    
    # DataLoader optimization settings - MEMORY CONSCIOUS
    'num_workers': 12,
    'pin_memory': False,
    'prefetch_factor': 2,                   # Back to 2 from 3
    'persistent_workers': True,
    'preload_into_ram': False,              # DISABLED: Too much memory (31GB)
    
    # Training mode settings
    'fixed_snippets': False,
    'shuffle_train': True,
    
    # Ablation study option
    'use_cnn_only': False,                  # Set to True to test without Transformer
}

# Derived constants - hop_length is calculated to ensure alignment
CONFIG['hop_length'] = int(CONFIG['sample_rate'] / CONFIG['roll_fps'])  # Hop length for STFT (calculated)
CONFIG['snippet_frames'] = int(CONFIG['snippet_duration'] * CONFIG['roll_fps'])  # Number of frames per snippet

MIN_PITCH = 21  # Lowest MIDI pitch for 88-key piano (A0)
MAX_PITCH = 108  # Highest MIDI pitch for 88-key piano (C8)
NUM_PEDALS = 3  # Sustain, soft, sostenuto
NUM_OUTPUTS = CONFIG['num_keys'] + NUM_PEDALS  # Total output channels (88 keys + 3 pedals = 91)