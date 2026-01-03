CONFIG = {
    'sample_rate': 48000,
    'roll_fps': 100,
    'num_keys': 88,
    'n_mels': 88 * 4,                       # 4 bins per key (352 total)
    'n_fft': 4096,                          # Larger FFT for better freq resolution
    'plot_pngs': True,
    
    # Data paths
    'data_dir': './processed_data_17_18',
    'split_year_folder': "2018",
    'maestro_json': './maestro-v3.0.0.json',  # Path to MAESTRO metadata for proper train/val/test split
    
    # Onset/Duration parameters (NEW)
    'onset_frames': 2,                      # Mark onset in N consecutive frames (helps with alignment)
    'duration_mode': 'log',                 # 'bins' for classification, 'log' for log-regression, 'linear' for linear regression
    'num_duration_bins': 8,                 # Number of duration bins (only used if duration_mode='bins')
    'min_note_duration': 0.05,              # Minimum note duration in seconds
    
    # Training parameters
    'snippet_duration': 3.0,
    'batch_size': 32,                        # REDUCED for faster testing
    'learning_rate': 1e-4,
    'num_epochs': 500,                       # REDUCED for quick test
    'hidden_size': 256,
    'num_heads': 8,
    'num_layers': 4,
    'dropout': 0.2,
    'weight_decay': 1e-4,
    
    # Multi-task loss weights (UPDATED for onset + duration)
    'onset_weight': 10.0,                   # INCREASED: Onset detection is the hardest task
    'duration_weight': 1.0,                 # DECREASED: Duration is masked (only at onsets), easier task
    'frame_weight': 0.1,                    # DECREASED: Frame is auxiliary, should not dominate
    'consistency_weight': 0.5,              # Temporal consistency loss weight
    
    # Focal Loss parameters (for handling extreme class imbalance)
    'use_focal_loss': True,                 # Use Focal Loss instead of BCE
    'onset_focal_alpha': 0.90,              # INCREASED: Alpha for onset (need more focus on rare positives)
    'onset_focal_gamma': 2.0,               # Gamma for onset (2.0 = standard)
    'frame_focal_alpha': 0.25,              # Alpha for frame (lower since frames are less rare)
    'frame_focal_gamma': 2.0,               # Gamma for frame
    
    # Data augmentation settings
    'snippets_per_file': 50,                # REDUCED for faster testing
    'data_fraction': 0.15,                     # REDUCED: Use 5% of dataset for quick test
    
    # Model architecture
    'use_cnn_only': False,                  # If True, use CNN-only model (ablation study)
    
    # Learning rate scheduler settings
    'use_lr_scheduler': True,
    'scheduler_type': 'cosine_warmup',
    'warmup_epochs': 5,                     # REDUCED for quick test
    'scheduler_min_lr': 1e-6,
    
    # DataLoader optimization settings
    'num_workers': 12,                       # Set to 0 for debugging/testing
    'pin_memory': False,
    'prefetch_factor': 2,
    'persistent_workers': True,            # Set to False when num_workers=0
    'preload_into_ram': False,
    
    # Training mode settings
    'shuffle_train': True,
}

# Derived constants
CONFIG['hop_length'] = int(CONFIG['sample_rate'] / CONFIG['roll_fps'])  # Hop length for STFT
CONFIG['snippet_frames'] = int(CONFIG['snippet_duration'] * CONFIG['roll_fps'])  # Frames per snippet

MIN_PITCH = 21  # Lowest MIDI pitch for 88-key piano (A0)
MAX_PITCH = 108  # Highest MIDI pitch for 88-key piano (C8)
NUM_PEDALS = 3  # Sustain, soft, sostenuto
NUM_OUTPUTS = CONFIG['num_keys'] + NUM_PEDALS  # 88 keys + 3 pedals = 91