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
    'clip_duration_to_snippet': True,       # If True, clip target durations to snippet boundaries
    
    # Training parameters
    'snippet_bins': 256,                    # Snippet length in bins (2.56s at 100fps). Must be divisible by 64.
    'batch_size': 4,                        # REDUCED for faster testing
    'learning_rate': 5e-5,
    'num_epochs': 500,                       # REDUCED for quick test
    'hidden_size': 256,                     # Legacy param, kept for compatibility
    'dropout': 0.2,
    'weight_decay': 1e-7,
    
    # Multi-task loss weights (Simplified for U-Net)
    'onset_loss_weight': 1.0,
    'duration_loss_weight': 1.0,
    'frame_loss_weight': 1.0,
    
    # Smooth Loss parameters (NEW)
    'onset_tolerance_initial_sigma': 10.0,           # Initial σ in bins (30ms at 100fps)
    'onset_tolerance_final_sigma': 0.01,             # Final σ (nearly delta function)
    'onset_tolerance_anneal_f1_threshold': 0.4,     # Start annealing when train F1 > this
    'onset_tolerance_anneal_epochs': 50,            # Anneal over N epochs

    # U-Net Architecture parameters (NEW)
    'use_unet': True,                               # Use U-Net architecture
    'unet_encoder_channels': [352, 256, 256],        # Channel progression in encoder
    'unet_decoder_channels': [128, 64, 32],         # Channel progression in decoder
    'unet_downsample_factor': 2,                    # Time reduction per layer (4^3 = 64 total)
    'unet_num_layers': 3,                           # Number of encoder/decoder layers
    
    # Transformer parameters (Updated for bottleneck)
    'transformer_dim': 2048,                         # INCREASED from 256
    'num_heads': 8,                                
    'num_layers': 4,                                
    'transformer_ff_dim': 2048,                     # Explicit feedforward dim
    
    # Data augmentation settings
    'snippets_per_file': 50,                # REDUCED for faster testing
    'data_fraction': 1.0,                     # REDUCED: Use 5% of dataset for quick test
    'fixed_snippets': False,                # ONLY True for OVERFIT: Use FIXED snippets - see same data every epoch!

    # Model architecture
    'use_cnn_only': False,                  # If True, use CNN-only model (ablation study)
    
    # Learning rate scheduler settings
    'use_lr_scheduler': True,
    'scheduler_type': 'cosine_warmup',
    'warmup_epochs': 0,                     # REDUCED for quick test
    'scheduler_min_lr': 1e-7,
    
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

def validate_config(config):
    """Validate config parameters."""
    if config.get('use_unet', False):
        # Check snippet_bins divisibility
        total_reduction = config['unet_downsample_factor'] ** config['unet_num_layers']
        assert config['snippet_bins'] % total_reduction == 0, \
            f"snippet_bins ({config['snippet_bins']}) must be divisible by {total_reduction}"
        
        # Check transformer parameters match bottleneck
        expected_bottleneck_size = config['snippet_bins'] // total_reduction
        print(f"✓ Snippet bins: {config['snippet_bins']}")
        print(f"✓ Bottleneck size: {expected_bottleneck_size} frames")
        print(f"✓ Transformer will see {expected_bottleneck_size} time steps")
    
    # Check sigma annealing makes sense
    assert config['onset_tolerance_initial_sigma'] > config['onset_tolerance_final_sigma'], \
        "Initial sigma must be > final sigma"
    
    print(f"✓ Config validation passed")

validate_config(CONFIG)

MIN_PITCH = 21  # Lowest MIDI pitch for 88-key piano (A0)
MAX_PITCH = 108  # Highest MIDI pitch for 88-key piano (C8)
NUM_PEDALS = 3  # Sustain, soft, sostenuto
NUM_OUTPUTS = CONFIG['num_keys'] + NUM_PEDALS  # 88 keys + 3 pedals = 91