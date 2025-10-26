CONFIG = {
    'sample_rate': 48000,      # Sampling rate for audio processing
    'roll_fps': 100,           # Frames per second for piano roll (controllable parameter)
    'num_keys': 88,            # Number of piano keys (88 for grand pianos)
    'n_mels': 88*3,             # Number of Mel frequency bins
    'n_fft': 2048,             # FFT window size
    'plot_pngs': True,        # Boolean to control whether PNGs are plotted for all files
    
    # Data paths
    'data_dir': './data_subset',  # Directory containing processed data (use './processed_data' for full dataset)
    
    # Training parameters
    'snippet_duration': 3.0,   # Duration of each training snippet in seconds
    'batch_size': 8,           # Batch size for training
    'learning_rate': 1e-4,     # Learning rate for optimizer
    'num_epochs': 80,         # Number of training epochs
    'hidden_size': 256,        # Hidden size for transformer
    'num_heads': 8,            # Number of attention heads in transformer
    'num_layers': 4,           # Number of transformer layers
    'dropout': 0.3,            # Dropout rate
    'split_year_folder': "2008" # Subset of the MAESTRO dataset to process
}

# Derived constants - hop_length is calculated to ensure alignment
CONFIG['hop_length'] = int(CONFIG['sample_rate'] / CONFIG['roll_fps'])  # Hop length for STFT (calculated)
CONFIG['snippet_frames'] = int(CONFIG['snippet_duration'] * CONFIG['roll_fps'])  # Number of frames per snippet

MIN_PITCH = 21  # Lowest MIDI pitch for 88-key piano (A0)
MAX_PITCH = 108  # Highest MIDI pitch for 88-key piano (C8)
NUM_PEDALS = 3  # Sustain, soft, sostenuto
NUM_OUTPUTS = CONFIG['num_keys'] + NUM_PEDALS  # Total output channels (88 keys + 3 pedals = 91)