"""
Grid search configurations for time/frequency tradeoff experiments.

Usage:
1. Run data_preparation.py with different GRID_CONFIGS
2. Train models with each configuration
3. Compare validation performance (not just overfitting)

Key tradeoffs:
- Higher fps (100-200): Better onset detection, worse frequency resolution
- Lower fps (50-75): Better frequency resolution, worse onset timing
- More mel bins (3-5x keys): Better pitch separation, larger model needed
- Larger n_fft: Better frequency resolution, worse time resolution
"""

GRID_CONFIGS = {
    # Baseline (current)
    'baseline': {
        'roll_fps': 100,
        'n_mels': 88 * 3,  # 264 bins, ~3 per key
        'n_fft': 2048,
        'description': 'Current config: 10ms time, ~3 bins/key'
    },
    
    # Higher time resolution
    'high_time_res': {
        'roll_fps': 150,
        'n_mels': 88 * 3,
        'n_fft': 2048,
        'description': 'Better onset detection: ~6.7ms time, ~3 bins/key'
    },
    
    # Higher frequency resolution
    'high_freq_res': {
        'roll_fps': 75,
        'n_mels': 88 * 4,  # 352 bins, ~4 per key
        'n_fft': 4096,
        'description': 'Better pitch separation: ~13ms time, ~4 bins/key'
    },
    
    # Balanced high resolution
    'balanced_high': {
        'roll_fps': 100,
        'n_mels': 88 * 4,  # 352 bins
        'n_fft': 4096,
        'description': 'Both improved: 10ms time, ~4 bins/key'
    },
    
    # Lower resolution (faster training)
    'efficient': {
        'roll_fps': 50,
        'n_mels': 88 * 2,  # 176 bins, ~2 per key
        'n_fft': 2048,
        'description': 'Faster training: 20ms time, ~2 bins/key'
    },
    
    # Extreme time resolution (for very fast notes)
    'extreme_time': {
        'roll_fps': 200,
        'n_mels': 88 * 3,
        'n_fft': 1024,
        'description': 'Fastest onsets: 5ms time, ~3 bins/key, ~46Hz freq res'
    },
}

def get_config(name='baseline'):
    """Get a grid search configuration by name."""
    if name not in GRID_CONFIGS:
        raise ValueError(f"Config '{name}' not found. Available: {list(GRID_CONFIGS.keys())}")
    
    config = GRID_CONFIGS[name].copy()
    sample_rate = 48000
    
    # Calculate derived values
    config['sample_rate'] = sample_rate
    config['hop_length'] = int(sample_rate / config['roll_fps'])
    config['freq_resolution'] = sample_rate / config['n_fft']
    config['time_resolution_ms'] = 1000 / config['roll_fps']
    config['window_duration_ms'] = 1000 * config['n_fft'] / sample_rate
    config['bins_per_key'] = config['n_mels'] / 88
    
    return config

def print_config_comparison():
    """Print a comparison table of all configurations."""
    print("\n" + "="*100)
    print("TIME/FREQUENCY TRADEOFF GRID SEARCH CONFIGURATIONS")
    print("="*100)
    print(f"{'Config':<15} {'FPS':<6} {'Mel Bins':<10} {'FFT':<6} {'Time Res':<10} {'Bins/Key':<10} {'Description':<40}")
    print("-"*100)
    
    for name, cfg in GRID_CONFIGS.items():
        config = get_config(name)
        print(f"{name:<15} {config['roll_fps']:<6} {config['n_mels']:<10} {config['n_fft']:<6} "
              f"{config['time_resolution_ms']:.1f}ms{'':<5} {config['bins_per_key']:.1f}{'':<7} "
              f"{cfg['description']:<40}")
    
    print("="*100)
    print("\nRecommended experiment order:")
    print("1. baseline (current) - establish baseline performance")
    print("2. high_time_res - test if onset detection is limiting factor")
    print("3. high_freq_res - test if pitch separation is limiting factor")
    print("4. balanced_high - best of both (but larger model/slower)")
    print("\nNote: After changing config, you must re-run data_preparation.py!")
    print("="*100 + "\n")

if __name__ == "__main__":
    print_config_comparison()
    
    # Example: Show details for a specific config
    print("\nExample - Baseline config details:")
    config = get_config('baseline')
    for key, value in config.items():
        if key != 'description':
            print(f"  {key}: {value}")
