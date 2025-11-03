"""
Prepare high-resolution dataset (4 bins per key, larger FFT).
Run this to generate data with better frequency resolution.
"""

import os
import sys
from config_high_res import CONFIG_HIGH_RES

# Import the data preparation module
from data_preparation import process_dataset

if __name__ == "__main__":
    # Prepare 2018 data with high resolution
    maestro_dir = "../maestro-v3.0.0"
    year_folder = "2018"
    data_dir = os.path.join(maestro_dir, year_folder)
    
    # Output to a separate directory for high-res data
    output_dir = f"./processed_data_{year_folder}_4bins"
    
    print(f"\n{'='*80}")
    print(f"PREPARING HIGH-RESOLUTION DATASET")
    print(f"{'='*80}")
    print(f"Source: {data_dir}")
    print(f"Output: {output_dir}")
    print(f"Resolution: {CONFIG_HIGH_RES['n_mels']} mel bins (4 per key)")
    print(f"FFT size: {CONFIG_HIGH_RES['n_fft']} samples")
    print(f"{'='*80}\n")
    
    # Process with high-res config
    process_dataset(data_dir, output_dir, CONFIG_HIGH_RES)
    
    print(f"\n{'='*80}")
    print(f"HIGH-RESOLUTION DATA READY!")
    print(f"{'='*80}")
    print(f"To train with this data, update config.py:")
    print(f"  'data_dir': '{output_dir}'")
    print(f"  'n_mels': {CONFIG_HIGH_RES['n_mels']}")
    print(f"  'n_fft': {CONFIG_HIGH_RES['n_fft']}")
    print(f"{'='*80}\n")
