"""
Configuration file for piano transcription inference.
"""

INFERENCE_CONFIG = {
    # Path to the audio file to transcribe (WAV format)
    'audio_path': 'Test_data/MIDI-UNPROCESSED_01-03_R1_2014_MID--AUDIO_01_R1_2014_wav--2.wav',
    
    # Path to the corresponding MIDI file (for ground truth comparison)
    # Set to None if you don't have ground truth
    'midi_path': 'Test_data/MIDI-UNPROCESSED_01-03_R1_2014_MID--AUDIO_01_R1_2014_wav--2.midi',
    
    # Time range to transcribe (in seconds)
    'start_time': 0.0,  # Start time in seconds
    'end_time': 10.0,   # End time in seconds
    
    # Path to the trained model
    'model_path': 'training_run_001/piano_transcription_model.pth',
    
    # Output directory for results
    'output_dir': 'inference_results',
    
    # Visualization settings
    'plot_full_range': True,  # If True, plot all 91 outputs; if False, focus on active range
}
