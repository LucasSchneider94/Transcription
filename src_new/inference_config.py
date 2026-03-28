"""
Configuration file for piano transcription inference.
"""

INFERENCE_CONFIG = {
    # Path to the audio file to transcribe (WAV format)
    'audio_path': 'src_new/Test_data/MIDI-UNPROCESSED_01-03_R1_2014_MID--AUDIO_01_R1_2014_wav--2.wav',
    #MIDI-UNPROCESSED_01-03_R1_2014_MID--AUDIO_01_R1_2014_wav--2 (Scarlatti A-Major Sonata)
    #MIDI-UNPROCESSED_01-03_R1_2014_MID--AUDIO_01_R1_2014_wav--1 (Scarlatti D-Major Sonata)
    #MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1 (Berg Sonata)

    # Path to the corresponding MIDI file (for ground truth comparison)
    # Set to None if you don't have ground truth
    'midi_path': 'src_new/Test_data/MIDI-UNPROCESSED_01-03_R1_2014_MID--AUDIO_01_R1_2014_wav--2.midi',
    
    # Time range to transcribe (in seconds)
    'start_time': 0.0,  # Start time in seconds
    'end_time': 20.0,   # End time in seconds
    
    # Path to the trained model
    'model_path': 'training_run_002/model.pth',
    
    # Output directory for results
    'output_dir': 'inference_results',
    
    # Visualization settings
    'plot_full_range': True,  # If True, plot all 91 outputs; if False, focus on active range
}
