# Piano Transcription Model

A deep learning model for automatic piano transcription using spectrograms and transformer-based architecture.

## Features

- **Multi-output prediction**: 88 piano keys + 3 pedals (sustain, soft, sostenuto)
- **Transformer architecture**: Attention-based model for temporal dependencies
- **Memory-efficient training**: Handles large datasets using memory-mapped files
- **Checkpoint system**: Resume training from any epoch
- **Comprehensive metrics**: Precision, Recall, F1 score tracking
- **Inference script**: Test trained models on new audio

## Project Structure

```
src_new/
├── config.py              # Main configuration file
├── model.py               # Transformer model architecture
├── data_preparation.py    # MAESTRO dataset preprocessing
├── train.py               # Training script
├── resume_training.py     # Resume training from checkpoint
├── resume_config.py       # Resume configuration
├── inference.py           # Model inference script
├── inference_config.py    # Inference configuration
└── .gitignore            # Git ignore rules
```

## Requirements

- Python 3.8+
- PyTorch
- librosa
- pretty_midi
- numpy
- matplotlib
- tqdm
- soundfile
- scipy

## Installation

```bash
pip install torch librosa pretty_midi numpy matplotlib tqdm soundfile scipy
```

## Usage

### 1. Data Preparation

Process the MAESTRO dataset:

```python
# Edit config.py to set your data paths
python data_preparation.py
```

### 2. Training

Start training from scratch:

```python
# Configure training parameters in config.py
python train.py
```

Training will create a new folder `training_run_XXX/` containing:
- `model.pth` - Best model checkpoint
- `checkpoint.pth` - Full checkpoint for resuming
- `training_curves.png` - Training metrics visualization
- `predictions_epoch_X.png` - Prediction samples

### 3. Resume Training

To continue training from a checkpoint:

```python
# Edit resume_config.py
RESUME_CONFIG = {
    'checkpoint_path': 'training_run_012/checkpoint.pth',
    'continue_in_same_folder': True,
    'additional_epochs': 50,
}

# Run resume script
python resume_training.py
```

### 4. Inference

Test your trained model:

```python
# Edit inference_config.py
INFERENCE_CONFIG = {
    'audio_path': 'path/to/audio.wav',
    'midi_path': 'path/to/midi.mid',  # Optional
    'start_time': 0.0,
    'end_time': 10.0,
    'model_path': 'training_run_012/model.pth',
    'output_dir': 'inference_results',
}

# Run inference
python inference.py
```

## Model Architecture

- **Input**: Mel-spectrogram (128 mel bins)
- **Encoder**: Convolutional layers for feature extraction
- **Transformer**: Multi-head self-attention with positional encoding
- **Output**: 91 binary predictions (88 keys + 3 pedals)

## Configuration

Key parameters in `config.py`:

- `num_epochs`: Training epochs
- `batch_size`: Batch size
- `learning_rate`: Learning rate
- `snippet_frames`: Frames per training snippet
- `hidden_size`: Transformer hidden dimension
- `num_heads`: Attention heads
- `num_layers`: Transformer layers

## Training Features

- **Reproducible splits**: Fixed random seed for train/val split
- **Data augmentation**: Random snippet sampling each epoch
- **Automatic checkpointing**: Save progress every epoch
- **Progress monitoring**: Training curves updated every 10 epochs
- **Early stopping**: Best model saved based on validation loss

## License

MIT License

## Acknowledgments

- MAESTRO dataset: https://magenta.tensorflow.org/datasets/maestro
- Inspired by Onsets and Frames architecture
