"""
Configuration file for resuming training from a checkpoint.
"""

RESUME_CONFIG = {
    # Path to the checkpoint to resume from (set to None to start fresh)
    # Example: 'training_run_012/checkpoint_epoch_100.pth'
    'checkpoint_path': "training_run_013/checkpoint.pth",
    
    # If resuming, should we continue in the same folder or create a new one?
    # True = continue in same folder, False = create new training run folder
    'continue_in_same_folder': True,
    
    # Number of additional epochs to train (added to the checkpoint epoch)
    # If checkpoint was at epoch 100 and additional_epochs=50, will train to epoch 150
    'additional_epochs': None,  # Set to None to use CONFIG['num_epochs'] as total
}
