import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from config import CONFIG

# Simulate the learning rate schedule used in training
def get_lr_schedule(base_lr, num_epochs, warmup_epochs, min_lr):
    """
    Recreate the cosine warmup schedule used in training.
    """
    # Create a dummy optimizer and model
    dummy_model = torch.nn.Linear(10, 10)
    optimizer = optim.Adam(dummy_model.parameters(), lr=base_lr)
    
    # Create the scheduler exactly as in train.py
    main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs - warmup_epochs,
        eta_min=min_lr
    )
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,  # Start at 10% of base LR
        end_factor=1.0,    # Reach 100% at end of warmup
        total_iters=warmup_epochs
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs]
    )
    
    # Collect learning rates for each epoch
    lrs = []
    for epoch in range(num_epochs):
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()
    
    return lrs

# Get parameters from config
base_lr = CONFIG['learning_rate']
num_epochs = CONFIG['num_epochs']
warmup_epochs = CONFIG['warmup_epochs']
min_lr = CONFIG['scheduler_min_lr']

print(f"Simulating LR schedule with:")
print(f"  Base LR: {base_lr}")
print(f"  Warmup epochs: {warmup_epochs}")
print(f"  Total epochs: {num_epochs}")
print(f"  Min LR: {min_lr}")

# Get the learning rate schedule
lrs = get_lr_schedule(base_lr, num_epochs, warmup_epochs, min_lr)

# Find LR at epoch 550
if len(lrs) > 550:
    lr_at_550 = lrs[550]
    print(f"\nLearning rate at epoch 550: {lr_at_550:.2e}")
else:
    print(f"\nWarning: Schedule only has {len(lrs)} epochs")

# Plot the learning rate schedule
plt.figure(figsize=(15, 6))

# Full schedule
plt.subplot(1, 2, 1)
plt.plot(lrs, linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule (Full)')
plt.grid(True, alpha=0.3)
plt.yscale('log')
if len(lrs) > 550:
    plt.axvline(x=550, color='red', linestyle='--', linewidth=2, label=f'Epoch 550 (LR={lr_at_550:.2e})')
    plt.legend()

# Zoomed in around epoch 550
plt.subplot(1, 2, 2)
if len(lrs) > 550:
    start_epoch = max(0, 550 - 100)
    end_epoch = min(len(lrs), 550 + 100)
    plt.plot(range(start_epoch, end_epoch), lrs[start_epoch:end_epoch], linewidth=2)
    plt.axvline(x=550, color='red', linestyle='--', linewidth=2, label=f'Epoch 550')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule (Around Epoch 550)')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.legend()
else:
    plt.text(0.5, 0.5, 'Not enough epochs', ha='center', va='center', transform=plt.gca().transAxes)

plt.tight_layout()
plt.savefig('lr_schedule.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: lr_schedule.png")

# Print some key LR values
print(f"\nKey learning rate values:")
print(f"  Epoch 0: {lrs[0]:.2e}")
print(f"  Epoch {warmup_epochs} (after warmup): {lrs[warmup_epochs]:.2e}")
if len(lrs) > 100:
    print(f"  Epoch 100: {lrs[100]:.2e}")
if len(lrs) > 500:
    print(f"  Epoch 500: {lrs[500]:.2e}")
if len(lrs) > 550:
    print(f"  Epoch 550: {lrs[550]:.2e}")
if len(lrs) > 600:
    print(f"  Epoch 600: {lrs[600]:.2e}")
if len(lrs) > 800:
    print(f"  Epoch 800: {lrs[800]:.2e}")
print(f"  Final epoch: {lrs[-1]:.2e}")
