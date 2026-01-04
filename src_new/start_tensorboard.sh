#!/bin/bash
source .venv/bin/activate
echo "Starting TensorBoard..."
echo "Open http://localhost:6006 in your browser"
tensorboard --logdir=.
