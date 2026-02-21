#!/bin/bash

echo "Setting up MLX for Apple Silicon..."

# Check if python3 is installed
if ! command -v python3 &> /dev/null
then
    echo "Python3 is not installed. Please install Python3 first."
    exit
fi

# Create and activate a virtual environment
python3 -m venv aienv
source aienv/bin/activate

# Install the necessary MLX and HuggingFace libraries
pip install --upgrade pip
pip install mlx-lm

echo "Setup Complete!"
echo "To activate the environment in the future, run: source aienv/bin/activate"
echo "To run your first model, try: python run_mlx.py"
