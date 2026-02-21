#!/bin/bash

echo "Setting up Core ML / ANEMLL environment for Apple Neural Engine..."

cd anemll
export PYTHON_PATH="/opt/homebrew/opt/python@3.9/bin/python3.9"

echo "Creating python 3.9 environment..."
$PYTHON_PATH -m venv env-anemll
source env-anemll/bin/activate

echo "Installing dependencies..."
./install_dependencies.sh

echo "Setup Complete!"
echo "To convert a model to Core ML and run on Neural Engine, run: bash run_coreml.sh (or inspect it for commands)"
