#!/bin/bash

# Navigate to the anemll directory
cd "$(dirname "$0")/anemll" || exit 1

echo "Activating ANEMLL virtual environment..."
source env-anemll/bin/activate

echo "Running Qwen2.5-0.5B-Instruct pipeline for Apple Neural Engine..."
echo "This script will download the model, convert it, and run a test inference on the ANE."
echo "Please be patient as the download and conversion take time."
echo "--------------------------------------------------------"

# Use the generic HF conversion test script with Qwen 2.5 0.5B Instruct
./tests/conv/test_hf_model.sh Qwen/Qwen2.5-0.5B-Instruct /tmp/qwen05b


