#!/bin/bash

# Navigate to the workspace where the python environment is
cd "$(dirname "$0")" || exit 1

echo "======================================================"
echo "   Apple Neural Engine (ANE) - Model Downloader       "
echo "======================================================"
echo ""
echo "This script will download, optimize, and compile models"
echo "to run natively on your M4 Mac Mini's Neural Engine."
echo ""
echo "Select a model to compile:"
echo "1) Qwen 2.5 3B Instruct (Recommended - ~6GB RAM, Fast)"
echo "2) Qwen 2.5 7B Instruct (Heavy - ~14GB RAM)"
echo "3) Gemma 3 4B IT (Requires HuggingFace account & License Acceptance)"
echo "4) Exit"
echo ""
read -p "Enter choice [1-4]: " choice

source anemll/env-anemll/bin/activate

case $choice in
    1)
        echo "Starting conversion for Qwen/Qwen2.5-3B-Instruct..."
        echo "This may take 10-20 minutes. Please be patient."
        ./anemll/tests/conv/test_hf_model.sh Qwen/Qwen2.5-3B-Instruct /tmp/qwen3b
        echo "Done! You can now select 'qwen3b' in the UI or use it via the API."
        ;;
    2)
        echo "Starting conversion for Qwen/Qwen2.5-7B-Instruct..."
        echo "This may take 20-40 minutes and heavy RAM usage. Please be patient."
        ./anemll/tests/conv/test_hf_model.sh Qwen/Qwen2.5-7B-Instruct /tmp/qwen7b
        echo "Done! You can now select 'qwen7b' in the UI or use it via the API."
        ;;
    3)
        echo "Checking if you are logged into HuggingFace..."
        huggingface-cli whoami
        if [ $? -ne 0 ]; then
            echo "You must log in to HuggingFace first. Please run 'huggingface-cli login'"
            echo "And make sure you have accepted the Gemma 3 license at:"
            echo "https://huggingface.co/google/gemma-3-4b-it"
            exit 1
        fi
        echo "Starting conversion for google/gemma-3-4b-it..."
        echo "This may take 15-30 minutes. Please be patient."
        ./anemll/tests/conv/test_hf_model.sh google/gemma-3-4b-it /tmp/gemma3_4b
        echo "Done! You can now select 'gemma3_4b' in the UI or use it via the API."
        ;;
    4)
        echo "Exiting."
        exit 0
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "To restart the web UI and load the new model, run:"
echo "bash start_ui.sh"
