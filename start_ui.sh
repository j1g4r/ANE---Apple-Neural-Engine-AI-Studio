#!/bin/bash

# Navigate to the workspace where the python environment is
cd "$(dirname "$0")" || exit 1

echo "Activating ANEMLL virtual environment for the AI server..."
source anemll/env-anemll/bin/activate

# Ensure fastapi and uvicorn are installed
pip install fastapi uvicorn sse_starlette

echo ""
echo "==========================================="
echo "   Apple Neural Engine (ANE) Web UI        "
echo "==========================================="
echo "Starting FastAPI Inference Server..."

# Run the API Server in the background
# We run uvicorn on port 11435 to avoid typical Laravel 8000 conflicts
uvicorn chat_server:app --host 0.0.0.0 --port 11436 &
SERVER_PID=$!

# Wait lightly for server to boot up
sleep 3

# Open the default web browser (macOS uses 'open')
echo "Opening browser to http://127.0.0.1:11436"
open http://127.0.0.1:11436

echo "Press Ctrl+C to stop the server and exit out."
# Wait for the background process so the script doesn't exit immediately
wait $SERVER_PID
