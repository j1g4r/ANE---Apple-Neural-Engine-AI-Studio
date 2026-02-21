#!/bin/bash

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

# Force a stable temporary directory locally to bypass macOS restrictions
mkdir -p "$PWD/tmp_cache"
export TMPDIR="$PWD/tmp_cache"

# Determine local network IP
LOCAL_IP=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}' | head -n 1)
if [ -z "$LOCAL_IP" ]; then
    LOCAL_IP="127.0.0.1"
fi

# Run the API Server in the background
# We run uvicorn on port 11436 to avoid typical Laravel 8000 conflicts
uvicorn ane_studio.server:app --host 0.0.0.0 --port 11436 &
SERVER_PID=$!

sleep 3

echo ""
echo "==========================================="
echo "🚀 ANE Server is LIVE"
echo "🔗 Local Access:   http://127.0.0.1:11436"
echo "🌐 Network Access: http://$LOCAL_IP:11436"
echo "==========================================="
echo ""

# Open browser locally
open http://127.0.0.1:11436

echo "Press Ctrl+C to stop the server and exit out."

cleanup() {
    echo "Stopping servers..."
    kill $SERVER_PID 2>/dev/null
    exit 0
}

trap cleanup INT TERM
wait $SERVER_PID
