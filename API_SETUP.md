# ANE Studio API Setup Guide

This guide helps you configure the ANE Studio API server for use with external clients like **Zed Editor** and other Ollama-compatible applications.

## Quick Start

1. Start the server:
   ```bash
   cd /Users/jp_mac/workspace/JP/ANE
   python ane_studio/app.py
   ```

2. Test the API:
   ```bash
   python tests/test_endpoints.py --network
   ```

3. Configure your client with the Ollama URL shown in the test output.

## API Endpoints

The following Ollama-compatible endpoints are available:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/version` | GET | Returns server version |
| `/api/tags` | GET | Lists available models |
| `/api/ps` | GET | Shows running models |
| `/api/show` | POST | Get model details |
| `/api/chat` | POST | Chat completion (streaming) |
| `/api/generate` | POST | Text generation (streaming) |
| `/` | GET | Server info (API clients) or UI (browsers) |

## Zed Editor Configuration

### Settings.json

Add this to your Zed settings:

```json
{
  "assistant": {
    "default_model": {
      "provider": "ollama",
      "model": "llama-1b"
    },
    "version": "2"
  }
}
```

### Environment Variable

Set the Ollama API URL:

```bash
export OLLAMA_API_BASE="http://YOUR_IP:11436"
```

Replace `YOUR_IP` with your local network IP (e.g., `192.168.0.11`).

## Testing the API

### Quick Test (Localhost)

```bash
python tests/test_endpoints.py
```

### Network Test (Accessible from other devices)

```bash
python tests/test_endpoints.py --network
```

### Manual Testing

Test version endpoint:
```bash
curl http://127.0.0.1:11436/api/version
```

Test models list:
```bash
curl http://127.0.0.1:11436/api/tags
```

Test chat:
```bash
curl -X POST http://127.0.0.1:11436/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model": "llama-1b", "messages": [{"role": "user", "content": "Hi"}]}'
```

## Network Troubleshooting

### macOS Firewall

If you can't connect from other devices, check your firewall:

1. Open **System Preferences** > **Security & Privacy** > **Firewall**
2. If the firewall is on, you may need to:
   - Click **Firewall Options**
   - Add Python to the allowed applications, OR
   - Temporarily disable the firewall for testing

### First Run Prompt

On the first run, macOS will show a dialog asking to allow incoming connections. **Click "Allow"** or the server won't be accessible from the network.

### Check Server Binding

The server should bind to `0.0.0.0` (all interfaces). This is configured in the startup scripts.

Verify with:
```bash
lsof -i :11436
```

You should see `*:11436` indicating it's listening on all interfaces.

## Running Tests

### All Tests
```bash
python tests/test_endpoints.py --url http://127.0.0.1:11436
```

### Network Accessibility
```bash
python tests/test_endpoints.py --network
```

### Available Models
```bash
curl http://127.0.0.1:11436/api/tags | python3 -m json.tool
```

## Common Issues

### 404 Not Found on /api/version

The server wasn't restarted after updates. Kill the old process and restart:
```bash
pkill -f "ane_studio/server.py"
python ane_studio/app.py
```

### Connection Refused

The server isn't running. Start it with:
```bash
python ane_studio/app.py
```

### CORS Errors in Browser

CORS is enabled with `allow_origins=["*"]` which allows all origins. If you see CORS errors, check that the server is using the updated code.

### Model Not Found Errors

Make sure you have models installed:
```bash
curl http://127.0.0.1:11436/api/tags
```

If no models are listed, download one through the web UI first.

## API Response Formats

### Version Response
```json
{
  "version": "0.3.0"
}
```

### Models Response
```json
{
  "models": [
    {
      "name": "llama-1b",
      "model": "llama-1b",
      "details": {
        "family": "llama",
        "parameter_size": "1B",
        "quantization_level": "ANE_FP16"
      }
    }
  ]
}
```

### Chat Stream Format (NDJSON)
```
{"model": "llama-1b", "message": {"role": "assistant", "content": "Hello"}, "done": false}
{"model": "llama-1b", "message": {"role": "assistant", "content": " there"}, "done": false}
{"model": "llama-1b", "message": {"role": "assistant", "content": ""}, "done": true}
```

## Support

If you encounter issues:

1. Run the diagnostics: `python tests/diagnose_network.py`
2. Check server logs in the terminal where you started `app.py`
3. Verify all endpoints with: `python tests/test_endpoints.py`
