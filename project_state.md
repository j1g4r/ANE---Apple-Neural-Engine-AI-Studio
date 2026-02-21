# ANE Studio Project State

## Completed Tasks
- [x] Initial FastAPI server implementation.
- [x] Model download and compilation pipeline.
- [x] Chat interface with streaming support.
- [x] Model hub and registry management.
- [x] HuggingFace token integration.

## Current Task
- **Investigating Network Accessibility**: The app is accessible locally but not via the network.

## Pending Tasks
- [ ] Verify `0.0.0.0` binding in all server entry points.
- [ ] Dynamically detect and display local network IP in the UI.
- [ ] Fix port discrepancies between `boot_api.sh`, `chat_server.py`, and `ane_studio/server.py`.
- [ ] Verify macOS firewall status if accessibility issues persist.
