# ANE Studio Project Guidelines

## Core Principles
- **Performance First**: Optimize for Apple Neural Engine (ANE).
- **Responsive UI**: Glassmorphism, smooth animations, and real-time feedback.
- **Security**: Bound to 0.0.0.0 for network access but mindful of local execution.
- **Maintainability**: Clear separation of logic between `anemll` (core) and `ane_studio` (API/UI).

## Tech Stack
- **Backend**: FastAPI (Python 3.12+).
- **Frontend**: Vue 3 (CDN), Tailwind CSS (CDN).
- **Inference**: anemll (Apple Neural Engine via CoreML).

## Network Access
- The API server should be accessible across the local network.
- Ensure `uvicorn` is bound to `0.0.0.0`.
- Display the correct local IP address in the UI for users to connect from other devices (e.g., Zed editor on another machine).
