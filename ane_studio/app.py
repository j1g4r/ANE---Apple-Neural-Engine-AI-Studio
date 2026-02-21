"""
ANE Studio — Native Mac Application Launcher
Uses pywebview to create a native macOS WebKit window
and runs the FastAPI backend in a background thread.
"""
import os
import sys
import threading
import time
import signal

# Force stable temp directory before any other imports
os.makedirs(os.path.join(os.path.dirname(__file__), "tmp_cache"), exist_ok=True)
os.environ["TMPDIR"] = os.path.join(os.path.dirname(__file__), "tmp_cache")

def start_server(port=11436):
    """Start the FastAPI server in a background thread."""
    from server import run_server
    run_server(port=port, host="0.0.0.0")

def wait_for_server(port, timeout=15):
    import urllib.request
    import urllib.error
    start_time = time.time()
    url = f"http://127.0.0.1:{port}/api/server/status"
    while time.time() - start_time < timeout:
        try:
            with urllib.request.urlopen(url) as response:
                if response.status == 200:
                    return True
        except urllib.error.URLError:
            time.sleep(0.5)
    return False

def main():
    port = 11436

    # Start FastAPI in background thread
    server_thread = threading.Thread(target=start_server, args=(port,), daemon=True)
    server_thread.start()

    # Wait dynamically for the server to be fully ready
    if not wait_for_server(port):
        print("Warning: Server didn't start in time. Window might be blank.")

    try:
        import webview
        # Create native macOS window
        webview.create_window(
            title="ANE Studio",
            url=f"http://127.0.0.1:{port}",
            width=1280,
            height=820,
            min_size=(900, 600),
            resizable=True,
            text_select=True,
        )
        webview.start(debug=False)
    except ImportError:
        # Fallback: open in browser if pywebview not installed
        import webbrowser
        print(f"pywebview not found. Opening in browser instead.")
        print(f"ANE Studio running at: http://127.0.0.1:{port}")
        webbrowser.open(f"http://127.0.0.1:{port}")
        # Keep the server alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down ANE Studio...")


if __name__ == "__main__":
    main()
