import os
import socket
import threading
import time
import webbrowser

from detector import create_app

APP_HOST = "127.0.0.1"
APP_PORT = 5000

app = create_app()

def _is_port_open(host: str, port: int, timeout: float = 0.3) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False

def open_browser_when_ready(url: str, max_wait_seconds: int = 12):
    """
    Wait until Flask server is actually listening, then open the browser.
    Works in normal Python + PyInstaller EXE.
    Never crashes the app if browser open fails.
    """
    try:
        waited = 0.0
        # Wait for the server to start listening
        while waited < max_wait_seconds:
            if _is_port_open(APP_HOST, APP_PORT):
                break
            time.sleep(0.25)
            waited += 0.25

        # Open browser (best-effort)
        webbrowser.open(url, new=1)
        print(f"[INFO] Opening browser at {url}")
    except Exception as e:
        print(f"[WARN] Could not open browser automatically: {e}")

if __name__ == "__main__":
    url = f"http://{APP_HOST}:{APP_PORT}"

    # Background thread so Flask startup isn't blocked
    t = threading.Thread(target=open_browser_when_ready, args=(url,), daemon=True)
    t.start()

    # IMPORTANT: disable reloader, otherwise browser opens twice
    app.run(host=APP_HOST, port=APP_PORT, debug=False, use_reloader=False)