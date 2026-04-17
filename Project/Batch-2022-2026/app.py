import threading
import webbrowser
import time
from detector import create_app

app = create_app()

def open_browser():
    """
    Opens browser automatically after Flask server starts.
    Works in both normal Python and PyInstaller EXE.
    """
    time.sleep(1.5)  # small delay to allow server to start
    webbrowser.open("http://127.0.0.1:5000")

if __name__ == "__main__":
    # Launch browser in background thread
    threading.Thread(target=open_browser, daemon=True).start()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False  # IMPORTANT: prevents double browser opening
    )