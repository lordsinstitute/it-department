from detector import create_app
import threading
import webbrowser
import time

app = create_app()


def open_browser():
    try:
        time.sleep(1.5)
        webbrowser.open("http://127.0.0.1:5000")
    except Exception:
        pass


if __name__ == "__main__":
    threading.Thread(target=open_browser, daemon=True).start()
    app.run(host="127.0.0.1", port=5000, debug=False)