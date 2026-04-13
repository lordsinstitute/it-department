from flask import Flask, render_template, request
from cryptography.fernet import Fernet
import qrcode
import io
import base64
import hashlib
from urllib.parse import quote, unquote

app = Flask(__name__)


def generate_key_from_password(password):
    return base64.urlsafe_b64encode(hashlib.sha256(password.encode()).digest())


def encode_token(token: str) -> str:
    """Encode token to be fully URL-safe with no = padding"""
    return base64.urlsafe_b64encode(token.encode()).decode().rstrip("=")


def decode_token(encoded: str) -> str:
    """Restore padding and decode back to original token"""
    padding = 4 - len(encoded) % 4
    if padding != 4:
        encoded += "=" * padding
    return base64.urlsafe_b64decode(encoded.encode()).decode()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    message = request.form.get("message")
    password = request.form.get("password")

    if not message or not password:
        return "Message and password required"

    key = generate_key_from_password(password)
    f = Fernet(key)

    encrypted_message = f.encrypt(message.encode()).decode()

    # Double-encode so no special chars remain in URL
    safe_token = encode_token(encrypted_message)

    full_url = request.url_root + "decrypt/" + safe_token

    # Force HTTPS for Render
    if full_url.startswith("http://"):
        full_url = full_url.replace("http://", "https://", 1)

    qr = qrcode.make(full_url)

    buffer = io.BytesIO()
    qr.save(buffer, format="PNG")
    buffer.seek(0)

    img_str = base64.b64encode(buffer.getvalue()).decode()

    return render_template("qr.html", qr_image=img_str)


@app.route("/decrypt/<path:token>", methods=["GET", "POST"])
def decrypt(token):
    try:
        # Decode the double-encoded token back to Fernet token
        fernet_token = decode_token(token)
    except Exception:
        return "Invalid or corrupted QR code token."

    if request.method == "POST":
        password = request.form.get("password")
        try:
            key = generate_key_from_password(password)
            f = Fernet(key)
            decrypted = f.decrypt(fernet_token.encode()).decode()
            return render_template("message.html", message=decrypted)
        except Exception:
            return render_template("password.html", error="Wrong password, try again.")

    return render_template("password.html", error=None)


if __name__ == "__main__":
    app.run(debug=False)