from cryptography.fernet import Fernet
import base64
import hashlib
import qrcode

def generate_key(password):
    return base64.urlsafe_b64encode(hashlib.sha256(password.encode()).digest())

def encrypt_message(message, password):
    key = generate_key(password)
    cipher = Fernet(key)
    encrypted = cipher.encrypt(message.encode())
    return encrypted.decode()

def generate_qr(data):
    img = qrcode.make(data)
    img.save("/tmp/secure_qr.png")