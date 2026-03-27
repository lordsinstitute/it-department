from cryptography.fernet import Fernet
import base64
import hashlib

def generate_key(password):
    return base64.urlsafe_b64encode(hashlib.sha256(password.encode()).digest())

def decrypt_message(encrypted_text, password):
    key = generate_key(password)
    cipher = Fernet(key)
    decrypted = cipher.decrypt(encrypted_text.encode())
    return decrypted.decode()