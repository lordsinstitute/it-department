import cv2
import base64
import hashlib
from cryptography.fernet import Fernet

def generate_key_from_password(password):
    key = hashlib.sha256(password.encode()).digest()
    return base64.urlsafe_b64encode(key)

def scan_and_decrypt(password):
    detector = cv2.QRCodeDetector()
    img = cv2.imread("secure_qr.png")

    data, bbox, _ = detector.detectAndDecode(img)

    if data:
        try:
            key = generate_key_from_password(password)
            cipher = Fernet(key)
            decrypted = cipher.decrypt(data.encode())
            return decrypted.decode()
        except:
            return "Incorrect Password!"
    else:
        return "No QR code detected!"