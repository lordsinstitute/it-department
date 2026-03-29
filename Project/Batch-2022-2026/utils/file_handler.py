import os
from werkzeug.utils import secure_filename  # FIXED: was "import secure_filename"
import hashlib
import time

class FileHandler:
    """Secure file upload and handling"""
    
    ALLOWED_EXTENSIONS = {'pdf', 'txt'}
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    
    def __init__(self, upload_folder):
        self.upload_folder = upload_folder
        os.makedirs(upload_folder, exist_ok=True)
    
    def is_allowed_file(self, filename):
        """Check if file extension is allowed"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in self.ALLOWED_EXTENSIONS
    
    def save_file(self, file):
        """Save uploaded file securely"""
        if not file or file.filename == '':
            raise ValueError("No file provided")
        
        if not self.is_allowed_file(file.filename):
            raise ValueError(f"File type not allowed. Allowed types: {', '.join(self.ALLOWED_EXTENSIONS)}")
        
        # Check file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > self.MAX_FILE_SIZE:
            raise ValueError(f"File size exceeds maximum allowed size of {self.MAX_FILE_SIZE / 1024 / 1024:.1f}MB")
        
        # Secure the filename
        filename = secure_filename(file.filename)
        unique_filename = f"{int(time.time())}_{filename}"
        filepath = os.path.join(self.upload_folder, unique_filename)
        file.save(filepath)
        
        return filepath, filename
    
    def calculate_file_hash(self, filepath):
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for byte_block in iter(lambda: f.read(4096), b''):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def cleanup_file(self, filepath):
        """Safely delete a file"""
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                return True
        except Exception:
            pass
        return False
