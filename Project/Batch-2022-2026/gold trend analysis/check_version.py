import sys

print(f"Python version: {sys.version}")
print(f"Version info: {sys.version_info}")

if sys.version_info < (3, 8):
    print("❌ ERROR: Python 3.8 or higher is required!")
    sys.exit(1)
elif sys.version_info >= (3, 8) and sys.version_info < (3, 11):
    print("✅ Python version is compatible!")
else:
    print("⚠️ WARNING: Python 3.11+ detected. Some packages may need adjustment.")

print("\nInstalling dependencies...")
print("Run: pip install -r requirements.txt")
