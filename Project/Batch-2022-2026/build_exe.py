import os
import shutil
import subprocess
import sys

def main():
    print("Building executable with PyInstaller...")

    # Clean old build artifacts
    for p in ["build", "dist"]:
        if os.path.exists(p):
            shutil.rmtree(p, ignore_errors=True)
    for f in os.listdir("."):
        if f.endswith(".spec"):
            try:
                os.remove(f)
            except OSError:
                pass

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--name", "cyber_threat_ai_ti",
        "--add-data", "templates;templates",
        "--add-data", "static;static",
        "--add-data", "models;models",
        "--add-data", "uploads;uploads",
        "--add-data", "database;database",
        "app.py"
    ]

    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)
    print("\nSUCCESS ✅")
    print("EXE is at: dist/cyber_threat_ai_ti.exe")

if __name__ == "__main__":
    main()