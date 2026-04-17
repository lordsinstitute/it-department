import os
import sys
import shutil
from PyInstaller import __main__ as pyi

def build_exe():
    """Build executable using PyInstaller"""
    
    # Clean up old build
    import time
    if os.path.exists('dist'):
    	shutil.rmtree('dist')
    if os.path.exists('build'):
        shutil.rmtree('build')
    if os.path.exists('pdf_malware_scanner.spec'):
        os.remove('pdf_malware_scanner.spec')
    
    # PyInstaller arguments
    pyi_args = [
        'app.py',
        '--name=pdf_malware_scanner',
        '--onefile',
        '--windowed',
        '--add-data=templates;templates',
        '--add-data=static;static',
        '--add-data=models;models',
        '--add-data=detector;detector',
        '--add-data=utils;utils',
	'--add-data=instance;instance',
	'--add-data=uploads;uploads',
        '--collect-all=flask',
        '--collect-all=flask_sqlalchemy',
	'--collect-all=jinja2',
        '--hidden-import=werkzeug.security',
        '--hidden-import=reportlab',
        '--hidden-import=pypdf',
        '--icon=NONE',
        '--distpath=./dist',
        '--workpath=./build',
        '--specpath=./'
    ]
    
    print("Building executable...")
    pyi.run(pyi_args)
    
    print("\n" + "="*60)
    print("BUILD COMPLETE!")
    print("="*60)
    print("\nExecutable location: dist/pdf_malware_scanner.exe")
    print("\nTo run:")
    print("1. Navigate to dist folder")
    print("2. Run: pdf_malware_scanner.exe")
    print("3. Open browser to http://localhost:5000")
    print("4. Login with admin / admin123")

if __name__ == '__main__':
    build_exe()
