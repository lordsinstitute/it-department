@echo off
echo ========================================
echo  Ollama Model Setup
echo ========================================
echo.
echo This will download the AI model (about 4GB)
echo Make sure you have good internet connection!
echo.
pause
echo.

echo Pulling llama2 model...
ollama pull llama2

echo.
echo ========================================
echo  Setup Complete!
echo ========================================
echo.
echo Now you can:
echo 1. Run: start_ollama.bat (keep it open)
echo 2. Run: python app.py
echo 3. Use AI features in the dashboard
echo.
pause
