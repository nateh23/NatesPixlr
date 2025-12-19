@echo off
cd /d "%~dp0"
python gui.py
if errorlevel 1 (
    echo.
    echo Failed to launch GUI. Make sure Python is installed.
    pause
)
