@echo off
REM Traffic Signal Control Dashboard Startup Script
REM This script starts the Flask web server and opens the dashboard

echo.
echo ============================================================
echo   Traffic Signal Control Comparison Dashboard
echo ============================================================
echo.
echo Starting Flask server on http://127.0.0.1:5000
echo.

REM Check if Python virtual environment exists
if not exist "venv\Scripts\python.exe" (
    echo Error: Virtual environment not found!
    echo Please ensure venv is set up in the project directory
    echo.
    echo To create virtual environment, run:
    echo   python -m venv venv
    echo   venv\Scripts\pip install flask flask-cors torch torchvision pandas numpy matplotlib
    echo.
    pause
    exit /b 1
)

REM Check if Flask is installed
venv\Scripts\python.exe -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo Error: Flask not installed!
    echo Installing Flask and dependencies...
    venv\Scripts\pip install flask flask-cors torch torchvision pandas numpy matplotlib
)

echo.
echo Starting application...
echo Close this window to stop the server
echo.

REM Start Flask and open browser
start http://127.0.0.1:5000
venv\Scripts\python.exe app.py

echo.
echo Server stopped.
pause
