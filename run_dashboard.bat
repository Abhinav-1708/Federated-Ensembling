@echo off
echo Building frontend...
python build_frontend.py
if %ERRORLEVEL% NEQ 0 (
    echo Failed to build frontend.
    pause
    exit /b 1
)

echo Starting dashboard...
python dashboard.py
pause 