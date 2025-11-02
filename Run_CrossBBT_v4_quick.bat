@echo off
title Cross-BBT v4 (Quick Run)
echo ===============================
echo   Cross-BBT v4 - Quick Runner
echo ===============================
echo.

where python >nul 2>nul
if %errorlevel% neq 0 (
  echo Python not found. Please install Python 3.10+ and try again.
  echo https://www.python.org/downloads/
  pause
  exit /b 1
)

echo Installing/Updating required packages (this may take a minute)...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo.
echo Starting Streamlit app...
python -m streamlit run cross_bbt_app_v4.py

echo.
echo (Window will stay open so you can read any messages.)
pause
