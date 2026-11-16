@echo off
cd /d "%~dp0"
title Leverage Analyzer - update Time Machine data
if not exist ".venv\Scripts\python.exe" (
  echo Run setup.bat first.
  pause
  exit /b 1
)
echo Updating the Time Machine data (about 5-15 minutes). Leave this window open.
".venv\Scripts\python.exe" -m pip install -r requirements.txt -q
".venv\Scripts\python.exe" timemachine\update_data.py
echo.
echo Done! Restart the dashboard (run.bat) to see the new data.
pause
