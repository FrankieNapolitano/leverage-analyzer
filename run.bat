@echo off
cd /d "%~dp0"
title Leverage Analyzer
if not exist ".venv\Scripts\python.exe" (
  echo Run setup.bat first.
  pause
  exit /b 1
)
echo Starting Leverage Analyzer... (close this window to stop it)
".venv\Scripts\python.exe" -m streamlit run app.py --browser.gatherUsageStats false
pause
