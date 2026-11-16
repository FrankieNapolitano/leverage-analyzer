@echo off
cd /d "%~dp0"
title Leverage Analyzer - backtest
if not exist ".venv\Scripts\python.exe" (
  echo Run setup.bat first.
  pause
  exit /b 1
)
echo Running the backtest. This takes about 20-45 minutes. Leave this window open.
echo You can close it and run again later - it picks up where it left off.
echo.
".venv\Scripts\python.exe" backtest.py
echo.
echo Done! Results saved to backtest_results.csv and backtest_summary.txt
pause
