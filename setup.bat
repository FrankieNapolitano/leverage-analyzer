@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"
title Leverage Analyzer - setup
echo.
echo ==========================================
echo   Leverage Analyzer - one-time setup
echo ==========================================
echo.

REM ---------- 1. Find or install Python ----------
set "PY="
py -3 --version >nul 2>nul && set "PY=py -3"
if not defined PY python --version >nul 2>nul && set "PY=python"
if defined PY goto have_python

echo [1/5] Installing Python 3.12 (takes 1-3 minutes)...
winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PY="%LOCALAPPDATA%\Programs\Python\Python312\python.exe""
if not defined PY (
  echo.
  echo Python install didn't finish. Install it from https://www.python.org/downloads/
  echo ^(check "Add python.exe to PATH"^), then double-click setup.bat again.
  pause & exit /b 1
)
:have_python
echo [1/5] Python ready:
%PY% --version

REM ---------- 2. VS Code (optional, for reading/editing the code) ----------
where code >nul 2>nul
if errorlevel 1 (
  echo [2/5] Installing VS Code...
  winget install -e --id Microsoft.VisualStudioCode --scope user --accept-package-agreements --accept-source-agreements >nul
) else (
  echo [2/5] VS Code already installed.
)

REM ---------- 3. Private Python environment + libraries ----------
echo [3/5] Installing libraries (yfinance, pandas, plotly, streamlit, anthropic)...
if not exist ".venv\Scripts\python.exe" %PY% -m venv .venv
".venv\Scripts\python.exe" -m pip install --upgrade pip -q
".venv\Scripts\python.exe" -m pip install -r requirements.txt -q
if errorlevel 1 ( echo Library install failed - check your internet and run setup.bat again. & pause & exit /b 1 )

REM Skip Streamlit's first-run email question
if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit"
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
  > "%USERPROFILE%\.streamlit\credentials.toml" echo [general]
  >> "%USERPROFILE%\.streamlit\credentials.toml" echo email = ""
)

REM ---------- 4. Anthropic API key ----------
if defined ANTHROPIC_API_KEY (
  echo [4/5] Anthropic API key already saved.
  goto key_done
)
echo.
echo [4/5] Anthropic API key (for the Claude write-up)
echo       Get one at: https://console.anthropic.com/settings/keys
echo       Paste it below and press Enter. (Right-click pastes in this window.)
echo       Or just press Enter to skip - the dashboard still works without it.
set "KEY="
set /p "KEY=      Key: "
if "!KEY!"=="" goto key_done
setx ANTHROPIC_API_KEY "!KEY!" >nul
set "ANTHROPIC_API_KEY=!KEY!"
echo       Saved.
:key_done

REM ---------- 5. Desktop shortcut ----------
echo [5/5] Adding a "Leverage Analyzer" shortcut to your Desktop...
powershell -NoProfile -Command "$s=(New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop')+'\Leverage Analyzer.lnk');$s.TargetPath='%~dp0run.bat';$s.WorkingDirectory='%~dp0';$s.Save()" >nul 2>nul

echo.
echo Setup complete! Starting the dashboard in your browser...
echo (Next time, just double-click "Leverage Analyzer" on your Desktop.)
echo.
call "%~dp0run.bat"
