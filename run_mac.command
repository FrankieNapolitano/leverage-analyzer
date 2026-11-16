#!/bin/bash
cd "$(dirname "$0")"
[ -f ~/.zshrc ] && source ~/.zshrc 2>/dev/null
if [ ! -x .venv/bin/python ]; then echo "Run setup_mac.command first."; read -p "Press Enter..."; exit 1; fi
echo "Starting Leverage Analyzer... (close this window to stop it)"
.venv/bin/python -m streamlit run app.py --browser.gatherUsageStats false
