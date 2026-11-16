#!/bin/bash
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then echo "Run setup_mac.command first."; read -p "Press Enter..."; exit 1; fi
echo "Updating the Time Machine data (about 5-15 minutes). Leave this window open."
.venv/bin/python -m pip install -r requirements.txt -q
.venv/bin/python timemachine/update_data.py
echo ""
echo "Done! Restart the dashboard (run_mac.command) to see the new data."
read -p "Press Enter to close..."
