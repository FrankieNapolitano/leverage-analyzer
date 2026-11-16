#!/bin/bash
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then echo "Run setup_mac.command first."; read -p "Press Enter..."; exit 1; fi
echo "Running the backtest. About 20-45 minutes. Leave this window open."
echo "You can close it and run again later - it picks up where it left off."
echo ""
.venv/bin/python backtest.py
echo ""
echo "Done! Results saved to backtest_results.csv and backtest_summary.txt"
read -p "Press Enter to close..."
