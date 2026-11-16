#!/bin/bash
cd "$(dirname "$0")"
echo "=========================================="
echo "  Leverage Analyzer - one-time setup (Mac)"
echo "=========================================="
if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >/dev/null 2>&1; then
  echo "Python 3 isn't ready yet."
  echo "1) Go to https://www.python.org/downloads/ and install the latest Python for macOS"
  echo "2) Then double-click setup_mac.command again"
  open "https://www.python.org/downloads/"
  read -p "Press Enter to close..."; exit 1
fi
echo "[1/3] Python: $(python3 --version)"
echo "[2/3] Installing libraries (2-5 minutes, screen may look frozen)..."
[ -x .venv/bin/python ] || python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip -q
if ! .venv/bin/python -m pip install -r requirements.txt -q; then
  echo "Install failed - check your internet and run this again."; read -p "Press Enter..."; exit 1
fi
mkdir -p ~/.streamlit
[ -f ~/.streamlit/credentials.toml ] || printf '[general]\nemail = ""\n' > ~/.streamlit/credentials.toml
if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo ""
  echo "[3/3] Anthropic API key for the Claude write-up (get one at https://console.anthropic.com/settings/keys)"
  read -p "      Paste it and press Enter (or just Enter to skip): " KEY
  if [ -n "$KEY" ]; then
    echo "export ANTHROPIC_API_KEY=\"$KEY\"" >> ~/.zshrc
    export ANTHROPIC_API_KEY="$KEY"
    echo "      Saved."
  fi
else
  echo "[3/3] API key already saved."
fi
echo ""
echo "Setup complete! Starting the dashboard..."
exec "$(dirname "$0")/run_mac.command"
