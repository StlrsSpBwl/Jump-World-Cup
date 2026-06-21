#!/bin/zsh

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

if [[ ! -x ".venv/bin/streamlit" ]]; then
  echo "RBP Lab is not installed yet."
  echo
  echo "Run these commands once:"
  echo "  python3.11 -m venv .venv"
  echo "  .venv/bin/pip install -r requirements.txt"
  echo
  read "?Press Return to close."
  exit 1
fi

URL="http://localhost:8501"

(
  for attempt in {1..60}; do
    if curl --silent --fail "$URL/_stcore/health" >/dev/null 2>&1; then
      open "$URL"
      exit 0
    fi
    sleep 0.5
  done
) &

echo "Starting RBP Lab..."
echo "Your browser will open automatically."
echo "Close this Terminal window or press Control-C to stop the app."
echo

exec .venv/bin/streamlit run app.py --server.port 8501
