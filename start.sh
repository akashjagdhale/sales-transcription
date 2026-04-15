#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

UV="$HOME/.local/bin/uv"
if [[ ! -x "$UV" ]]; then
    if command -v uv &>/dev/null; then
        UV="$(command -v uv)"
    else
        echo "ERROR: uv not found at $HOME/.local/bin/uv"
        echo "       Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
fi

echo ""
echo "🎙️  Sales Transcription Cloud — Web UI"
echo "========================================"
echo "   Open in your browser: http://localhost:5001"
echo "   Press Ctrl+C to stop"
echo ""

"$UV" run --with flask --with requests --with sarvamai --with anthropic --with openai --with google-generativeai python3 app.py
