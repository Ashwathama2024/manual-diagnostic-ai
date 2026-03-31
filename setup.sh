#!/bin/bash
set -e

echo "=== ManualIQ Setup ==="

# 1. Create Virtual Environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# 2. Activate Virtual Environment
echo "Activating virtual environment..."
source .venv/bin/activate || source .venv/Scripts/activate

# 3. Install Dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Check/Install Ollama Models
echo "Checking Ollama status & models..."
if ! command -v ollama &> /dev/null; then
    echo "Warning: Ollama is not installed or not in PATH."
    echo "Please visit https://ollama.com/download to install it."
    exit 1
fi

echo "Pulling reasoning model (deepseek-r1:8b)..."
ollama pull deepseek-r1:8b

echo "Pulling embedding model (nomic-embed-text)..."
ollama pull nomic-embed-text

echo "=== Setup Complete! ==="
echo "Run 'python scripts/app.py' to start the Gradio UI,"
echo "or 'python scripts/server.py' to start the FastAPI server."
