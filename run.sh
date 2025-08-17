#!/usr/bin/env bash
set -e

# Always run relative to project root
cd "$(dirname "$0")"

# Create venv if missing
if [ ! -d ".venv" ]; then
    echo "🔹 No virtual environment found. Creating one..."
    python -m venv .venv
    echo "🔹 Activating virtual environment"
    source .venv/bin/activate
    echo "🔹 Installing requirements..."
else
    echo "🔹 Activating existing virtual environment"
    source .venv/bin/activate
    echo "🔹 Ensuring requirements are satisfied..."
fi
pip install --upgrade pip
pip install -r requirements.txt

# Run Streamlit app
echo "🚀 Starting Streamlit..."
python -m streamlit run /home/kunalchand/Desktop/Projects/Others/rag-app/src/app/streamlit_app.py