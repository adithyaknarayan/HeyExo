#!/bin/bash

# Exit on error
set -e

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run ./install.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Run the Streamlit app
echo "🚀 Starting Exoskeleton Interface..."
streamlit run user_interface.py

