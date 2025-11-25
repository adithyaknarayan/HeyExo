#!/bin/bash

# Exit on error
set -e

echo "🦿 Installing Exoskeleton Interface Dependencies..."

# Check if python3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
else
    echo "✓ Virtual environment already exists."
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing requirements..."
pip install -r requirements.txt

echo "✅ Installation complete!"
echo "To run the application, use: ./run.sh"

