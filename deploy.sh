#!/bin/bash

# Exit on error
set -e

echo "📦 Preparing for deployment..."

# Check for required files
REQUIRED_FILES=("user_interface.py" "dialogue_system.py" "requirements.txt" "const_task.py" "const_assistance.py")
MISSING_FILES=0

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Missing required file: $file"
        MISSING_FILES=1
    fi
done

if [ $MISSING_FILES -eq 1 ]; then
    echo "⚠️  Deployment check failed. Please ensure all files are present."
    exit 1
fi

# Check for data directory
if [ ! -d "data" ]; then
    echo "⚠️  Warning: 'data' directory not found. Creating it..."
    mkdir data
fi

echo "✅ Deployment checks passed!"
echo "🚀 Starting deployment process..."

# Ensure scripts are executable
chmod +x install.sh run.sh

# Run installation
echo "📝 Running installation script..."
./install.sh

# Run application
echo "🏃 Starting application..."
./run.sh
