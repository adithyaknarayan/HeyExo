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
echo "To deploy this application:"
echo "1. Ensure python 3.9+ is installed on the target server"
echo "2. Copy all files to the server"
echo "3. Run ./install.sh on the server"
echo "4. Run ./run.sh to start the application"
echo ""
echo "For Docker deployment (optional):"
echo "A Dockerfile can be created to containerize this application."

