#!/bin/bash

# Script to run the ML training pipeline with virtual environment activation

set -e  # Exit on any error

echo "🚀 Starting ML Training Pipeline"
echo "=================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Creating one..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies if needed
echo "📦 Checking dependencies..."
pip install -r config/requirements.txt

# Run the training pipeline
echo "🤖 Starting model training..."
python src/model_training/model_training_15Jul_optimized1M.py "$@"

echo "✅ Training pipeline completed!" 