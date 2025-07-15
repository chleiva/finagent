#!/bin/bash

# Script to launch the Streamlit Model Dashboard

set -e  # Exit on any error

echo "🎨 Launching Model Training Dashboard"
echo "====================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Creating one..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dashboard dependencies if needed
echo "📦 Checking dashboard dependencies..."
pip install streamlit plotly

# Launch the dashboard
echo "🚀 Starting Streamlit dashboard..."
echo "📊 Dashboard will open in your browser at: http://localhost:8501"
echo "🔄 Press Ctrl+C to stop the dashboard"
echo ""

streamlit run src/evaluation/streamlit_model_dashboard.py
