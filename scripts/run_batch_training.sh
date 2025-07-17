#!/bin/bash

# Batch Training Launcher for Top 10 NASDAQ Shares
# ================================================

set -e  # Exit on any error

# Colors for better UI
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored text
print_color() {
    echo -e "${2}${1}${NC}"
}

# Check if virtual environment exists and activate it
if [ ! -d "venv" ]; then
    print_color "❌ Virtual environment not found. Creating one..." $YELLOW
    python3 -m venv venv
    print_color "✅ Virtual environment created" $GREEN
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies if needed
print_color "📦 Checking dependencies..." $CYAN
pip install -r config/requirements.txt > /dev/null 2>&1

print_color "🤖 BATCH TRAINING - TOP 10 NASDAQ SHARES" $PURPLE
print_color "=========================================" $PURPLE
echo
print_color "This will train individual models for:" $CYAN
echo "  • NVDA (Nvidia)"
echo "  • MSFT (Microsoft)"
echo "  • AAPL (Apple)"
echo "  • AMZN (Amazon)"
echo "  • GOOGL (Alphabet)"
echo "  • META (Meta Platforms)"
echo "  • AVGO (Broadcom)"
echo "  • TSLA (Tesla)"
echo "  • NFLX (Netflix)"
echo "  • COST (Costco)"
echo

print_color "⚠️  IMPORTANT NOTES:" $YELLOW
echo "  • Each model will take 5-15 minutes to train"
echo "  • Total time: ~2-3 hours for all 10 models"
echo "  • Models will be saved in model_artifacts/"
echo "  • You can skip individual models during training"
echo "  • Use Ctrl+C to stop at any time"
echo

read -p "Are you sure you want to start batch training? (y/n): " confirm

if [[ $confirm =~ ^[Yy]$ ]]; then
    print_color "🚀 Starting batch training..." $GREEN
    echo
    
    # Run the batch training script
    python src/scripts/batch_train_top10.py --no-interactive
    
    print_color "🎉 Batch training completed!" $GREEN
    echo
    print_color "📁 Your models are now available in model_artifacts/" $CYAN
    print_color "🔍 Use ModelWrapper to access your trained models:" $CYAN
    echo
    echo "  from src.utils.model_wrapper import ModelWrapper"
    echo "  model = ModelWrapper('AAPL')  # Load AAPL model"
    echo "  predictions = model.predict(your_data)"
    echo
else
    print_color "❌ Batch training cancelled." $RED
fi 