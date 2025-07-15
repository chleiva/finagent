#!/bin/bash

# Interactive Launcher for Trading Model Pipeline
# Run this from the root of the repository

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

# Function to check if virtual environment exists
check_venv() {
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
}

# Function to show main menu
show_menu() {
    clear
    print_color "🤖 Trading Model Pipeline Launcher" $PURPLE
    print_color "==================================" $PURPLE
    echo
    print_color "What would you like to do?" $CYAN
    echo
    echo "1. 🚀 Train a new model"
    echo "2. 📊 Launch interactive dashboard"
    echo "3. 📋 View model training log"
    echo "4. 🔍 Interactive model comparison"
    echo "5. 📈 Start Jupyter notebook"
    echo "6. 🛠️  Setup/Install dependencies"
    echo "7. 📁 Browse available data files"
    echo "8. 🗂️  Model artifact management"
    echo "9. 📖 Quick Reference (show commands & exit)"
    echo "10. ❌ Exit"
    echo
}

# Function to train model
train_model() {
    print_color "🚀 Model Training Setup" $BLUE
    print_color "=======================" $BLUE
    echo
    
    # Check available data files
    data_files=($(ls data/raw/*.csv 2>/dev/null | head -10))
    
    if [ ${#data_files[@]} -eq 0 ]; then
        print_color "❌ No data files found in data/raw/" $RED
        print_color "Please add some CSV files to data/raw/ first" $YELLOW
        return
    fi
    
    print_color "Available data files:" $CYAN
    for i in "${!data_files[@]}"; do
        filename=$(basename "${data_files[$i]}")
        echo "  $((i+1)). $filename"
    done
    echo "  0. Enter custom path"
    echo
    
    # Get data file choice
    read -p "Select data file (1-${#data_files[@]}, or 0 for custom): " file_choice
    
    if [ "$file_choice" = "0" ]; then
        read -p "Enter custom file path: " training_file
    else
        if [ "$file_choice" -ge 1 ] && [ "$file_choice" -le ${#data_files[@]} ]; then
            training_file="${data_files[$((file_choice-1))]}"
        else
            print_color "❌ Invalid choice" $RED
            return
        fi
    fi
    
    # Get model description
    echo
    print_color "Model Description Examples:" $CYAN
    echo "  - 'AAPL 2024-12 with 22 features'"
    echo "  - 'High-impact features only'"
    echo "  - 'Experiment with reduced feature set'"
    echo "  - 'Testing new preprocessing pipeline'"
    echo
    read -p "Enter model description: " model_description
    
    # Ask for test file
    echo
    read -p "Do you want to use a test file? (y/n): " use_test
    test_file=""
    if [[ $use_test =~ ^[Yy]$ ]]; then
        echo
        print_color "Available test files:" $CYAN
        for i in "${!data_files[@]}"; do
            filename=$(basename "${data_files[$i]}")
            echo "  $((i+1)). $filename"
        done
        echo "  0. Enter custom path"
        echo
        read -p "Select test file (1-${#data_files[@]}, or 0 for custom): " test_choice
        
        if [ "$test_choice" = "0" ]; then
            read -p "Enter custom test file path: " test_file
        else
            if [ "$test_choice" -ge 1 ] && [ "$test_choice" -le ${#data_files[@]} ]; then
                test_file="${data_files[$((test_choice-1))]}"
            fi
        fi
    fi
    
    # Ask for fast mode
    echo
    read -p "Use fast mode for quick testing? (y/n): " fast_mode
    
    # Build command
    cmd="python src/model_training/model_training_15Jul_optimized1M.py \"$training_file\" --description \"$model_description\""
    
    if [ ! -z "$test_file" ]; then
        cmd="$cmd --test-csv \"$test_file\""
    fi
    
    if [[ $fast_mode =~ ^[Yy]$ ]]; then
        cmd="$cmd --fast"
    fi
    
    echo
    print_color "🚀 Starting training with command:" $GREEN
    echo "$cmd"
    echo
    print_color "Press Enter to start, or Ctrl+C to cancel..." $YELLOW
    read
    
    # Execute training
    eval $cmd
}

# Function to launch dashboard
launch_dashboard() {
    print_color "📊 Launching Interactive Dashboard" $BLUE
    print_color "=================================" $BLUE
    echo
    print_color "The dashboard will open in your browser at: http://localhost:8501" $CYAN
    print_color "Press Ctrl+C to stop the dashboard" $YELLOW
    echo
    
    streamlit run src/evaluation/streamlit_model_dashboard.py
}

# Function to view log
view_log() {
    print_color "📋 Model Training Log" $BLUE
    print_color "====================" $BLUE
    echo
    
    if [ ! -f "model_training_log.csv" ]; then
        print_color "❌ No training log found. Train a model first!" $RED
        return
    fi
    
    print_color "Recent training runs:" $CYAN
    echo
    
    # Show recent runs in a nice format
    python -c "
import pandas as pd
try:
    df = pd.read_csv('model_training_log.csv')
    if len(df) > 0:
        # Sort by date, most recent first
        df['Date_Start'] = pd.to_datetime(df['Date_Start'])
        df = df.sort_values('Date_Start', ascending=False)
        
        # Show last 5 runs
        recent = df.head(5)
        for i, row in recent.iterrows():
            status = '✅' if row['Status'] == 'Success' else '❌'
            date = row['Date_Start'].strftime('%Y-%m-%d %H:%M')
            desc = row['Model_Description'][:50] + '...' if len(str(row['Model_Description'])) > 50 else row['Model_Description']
            auc = f\"{row['Test_AUC']:.4f}\" if pd.notna(row['Test_AUC']) else 'N/A'
            print(f\"{status} {date} | {desc} | AUC: {auc}\")
    else:
        print('No training runs found.')
except Exception as e:
    print(f'Error reading log: {e}')
"
    
    echo
    read -p "Press Enter to continue..."
}

# Function to interactive comparison
interactive_comparison() {
    print_color "🔍 Interactive Model Comparison" $BLUE
    print_color "==============================" $BLUE
    echo
    
    if [ ! -f "model_training_log.csv" ]; then
        print_color "❌ No training log found. Train a model first!" $RED
        return
    fi
    
    print_color "Launching interactive model comparison tool..." $CYAN
    python src/evaluation/model_comparison_tool.py
}

# Function to start Jupyter
start_jupyter() {
    print_color "📈 Starting Jupyter Notebook" $BLUE
    print_color "===========================" $BLUE
    echo
    print_color "Jupyter will open in your browser" $CYAN
    print_color "Press Ctrl+C to stop Jupyter" $YELLOW
    echo
    
    jupyter notebook --notebook-dir=notebooks/
}

# Function to setup dependencies
setup_dependencies() {
    print_color "🛠️  Setting up Dependencies" $BLUE
    print_color "=========================" $BLUE
    echo
    
    print_color "Creating virtual environment..." $CYAN
    python3 -m venv venv
    
    print_color "Activating virtual environment..." $CYAN
    source venv/bin/activate
    
    print_color "Installing dependencies..." $CYAN
    pip install -r config/requirements.txt
    
    print_color "✅ Setup complete!" $GREEN
    echo
    read -p "Press Enter to continue..."
}

# Function to browse data files
browse_data() {
    print_color "📁 Available Data Files" $BLUE
    print_color "======================" $BLUE
    echo
    
    # Check different data directories
    directories=("data/raw" "data/processed" "data/external")
    
    for dir in "${directories[@]}"; do
        if [ -d "$dir" ]; then
            print_color "�� $dir:" $CYAN
            files=($(ls "$dir"/*.csv 2>/dev/null | head -5))
            if [ ${#files[@]} -gt 0 ]; then
                for file in "${files[@]}"; do
                    filename=$(basename "$file")
                    size=$(du -h "$file" | cut -f1)
                    echo "  📄 $filename ($size)"
                done
                if [ $(ls "$dir"/*.csv 2>/dev/null | wc -l) -gt 5 ]; then
                    echo "  ... and $(($(ls "$dir"/*.csv 2>/dev/null | wc -l) - 5)) more files"
                fi
            else
                echo "  (no CSV files)"
            fi
            echo
        fi
    done
    
    read -p "Press Enter to continue..."
}

# Main loop
while true; do
    show_menu
    read -p "Enter your choice (1-10): " choice
    
    case $choice in
        1)
            check_venv
            train_model
            ;;
        2)
            check_venv
            launch_dashboard
            ;;
        3)
            check_venv
            view_log
            ;;
        4)
            check_venv
            interactive_comparison
            ;;
        5)
            check_venv
            start_jupyter
            ;;
        6)
            setup_dependencies
            ;;
        7)
            browse_data
            ;;
        8)
            model_management_menu
            ;;
        9)
            show_quick_reference
            ;;
        10)
            print_color "👋 Goodbye!" $GREEN
            exit 0
            ;;
        *)
            print_color "❌ Invalid choice. Please try again." $RED
            sleep 2
            ;;
    esac
    
    echo
    read -p "Press Enter to return to main menu..."
done

# Function for model management menu
model_management_menu() {
    while true; do
        print_color "🗂️  Model Artifact Management" $BLUE
        print_color "=============================" $BLUE
        echo
        echo "1. 📋 List all saved models"
        echo "2. 🔍 Find models by description"
        echo "3. 📊 Show model details"
        echo "4. 🗑️  Clean old models"
        echo "5. 📁 Browse artifact directories"
        echo "6. ↩️  Back to main menu"
        echo
        read -p "Enter your choice (1-6): " model_choice
        
        case $model_choice in
            1)
                check_venv
                python src/utils/model_manager.py list
                ;;
            2)
                check_venv
                echo "Enter keywords to search for (e.g., AAPL 2024):"
                read -p "Keywords: " keywords
                if [ ! -z "$keywords" ]; then
                    python src/utils/model_manager.py find --keywords $keywords
                fi
                ;;
            3)
                check_venv
                echo "Enter model ID to show details:"
                read -p "Model ID: " model_id
                if [ ! -z "$model_id" ]; then
                    python src/utils/model_manager.py show --model-id "$model_id"
                fi
                ;;
            4)
                check_venv
                echo "Enter number of days old for cleaning (default 30):"
                read -p "Days: " days
                if [ -z "$days" ]; then
                    days=30
                fi
                python src/utils/model_manager.py clean --days $days
                ;;
            5)
                if [ -d "model_artifacts" ]; then
                    print_color "📁 Model Artifacts Directory Structure:" $CYAN
                    tree model_artifacts -L 2 2>/dev/null || ls -la model_artifacts/
                else
                    print_color "❌ No model artifacts directory found." $RED
                fi
                ;;
            6)
                return
                ;;
            *)
                print_color "❌ Invalid choice. Please try again." $RED
                sleep 2
                ;;
        esac
        
        echo
        read -p "Press Enter to continue..."
    done
}

# Function to show quick reference
show_quick_reference() {
    clear
    print_color "📖 QUICK REFERENCE - Available Commands" $PURPLE
    print_color "=======================================" $PURPLE
    echo
    
    print_color "🚀 MODEL TRAINING:" $BLUE
    echo "  # Train with specific data file"
    echo "  python src/model_training/model_training_15Jul_optimized1M.py data/raw/monthly_AAPL_2025-01.csv --description \"AAPL January 2025\" --fast"
    echo
    echo "  # Train with test data"
    echo "  python src/model_training/model_training_15Jul_optimized1M.py data/raw/monthly_AAPL_2025-01.csv --test-csv data/raw/monthly_AAPL_2025-02.csv --description \"AAPL with test data\""
    echo
    
    print_color "📊 DASHBOARD & MONITORING:" $BLUE
    echo "  # Launch interactive dashboard"
    echo "  streamlit run src/evaluation/streamlit_model_dashboard.py"
    echo
    echo "  # View training log"
    echo "  python src/utils/view_model_log.py"
    echo
    echo "  # Interactive model comparison"
    echo "  python src/evaluation/model_comparison_tool.py"
    echo
    
    print_color "🗂️ MODEL MANAGEMENT:" $BLUE
    echo "  # List all saved models"
    echo "  python src/utils/model_manager.py list"
    echo
    echo "  # Find models by description"
    echo "  python src/utils/model_manager.py find --keywords AAPL 2024"
    echo
    echo "  # Show model details"
    echo "  python src/utils/model_manager.py show --model-id 20241215_143022_AAPL_January_2025"
    echo
    echo "  # Clean old models (older than 30 days)"
    echo "  python src/utils/model_manager.py clean --days 30"
    echo
    
    print_color "📈 DEVELOPMENT:" $BLUE
    echo "  # Start Jupyter notebook"
    echo "  jupyter notebook --notebook-dir=notebooks/"
    echo
    echo "  # Activate virtual environment"
    echo "  source venv/bin/activate"
    echo
    echo "  # Install dependencies"
    echo "  pip install -r config/requirements.txt"
    echo
    
    print_color "🛠️ UTILITIES:" $BLUE
    echo "  # Data processing"
    echo "  python src/data_processing/concatenate_all.py --symbols AAPL,MSFT --output training_data.csv"
    echo
    echo "  # Model evaluation"
    echo "  python src/evaluation/evaluate.py"
    echo
    echo "  # Makefile commands"
    echo "  make train          # Train model"
    echo "  make dashboard      # Launch dashboard"
    echo "  make setup          # Setup environment"
    echo
    
    print_color "📁 IMPORTANT FILES:" $BLUE
    echo "  📊 model_training_log.csv          # Training history"
    echo "  📋 model_artifacts/model_index.csv # Model index"
    echo "  📁 model_artifacts/                # All saved models"
    echo "  📁 data/raw/                       # Raw data files"
    echo "  📁 evaluation_results/             # Evaluation outputs"
    echo
    
    print_color "💡 TIPS:" $GREEN
    echo "  • Always activate virtual environment: source venv/bin/activate"
    echo "  • Use --fast flag for quick testing"
    echo "  • Use descriptive model descriptions for easy finding"
    echo "  • Check dashboard for real-time training monitoring"
    echo "  • Use model manager to find and organize your models"
    echo
    
    print_color "🎯 NEXT STEPS:" $YELLOW
    echo "  1. Activate virtual environment: source venv/bin/activate"
    echo "  2. Train a model: python src/model_training/model_training_15Jul_optimized1M.py data/raw/monthly_AAPL_2025-01.csv --description \"My first model\""
    echo "  3. View results: streamlit run src/evaluation/streamlit_model_dashboard.py"
    echo "  4. Find your model: python src/utils/model_manager.py list"
    echo
    
    print_color "✅ Quick reference complete! You can now run commands manually." $GREEN
    echo
    print_color "👋 Exiting launcher..." $CYAN
    exit 0
}
