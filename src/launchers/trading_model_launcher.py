#!/usr/bin/env python3
"""
Trading Model Launcher - Comprehensive ML Pipeline Orchestrator
===============================================================

This launcher combines dataset preparation and model training into a single,
user-friendly interface with both command-line and interactive modes.

Features:
- Smart dataset concatenation with stock/time filtering
- Interactive stock selection with search and categories
- Flexible time period selection
- Model training with comprehensive logging
- Progress tracking and validation
- Error handling and recovery

Usage:
    # Interactive mode
    python trading_model_launcher.py
    
    # Command-line mode
    python trading_model_launcher.py --stocks AAPL,MSFT,GOOGL --year 2025 --month 01 --description "Tech stocks Q1 2025"
    
    # Quick mode (all stocks, all time)
    python trading_model_launcher.py --quick --description "Full dataset training"
"""

import os
import sys
import argparse
import subprocess
import time
import datetime
import glob
import uuid
from pathlib import Path
import pandas as pd

# Available stocks (auto-detected)
AVAILABLE_STOCKS = [
    'AAPL', 'ADBE', 'AMAT', 'AMD', 'AMZN', 'ANGO', 'ARM', 'ASML', 'AVGO', 'AZN',
    'BIGC', 'BIIB', 'BKNG', 'CMCSA', 'COIN', 'COST', 'CRWD', 'CSCO', 'GOOGL', 'GRAB',
    'INTC', 'INTU', 'KLAC', 'MELI', 'META', 'METC', 'MRVL', 'MSFT', 'MSTR', 'NET',
    'NFLX', 'NVDA', 'PEP', 'PLTR', 'PUBM', 'PYPL', 'QCOM', 'ROKU', 'SNOW', 'SOFI',
    'SPY', 'TMUS', 'TSLA', 'TXN', 'UBER', 'VRTX', 'WDAY', 'ZS'
]

# Stock categories for easier selection
STOCK_CATEGORIES = {
    'Tech Giants': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA'],
    'Semiconductors': ['AMD', 'INTC', 'NVDA', 'ASML', 'KLAC', 'MRVL', 'QCOM'],
    'Software': ['ADBE', 'CRM', 'SNOW', 'NET', 'PLTR', 'ZS', 'WDAY'],
    'Fintech': ['PYPL', 'COIN', 'SOFI', 'PUBM'],
    'Entertainment': ['NFLX', 'ROKU', 'MELI'],
    'Transportation': ['UBER', 'TSLA'],
    'Biotech': ['BIIB', 'VRTX', 'ANGO'],
    'All': AVAILABLE_STOCKS
}

def print_banner():
    """Print the application banner"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🚀 TRADING MODEL LAUNCHER 🚀                              ║
║                                                                              ║
║  Comprehensive ML Pipeline: Dataset Prep → Model Training → Evaluation      ║
║                                                                              ║
║  Features:                                                                   ║
║  • Smart dataset concatenation with stock/time filtering                    ║
║  • Interactive stock selection with categories                              ║
║  • Flexible time period selection                                           ║
║  • Model training with comprehensive logging                                ║
║  • Progress tracking and validation                                         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

def get_available_years_months():
    """Get available years and months from existing files"""
    files = glob.glob("monthly_*.csv")
    years_months = set()
    
    for file in files:
        # Extract year-month from filename like "monthly_AAPL_2025-07.csv"
        parts = file.split('_')
        if len(parts) >= 3:
            date_part = parts[2].replace('.csv', '')
            if '-' in date_part:
                years_months.add(date_part)
    
    years = sorted(list(set([ym.split('-')[0] for ym in years_months])))
    months = sorted(list(set([ym.split('-')[1] for ym in years_months])))
    
    return years, months

def interactive_stock_selection():
    """Interactive stock selection with categories and search"""
    print("\n📊 STOCK SELECTION")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("1. Select by category")
        print("2. Search stocks")
        print("3. Select all stocks")
        print("4. Manual selection")
        print("5. View available stocks")
        
        choice = input("\nChoose option (1-5): ").strip()
        
        if choice == '1':
            return select_by_category()
        elif choice == '2':
            return search_stocks()
        elif choice == '3':
            return AVAILABLE_STOCKS
        elif choice == '4':
            return manual_stock_selection()
        elif choice == '5':
            show_available_stocks()
        else:
            print("❌ Invalid choice. Please try again.")

def select_by_category():
    """Select stocks by category"""
    print("\n📂 STOCK CATEGORIES")
    print("-" * 30)
    
    categories = list(STOCK_CATEGORIES.keys())
    for i, category in enumerate(categories, 1):
        stocks = STOCK_CATEGORIES[category]
        print(f"{i}. {category} ({len(stocks)} stocks): {', '.join(stocks[:5])}{'...' if len(stocks) > 5 else ''}")
    
    while True:
        try:
            choice = input(f"\nSelect category (1-{len(categories)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(categories):
                selected_category = categories[idx]
                selected_stocks = STOCK_CATEGORIES[selected_category]
                print(f"\n✅ Selected {len(selected_stocks)} stocks from '{selected_category}': {', '.join(selected_stocks)}")
                return selected_stocks
            else:
                print("❌ Invalid category number. Please try again.")
        except ValueError:
            print("❌ Please enter a valid number.")

def search_stocks():
    """Search stocks by name"""
    print("\n🔍 STOCK SEARCH")
    print("-" * 20)
    
    while True:
        search_term = input("Enter stock symbol or partial name (e.g., 'AAPL' or 'APP'): ").strip().upper()
        if not search_term:
            print("❌ Search term cannot be empty.")
            continue
        
        matches = [stock for stock in AVAILABLE_STOCKS if search_term in stock]
        
        if not matches:
            print(f"❌ No stocks found matching '{search_term}'")
            continue
        
        print(f"\nFound {len(matches)} matches:")
        for i, stock in enumerate(matches, 1):
            print(f"  {i}. {stock}")
        
        if len(matches) == 1:
            print(f"\n✅ Selected: {matches[0]}")
            return matches
        
        while True:
            choice = input(f"\nSelect stocks (1-{len(matches)}, or 'all' for all matches): ").strip()
            if choice.lower() == 'all':
                print(f"\n✅ Selected all {len(matches)} stocks: {', '.join(matches)}")
                return matches
            else:
                try:
                    indices = [int(x.strip()) - 1 for x in choice.split(',')]
                    selected = [matches[i] for i in indices if 0 <= i < len(matches)]
                    if selected:
                        print(f"\n✅ Selected {len(selected)} stocks: {', '.join(selected)}")
                        return selected
                    else:
                        print("❌ No valid selections. Please try again.")
                except ValueError:
                    print("❌ Invalid input. Please enter numbers separated by commas.")

def manual_stock_selection():
    """Manual stock selection"""
    print("\n✏️ MANUAL STOCK SELECTION")
    print("-" * 30)
    print("Enter stock symbols separated by commas (e.g., AAPL,MSFT,GOOGL)")
    print("Available stocks:", ', '.join(AVAILABLE_STOCKS))
    
    while True:
        input_stocks = input("\nEnter stock symbols: ").strip().upper()
        if not input_stocks:
            print("❌ No stocks entered.")
            continue
        
        selected_stocks = [stock.strip() for stock in input_stocks.split(',')]
        valid_stocks = [stock for stock in selected_stocks if stock in AVAILABLE_STOCKS]
        invalid_stocks = [stock for stock in selected_stocks if stock not in AVAILABLE_STOCKS]
        
        if invalid_stocks:
            print(f"❌ Invalid stocks: {', '.join(invalid_stocks)}")
            print("Available stocks:", ', '.join(AVAILABLE_STOCKS))
            continue
        
        if not valid_stocks:
            print("❌ No valid stocks selected.")
            continue
        
        print(f"\n✅ Selected {len(valid_stocks)} stocks: {', '.join(valid_stocks)}")
        return valid_stocks

def show_available_stocks():
    """Show all available stocks"""
    print("\n📋 AVAILABLE STOCKS")
    print("-" * 20)
    print(f"Total: {len(AVAILABLE_STOCKS)} stocks")
    print()
    
    # Group by category for display
    for category, stocks in STOCK_CATEGORIES.items():
        if category != 'All':
            print(f"{category}: {', '.join(stocks)}")
            print()

def interactive_time_selection():
    """Interactive time period selection"""
    print("\n📅 TIME PERIOD SELECTION")
    print("=" * 40)
    
    years, months = get_available_years_months()
    
    print(f"Available years: {', '.join(years)}")
    print(f"Available months: {', '.join(months)}")
    
    # Year selection
    while True:
        year_choice = input(f"\nSelect year (or 'all' for all years): ").strip()
        if year_choice.lower() == 'all':
            selected_year = None
            break
        elif year_choice in years:
            selected_year = year_choice
            break
        else:
            print(f"❌ Invalid year. Available: {', '.join(years)}")
    
    # Month selection
    while True:
        month_choice = input(f"Select month (or 'all' for all months): ").strip()
        if month_choice.lower() == 'all':
            selected_month = None
            break
        elif month_choice in months:
            selected_month = month_choice
            break
        else:
            print(f"❌ Invalid month. Available: {', '.join(months)}")
    
    return selected_year, selected_month

def get_model_description():
    """Get model description from user"""
    print("\n📝 MODEL DESCRIPTION")
    print("-" * 30)
    print("Please provide a description for this model run to help identify it later.")
    print("Examples:")
    print("  - 'Tech stocks Q1 2025'")
    print("  - 'AAPL and MSFT 2024-2025'")
    print("  - 'All stocks comprehensive training'")
    print("  - 'Semiconductor sector analysis'")
    print()
    
    while True:
        description = input("Model description: ").strip()
        if description:
            return description
        else:
            print("❌ Description cannot be empty. Please try again.")

def validate_dataset_parameters(stocks, year, month):
    """Validate that the dataset parameters will produce files"""
    if not stocks:
        print("❌ No stocks selected.")
        return False
    
    # Check if files exist for the given parameters in data/raw/
    files_found = []
    data_dir = "data/raw"
    
    for stock in stocks:
        if year and month:
            pattern = os.path.join(data_dir, f"monthly_{stock}_{year}-{month}.csv")
        elif year:
            pattern = os.path.join(data_dir, f"monthly_{stock}_{year}-*.csv")
        elif month:
            pattern = os.path.join(data_dir, f"monthly_{stock}_*-{month}.csv")
        else:
            pattern = os.path.join(data_dir, f"monthly_{stock}_*.csv")
        
        matching_files = glob.glob(pattern)
        files_found.extend(matching_files)
    
    if not files_found:
        print("❌ No files found for the selected parameters.")
        print(f"Stocks: {', '.join(stocks)}")
        print(f"Year: {year or 'all'}")
        print(f"Month: {month or 'all'}")
        print(f"Directory: {data_dir}/")
        return False
    
    print(f"✅ Found {len(files_found)} files for the selected parameters")
    return True

def run_concatenation(stocks, year, month, output_file):
    """Run the concatenation script"""
    print(f"\n🔄 PREPARING DATASET")
    print("=" * 40)
    
    # Build command
    cmd = ["python", "src/data_processing/concatenate_all.py", "--output", output_file]
    
    if stocks and len(stocks) == 1:
        # Single stock
        cmd.extend(["--symbol", stocks[0]])
    elif stocks:
        # Multiple stocks
        cmd.extend(["--symbols", ",".join(stocks)])
    
    if year:
        cmd.extend(["--year", year])
    if month:
        cmd.extend(["--month", month])
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("✅ Dataset preparation completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Dataset preparation failed: {e}")
        return False

def run_model_training(input_file, test_csv, description, fast_mode):
    """Run the model training script"""
    print(f"\n🤖 TRAINING MODEL")
    print("=" * 40)
    
    # Build command
    cmd = ["python", "src/model_training/model_training_15Jul_optimized1M.py", input_file]
    
    if fast_mode:
        cmd.append("--fast")
    if test_csv:
        cmd.extend(["--test-csv", test_csv])
    if description:
        cmd.extend(["--description", description])
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("✅ Model training completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Model training failed: {e}")
        return False

def interactive_mode():
    """Run in interactive mode"""
    print_banner()
    
    print("🎯 INTERACTIVE MODE")
    print("=" * 30)
    print("This mode will guide you through the entire process step by step.")
    print()
    
    # Step 1: Stock selection
    stocks = interactive_stock_selection()
    
    # Step 2: Time selection
    year, month = interactive_time_selection()
    
    # Step 3: Validation
    if not validate_dataset_parameters(stocks, year, month):
        print("❌ Invalid parameters. Exiting.")
        return False
    
    # Step 4: Model description
    description = get_model_description()
    
    # Step 5: Training options
    print("\n⚙️ TRAINING OPTIONS")
    print("-" * 20)
    fast_mode = input("Use fast mode for quicker training? (y/n): ").strip().lower() == 'y'
    
    test_csv = input("Test CSV file path (optional, press Enter to skip): ").strip()
    if not test_csv:
        test_csv = None
    
    # Step 6: Confirmation
    print("\n📋 CONFIRMATION")
    print("=" * 30)
    print(f"Stocks: {', '.join(stocks)}")
    print(f"Year: {year or 'all'}")
    print(f"Month: {month or 'all'}")
    print(f"Description: {description}")
    print(f"Fast mode: {fast_mode}")
    print(f"Test CSV: {test_csv or 'None'}")
    
    confirm = input("\nProceed with these settings? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ Cancelled by user.")
        return False
    
    # Step 7: Execute pipeline
    return execute_pipeline(stocks, year, month, description, fast_mode, test_csv, keep_temp=False)

def command_line_mode(args):
    """Run in command-line mode"""
    print_banner()
    
    print("💻 COMMAND-LINE MODE")
    print("=" * 30)
    
    # Parse stocks
    if args.stocks:
        stocks = [s.strip().upper() for s in args.stocks.split(',')]
        # Validate stocks
        invalid_stocks = [s for s in stocks if s not in AVAILABLE_STOCKS]
        if invalid_stocks:
            print(f"❌ Invalid stocks: {', '.join(invalid_stocks)}")
            return False
    else:
        stocks = AVAILABLE_STOCKS
    
    # Validate parameters
    if not validate_dataset_parameters(stocks, args.year, args.month):
        return False
    
    # Handle description
    description = args.description
    if not description:
        # Generate a default description based on parameters
        stocks_str = "_".join(stocks) if len(stocks) <= 3 else f"{len(stocks)}_stocks"
        year_str = f"_{args.year}" if args.year else ""
        month_str = f"_{args.month}" if args.month else ""
        description = f"{stocks_str}{year_str}{month_str}_auto_generated"
        print(f"📝 Using auto-generated description: {description}")
        print("💡 Tip: Use --description 'Your custom description' for better identification")
    
    # Execute pipeline
    return execute_pipeline(stocks, args.year, args.month, description, args.fast, args.test_csv, args.keep_temp)

def execute_pipeline(stocks, year, month, description, fast_mode, test_csv, keep_temp=False):
    """Execute the full pipeline"""
    print("\n🚀 EXECUTING PIPELINE")
    print("=" * 50)
    
    start_time = time.time()
    
    # Step 1: Create temporary directory for this run
    run_id = str(uuid.uuid4())[:8]  # First 8 characters of UUID
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create descriptive filename with run ID for uniqueness
    stocks_str = "_".join(stocks) if stocks else "all"
    year_str = f"_{year}" if year else ""
    month_str = f"_{month}" if month else ""
    
    # Create temporary directory
    temp_dir = f"temp_runs/{timestamp}_{run_id}"
    os.makedirs(temp_dir, exist_ok=True)
    
    output_file = os.path.join(temp_dir, f"training_dataset_{stocks_str}{year_str}{month_str}.csv")
    
    print(f"📁 Generated unique output file: {output_file}")
    print(f"🆔 Run ID: {run_id}")
    print(f"📂 Temporary directory: {temp_dir}")
    
    if not run_concatenation(stocks, year, month, output_file):
        print("❌ Pipeline failed at dataset preparation step.")
        return False
    
    # Step 2: Train model
    if not run_model_training(output_file, test_csv, description, fast_mode):
        print("❌ Pipeline failed at model training step.")
        return False
    
    duration = time.time() - start_time
    print(f"\n✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"⏱️ Total duration: {duration:.2f} seconds")
    print(f"📁 Output dataset: {output_file}")
    print(f"🆔 Run ID: {run_id}")
    
    # Clean up temporary directory unless keep-temp is specified
    if not keep_temp:
        try:
            import shutil
            shutil.rmtree(temp_dir)
            print(f"🗑️ Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            print(f"⚠️ Warning: Could not delete temporary directory {temp_dir}: {e}")
    else:
        print(f"💾 Kept temporary directory: {temp_dir}")
    
    return True

def cleanup_old_temp_files():
    """Clean up old temporary files and directories"""
    print("\n🧹 CLEANING UP OLD TEMPORARY FILES")
    print("=" * 40)
    
    # Clean up temp_runs directory
    if os.path.exists("temp_runs"):
        try:
            import shutil
            shutil.rmtree("temp_runs")
            print("✅ Cleaned up temp_runs/ directory")
        except Exception as e:
            print(f"⚠️ Warning: Could not clean up temp_runs/: {e}")
    else:
        print("✅ No temp_runs/ directory found")
    
    # Clean up old training dataset files in root
    old_files = []
    for file in os.listdir("."):
        if file.startswith("training_dataset_") and file.endswith(".csv"):
            old_files.append(file)
    
    if old_files:
        print(f"🗑️ Found {len(old_files)} old training dataset files in root:")
        for file in old_files:
            try:
                os.remove(file)
                print(f"  ✅ Removed: {file}")
            except Exception as e:
                print(f"  ⚠️ Could not remove {file}: {e}")
    else:
        print("✅ No old training dataset files found in root")
    
    print("🧹 Cleanup completed!")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Trading Model Launcher - Comprehensive ML Pipeline Orchestrator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python trading_model_launcher.py
  
  # Command-line mode - specific stocks and time
  python trading_model_launcher.py --stocks AAPL,MSFT,GOOGL --year 2025 --month 01 --description "Tech stocks Q1 2025"
  
  # Quick mode - all stocks, all time
  python trading_model_launcher.py --quick --description "Full dataset training"
  
  # Fast mode for testing
  python trading_model_launcher.py --stocks AAPL --year 2025 --fast --description "Quick test"
        """
    )
    
    parser.add_argument('--stocks', type=str, 
                       help='Comma-separated list of stock symbols (e.g., AAPL,MSFT,GOOGL)')
    parser.add_argument('--year', type=str, 
                       help='Year filter (e.g., 2025)')
    parser.add_argument('--month', type=str, 
                       help='Month filter (e.g., 01)')
    parser.add_argument('--description', type=str,
                       help='Model description/name')
    parser.add_argument('--fast', action='store_true',
                       help='Use fast mode for quicker training')
    parser.add_argument('--test-csv', type=str,
                       help='Test CSV file path for evaluation')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: all stocks, all time, fast training')
    parser.add_argument('--keep-temp', action='store_true',
                       help='Keep temporary concatenated dataset file (default: delete after training)')
    parser.add_argument('--cleanup-temp', action='store_true',
                       help='Clean up old temporary files before starting')
    
    args = parser.parse_args()
    
    # Handle quick mode
    if args.quick:
        args.stocks = None  # All stocks
        args.year = None    # All years
        args.month = None   # All months
        args.fast = True    # Fast mode
        if not args.description:
            args.description = "Quick training - all data"
    
    # Clean up old temporary files if requested
    if args.cleanup_temp:
        cleanup_old_temp_files()
    
    # Check if we have required files
    if not os.path.exists("src/data_processing/concatenate_all.py"):
        print("❌ Error: src/data_processing/concatenate_all.py not found")
        return 1
    
    if not os.path.exists("src/model_training/model_training_15Jul_optimized1M.py"):
        print("❌ Error: src/model_training/model_training_15Jul_optimized1M.py not found")
        return 1
    
    # Run appropriate mode
    if args.stocks or args.year or args.month or args.description or args.fast or args.test_csv or args.quick:
        # Command-line mode
        success = command_line_mode(args)
    else:
        # Interactive mode
        success = interactive_mode()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 