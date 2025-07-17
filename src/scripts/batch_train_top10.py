#!/usr/bin/env python3
"""
Batch Training Script for Top 10 NASDAQ Shares
==============================================

This script trains individual models for each of the top 10 NASDAQ shares
and organizes the artifacts for easy access.

Usage:
    python src/scripts/batch_train_top10.py
"""

import os
import sys
import subprocess
import time
import datetime
from pathlib import Path
import argparse

# Top 10 NASDAQ shares to train
TOP_10_NASDAQ = [
    "NVDA",   # Nvidia
    "MSFT",   # Microsoft
    "AAPL",   # Apple
    "AMZN",   # Amazon
    "GOOGL",  # Alphabet Inc. Class A
    "META",   # Meta Platforms
    "AVGO",   # Broadcom
    "TSLA",   # Tesla
    "NFLX",   # Netflix
    "COST"    # Costco
]

def check_data_availability(symbol):
    """Check if data is available for a given symbol"""
    data_dir = "data/raw"
    pattern = f"monthly_{symbol}_*.csv"
    
    import glob
    files = glob.glob(os.path.join(data_dir, pattern))
    return len(files) > 0, files

def train_single_model(symbol, description=None):
    """Train a single model for a given symbol"""
    if description is None:
        description = f"{symbol}-24-25"
    
    print(f"\n🚀 Training model for {symbol}")
    print("=" * 50)
    
    # Check data availability
    has_data, files = check_data_availability(symbol)
    if not has_data:
        print(f"❌ No data found for {symbol}")
        return False
    
    print(f"✅ Found {len(files)} data files for {symbol}")
    
    # Build command
    cmd = [
        "python", "src/launchers/trading_model_launcher.py",
        "--stocks", symbol,
        "--description", description
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        # Run the training
        result = subprocess.run(cmd, check=True)
        print(f"✅ Successfully trained model for {symbol}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to train model for {symbol}: {e}")
        return False

def main():
    """Main function to train all models"""
    parser = argparse.ArgumentParser(description="Batch train models for top 10 NASDAQ shares.")
    parser.add_argument('--no-interactive', action='store_true', help='Run in non-interactive mode (no prompts, no skips, no delays)')
    args = parser.parse_args()

    interactive = not args.no_interactive

    print("🤖 BATCH TRAINING - TOP 10 NASDAQ SHARES")
    print("=" * 60)
    print(f"📅 Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Target shares: {', '.join(TOP_10_NASDAQ)}")
    print()
    
    # Create batch run directory
    batch_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    batch_dir = f"batch_runs/top10_nasdaq_{batch_id}"
    os.makedirs(batch_dir, exist_ok=True)
    
    # Track results
    results = {
        'successful': [],
        'failed': [],
        'skipped': []
    }
    
    start_time = time.time()
    
    for i, symbol in enumerate(TOP_10_NASDAQ, 1):
        print(f"\n📊 Progress: {i}/{len(TOP_10_NASDAQ)} - {symbol}")
        
        if interactive:
            # Check if we should skip (optional - you can remove this)
            skip = input(f"Skip {symbol}? (y/n, default=n): ").strip().lower() == 'y'
            if skip:
                print(f"⏭️ Skipping {symbol}")
                results['skipped'].append(symbol)
                continue
        
        # Train the model
        success = train_single_model(symbol)
        
        if success:
            results['successful'].append(symbol)
        else:
            results['failed'].append(symbol)
        
        # Optional: Add delay between models to avoid overwhelming the system
        if interactive and i < len(TOP_10_NASDAQ):
            delay = input(f"Wait before next model? (seconds, default=5): ").strip()
            if delay:
                try:
                    delay_seconds = int(delay)
                    print(f"⏳ Waiting {delay_seconds} seconds...")
                    time.sleep(delay_seconds)
                except ValueError:
                    print("⏳ Waiting 5 seconds...")
                    time.sleep(5)
            else:
                print("⏳ Waiting 5 seconds...")
                time.sleep(5)
    
    # Summary
    total_time = time.time() - start_time
    print(f"\n🎉 BATCH TRAINING COMPLETED!")
    print("=" * 60)
    print(f"⏱️ Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print(f"✅ Successful: {len(results['successful'])} - {', '.join(results['successful'])}")
    print(f"❌ Failed: {len(results['failed'])} - {', '.join(results['failed'])}")
    print(f"⏭️ Skipped: {len(results['skipped'])} - {', '.join(results['skipped'])}")
    
    # Save batch results
    import json
    batch_results = {
        'batch_id': batch_id,
        'start_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_time_seconds': total_time,
        'results': results
    }
    
    with open(os.path.join(batch_dir, 'batch_results.json'), 'w') as f:
        json.dump(batch_results, f, indent=2)
    
    print(f"\n📁 Batch results saved to: {batch_dir}/")
    print("🔍 Use the ModelWrapper class to access your trained models!")

if __name__ == "__main__":
    main() 