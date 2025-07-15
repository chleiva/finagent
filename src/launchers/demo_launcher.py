#!/usr/bin/env python3
"""
Trading Model Launcher - Demonstration Script
=============================================

This script demonstrates various ways to use the trading model launcher
for different scenarios and use cases.
"""

import subprocess
import sys
import os

def run_demo(command, description):
    """Run a demo command and show the description"""
    print(f"\n{'='*60}")
    print(f"🎯 DEMO: {description}")
    print(f"{'='*60}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    # Ask user if they want to run this demo
    response = input("\nRun this demo? (y/n/s to skip all): ").strip().lower()
    if response == 's':
        print("Skipping all remaining demos.")
        return False
    elif response != 'y':
        print("Skipping this demo.")
        return True
    
    try:
        result = subprocess.run(command.split(), capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Demo completed successfully!")
        else:
            print(f"❌ Demo failed: {result.stderr}")
    except Exception as e:
        print(f"❌ Demo error: {e}")
    
    return True

def main():
    """Run various demos"""
    print("🚀 TRADING MODEL LAUNCHER - DEMONSTRATION")
    print("=" * 50)
    print("This script demonstrates different ways to use the launcher.")
    print("Each demo will show you a different use case.")
    print()
    
    demos = [
        {
            "command": "python trading_model_launcher.py --help",
            "description": "Show help and available options"
        },
        {
            "command": "python trading_model_launcher.py --stocks AAPL --year 2025 --month 01 --fast --description 'Single stock test'",
            "description": "Single stock, specific time period, fast mode"
        },
        {
            "command": "python trading_model_launcher.py --stocks AAPL,MSFT,GOOGL --year 2025 --description 'Tech giants 2025'",
            "description": "Multiple stocks, specific year, all months"
        },
        {
            "command": "python trading_model_launcher.py --stocks AMD,INTC,NVDA --year 2024 --month 12 --fast --description 'Semiconductor Dec 2024'",
            "description": "Semiconductor sector, specific month, fast mode"
        },
        {
            "command": "python trading_model_launcher.py --quick --description 'Quick comprehensive test'",
            "description": "Quick mode - all stocks, all time, fast training"
        }
    ]
    
    print("Available demos:")
    for i, demo in enumerate(demos, 1):
        print(f"{i}. {demo['description']}")
    
    print(f"\nYou can:")
    print("- Run each demo individually (recommended)")
    print("- Skip demos you don't want to run")
    print("- Type 's' to skip all remaining demos")
    print()
    
    for demo in demos:
        if not run_demo(demo["command"], demo["description"]):
            break
    
    print(f"\n{'='*60}")
    print("🎉 DEMONSTRATION COMPLETE!")
    print(f"{'='*60}")
    print("You've seen various ways to use the trading model launcher:")
    print("• Command-line mode with different parameters")
    print("• Single and multiple stock selection")
    print("• Time period filtering")
    print("• Fast mode for quick testing")
    print("• Quick mode for comprehensive training")
    print()
    print("Next steps:")
    print("1. Try interactive mode: python trading_model_launcher.py")
    print("2. Experiment with your own parameters")
    print("3. Check the model training log: python view_model_log.py")
    print("4. Compare models: python model_comparison_tool.py")
    print()
    print("Happy trading! 📈")

if __name__ == "__main__":
    main() 