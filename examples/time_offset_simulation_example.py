#!/usr/bin/env python3
"""
Example script demonstrating the time_offset feature in trading_simulation.py

This script shows how to use the time_offset argument to simulate trading 
at different points in the past using historical intraday data.
"""

import subprocess
import sys
import os
from datetime import datetime, timedelta

def run_simulation_example():
    """Run trading simulation with different time offsets"""
    
    # Add the parent directory to the path so we can run the trading simulation
    simulation_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'real_time', 'trading_simulation.py')
    
    print("=" * 80)
    print("🕐 TRADING SIMULATION TIME OFFSET EXAMPLES")
    print("=" * 80)
    
    examples = [
        {
            'offset': 0,
            'description': 'Real-time mode (no offset)',
            'usage': 'python trading_simulation.py'
        },
        {
            'offset': 60,
            'description': 'Simulate trading 1 hour ago',
            'usage': 'python trading_simulation.py --time_offset 60'
        },
        {
            'offset': 120,
            'description': 'Simulate trading 2 hours ago',
            'usage': 'python trading_simulation.py --time_offset 120'
        },
        {
            'offset': 240,
            'description': 'Simulate trading 4 hours ago',
            'usage': 'python trading_simulation.py --time_offset 240'
        }
    ]
    
    for example in examples:
        print(f"\n📋 Example: {example['description']}")
        print(f"   Usage: {example['usage']}")
        print(f"   Time offset: {example['offset']} minutes")
        
        if example['offset'] == 0:
            print("   🔴 Real-time mode: Uses live market data")
        else:
            simulated_time = datetime.now() - timedelta(minutes=example['offset'])
            print(f"   🕐 Simulated time: {simulated_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   📊 Data source: Intraday data from market open until {simulated_time.strftime('%H:%M')} minus 1 minute")
        
        print("-" * 60)

def show_usage():
    """Show detailed usage information"""
    print("\n" + "=" * 80)
    print("🚀 HOW TO USE THE TIME OFFSET FEATURE")
    print("=" * 80)
    
    print("\n1. BASIC USAGE:")
    print("   cd src/real_time")
    print("   python trading_simulation.py --time_offset <minutes>")
    
    print("\n2. EXAMPLES:")
    print("   # Real-time mode (default)")
    print("   python trading_simulation.py")
    print("   python trading_simulation.py --time_offset 0")
    
    print("\n   # Simulate trading 30 minutes ago")
    print("   python trading_simulation.py --time_offset 30")
    
    print("\n   # Simulate trading 2 hours ago")
    print("   python trading_simulation.py --time_offset 120")
    
    print("\n   # Test during market hours (e.g., 1 hour ago)")
    print("   python trading_simulation.py --time_offset 60")
    
    print("\n3. WHAT THE TIME OFFSET DOES:")
    print("   📅 Simulated Current Time = Actual Time - time_offset minutes")
    print("   📊 Real-time data: Uses intraday data closest to simulated time")
    print("   📈 Intraday data: Uses data from market open until simulated time - 1 minute")
    print("   📉 Daily data: Uses historical daily data (unchanged)")
    
    print("\n4. BENEFITS:")
    print("   ✅ Test your trading strategy outside market hours")
    print("   ✅ Backtest specific time periods")
    print("   ✅ Validate model performance on historical data")
    print("   ✅ Debug and experiment safely")
    
    print("\n5. LIMITATIONS:")
    print("   ⚠️  Only works with experimentation mode (no real trading)")
    print("   ⚠️  Requires sufficient intraday data in the database")
    print("   ⚠️  Simulated bid/ask spreads (0.1% around last price)")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    run_simulation_example()
    show_usage()
    
    print("\n🎯 QUICK START:")
    print("   Try running: python ../src/real_time/trading_simulation.py --time_offset 60")
    print("   This will simulate trading 1 hour ago using historical data.")
    print("\n   Press Ctrl+C to stop the simulation at any time.") 