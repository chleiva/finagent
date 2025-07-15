import glob
import argparse
import os
import pandas as pd

parser = argparse.ArgumentParser(description="Concatenate monthly CSV files with optional filters.")
parser.add_argument('--symbol', type=str, help='Share symbol, e.g., WDAY')
parser.add_argument('--symbols', type=str, help='Comma-separated list of symbols, e.g., AAPL,MSFT,GOOGL')
parser.add_argument('--year', type=str, help='Year, e.g., 2025')
parser.add_argument('--month', type=str, help='Month, e.g., 02')
parser.add_argument('--output', type=str, default='concatenated_data.csv', help='Output file name')
parser.add_argument('--verbose', action='store_true', help='Show detailed progress information')
args = parser.parse_args()

def get_matching_files(symbols, year, month):
    """Get all matching files based on parameters"""
    all_files = []
    
    # Set the data directory
    data_dir = "data/raw"
    if not os.path.exists(data_dir):
        print(f"❌ Data directory '{data_dir}' not found!")
        print("Please make sure you have data files in the data/raw/ directory.")
        exit(1)
    
    # Handle single symbol vs multiple symbols
    if args.symbol:
        symbols = [args.symbol]
    elif args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
    else:
        # No symbol specified, get all
        symbols = None
    
    if args.verbose:
        print(f"🔍 Searching in directory: {data_dir}")
        print(f"📊 Symbols: {symbols or 'all available'}")
        print(f"📅 Year: {year or 'all'}")
        print(f"📅 Month: {month or 'all'}")
    
    if symbols:
        # Specific symbols
        for symbol in symbols:
            if year and month:
                pattern = os.path.join(data_dir, f"monthly_{symbol}_{year}-{month}.csv")
            elif year:
                pattern = os.path.join(data_dir, f"monthly_{symbol}_{year}-*.csv")
            elif month:
                pattern = os.path.join(data_dir, f"monthly_{symbol}_*-{month}.csv")
            else:
                pattern = os.path.join(data_dir, f"monthly_{symbol}_*.csv")
            
            matching_files = glob.glob(pattern)
            if args.verbose:
                print(f"  🔍 {symbol}: Found {len(matching_files)} files")
            all_files.extend(matching_files)
    else:
        # All symbols
        if year and month:
            pattern = os.path.join(data_dir, f"monthly_*_{year}-{month}.csv")
        elif year:
            pattern = os.path.join(data_dir, f"monthly_*_{year}-*.csv")
        elif month:
            pattern = os.path.join(data_dir, f"monthly_*_*-{month}.csv")
        else:
            pattern = os.path.join(data_dir, f"monthly_*.csv")
        
        all_files = glob.glob(pattern)
        if args.verbose:
            print(f"  🔍 All symbols: Found {len(all_files)} files")
    
    return sorted(all_files)

# Get matching files
files = get_matching_files(args.symbols, args.year, args.month)

if not files:
    print(f"❌ No files matched the specified criteria.")
    print(f"Symbols: {args.symbol or args.symbols or 'all'}")
    print(f"Year: {args.year or 'all'}")
    print(f"Month: {args.month or 'all'}")
    print(f"Directory: data/raw/")
    exit(1)

print(f"✅ Found {len(files)} files to concatenate:")
for file in files[:10]:  # Show first 10 files
    filename = os.path.basename(file)
    print(f"  📄 {filename}")
if len(files) > 10:
    print(f"  ... and {len(files) - 10} more files")

# Concatenate files using pandas for better handling
print(f"\n🔄 Concatenating files into {args.output}...")

dataframes = []
for i, filename in enumerate(files):
    try:
        df = pd.read_csv(filename)
        dataframes.append(df)
        if args.verbose or (i + 1) % 10 == 0 or i == len(files) - 1:
            print(f"  ✅ Processed {i + 1}/{len(files)} files... ({os.path.basename(filename)})")
    except Exception as e:
        print(f"⚠️ Warning: Could not read {os.path.basename(filename)}: {e}")

if not dataframes:
    print("❌ No valid files could be read.")
    exit(1)

# Concatenate all dataframes
combined_df = pd.concat(dataframes, ignore_index=True)

# Save the combined dataset
combined_df.to_csv(args.output, index=False)

print(f"\n🎉 Successfully created {args.output}")
print(f"📊 Total rows: {len(combined_df):,}")
print(f"📊 Total columns: {len(combined_df.columns)}")
print(f"📁 File size: {os.path.getsize(args.output) / (1024*1024):.2f} MB")

# Show sample of the data
print(f"\n📋 Sample data (first 5 rows):")
print(combined_df.head())
