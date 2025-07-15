#!/usr/bin/env python3
"""
Stock Minute-by-Minute Feature Processor

This application processes stock data minute by minute for a given symbol and date,
creating features for each minute using the FeatureCalculator.
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta, time
from typing import Dict, List, Tuple
import argparse
from tqdm import tqdm
import time as time_module
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import psutil
import gc
import threading
from collections import defaultdict
import glob
from feature_optimizer import optimize_features
from features_cleaner import clean_and_validate_features

# Set threading env vars to 1 for true multiprocessing (must be set before any numpy/pandas import in workers)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Add the src directory to the path so we can import our features module
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from features import calculate_features
from extra_features import calculate_additional_features
from buy_label import evaluate_buy_signal

# Optimization functions
def get_optimal_worker_count():
    """Get optimal number of workers for Apple Silicon."""
    total_cores = mp.cpu_count()
    print(f"🔥 Detected {total_cores} cores, using {total_cores} workers")
    return total_cores

def set_high_priority():
    """Set high priority for the process."""
    try:
        p = psutil.Process(os.getpid())
        if hasattr(p, "nice"):
            p.nice(-10)
            print("🚀 Set high process priority")
    except Exception as e:
        print(f"⚠️  Couldn't set priority: {e}")

def optimize_libraries():
    """Set threading vars for numpy/pandas backend libraries."""
    os.environ['OMP_NUM_THREADS'] = str(mp.cpu_count())
    os.environ['MKL_NUM_THREADS'] = str(mp.cpu_count())
    os.environ['NUMEXPR_NUM_THREADS'] = str(mp.cpu_count())
    print(f"🔧 Optimized libraries for {mp.cpu_count()} threads")

def optimize_memory():
    """Free unused memory and check available RAM."""
    gc.collect()
    memory = psutil.virtual_memory()
    print(f"💾 Available memory: {memory.available / (1024**3):.1f} GB")
    if memory.percent > 80:
        print("⚠️  High memory usage - consider closing other apps")

def monitor_performance():
    """Run performance monitor in background."""
    def monitor():
        while True:
            cpu_percent = psutil.cpu_percent(interval=5)
            memory = psutil.virtual_memory()
            print(f"📊 CPU: {cpu_percent:.1f}% | Memory: {memory.percent:.1f}% | Free: {memory.available / (1024**3):.1f} GB")
            time_module.sleep(10)
    
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    return monitor_thread

class StockMinuteProcessor:
    """
    Process stock data minute by minute and calculate features.
    """
    
    def __init__(self, intra_day_file: str, daily_file: str):
        """
        Initialize the processor with data files.
        
        Args:
            intra_day_file: Path to intraday CSV file
            daily_file: Path to daily CSV file
        """
        self.intra_day_file = intra_day_file
        self.daily_file = daily_file
        self.intra_day_data = None
        self.daily_data = None
        
    def load_data(self) -> bool:
        """
        Load the CSV files into memory.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            print("📊 Loading data files...")
            
            # Load intraday data
            print(f"   Loading intraday data from: {self.intra_day_file}")
            self.intra_day_data = pd.read_csv(self.intra_day_file)
            
            # Load daily data
            print(f"   Loading daily data from: {self.daily_file}")
            self.daily_data = pd.read_csv(self.daily_file)
            
            print(f"   ✅ Loaded {len(self.intra_day_data):,} intraday records")
            print(f"   ✅ Loaded {len(self.daily_data):,} daily records")
            
            # Process timestamp columns
            self._process_timestamps()
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def _process_timestamps(self):
        """Process and standardize timestamp columns."""
        # Process intraday timestamps
        if 'ts_event_clean' in self.intra_day_data.columns:
            self.intra_day_data['ts_event_clean'] = pd.to_datetime(self.intra_day_data['ts_event_clean'])
        elif 'timestamp' in self.intra_day_data.columns:
            self.intra_day_data['timestamp'] = pd.to_datetime(self.intra_day_data['timestamp'])
        
        # Process daily timestamps
        if 'date' in self.daily_data.columns:
            self.daily_data['date'] = pd.to_datetime(self.daily_data['date'])
        
        # Sort data by timestamp
        if 'ts_event_clean' in self.intra_day_data.columns:
            self.intra_day_data = self.intra_day_data.sort_values('ts_event_clean')
        elif 'timestamp' in self.intra_day_data.columns:
            self.intra_day_data = self.intra_day_data.sort_values('timestamp')
            
        if 'date' in self.daily_data.columns:
            self.daily_data = self.daily_data.sort_values('date')
    
    def get_symbols(self) -> List[str]:
        """
        Get list of available symbols from the data.
        
        Returns:
            List of available symbols
        """
        intra_symbols = set()
        daily_symbols = set()
        
        # Get symbols from intraday data
        if 'symbol_price' in self.intra_day_data.columns:
            intra_symbols = set(self.intra_day_data['symbol_price'].unique())
        elif 'symbol' in self.intra_day_data.columns:
            intra_symbols = set(self.intra_day_data['symbol'].unique())
        
        # Get symbols from daily data
        if 'symbol' in self.daily_data.columns:
            daily_symbols = set(self.daily_data['symbol'].unique())
        
        # Return intersection of both
        return sorted(list(intra_symbols.intersection(daily_symbols)))
    
    def get_available_dates(self, symbol: str) -> List[str]:
        """
        Get available dates for a given symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            List of available dates
        """
        # Filter intraday data for the symbol
        symbol_data = self.intra_day_data[
            (self.intra_day_data['symbol_price'] == symbol) if 'symbol_price' in self.intra_day_data.columns 
            else (self.intra_day_data['symbol'] == symbol)
        ]
        
        # Extract dates
        if 'ts_event_clean' in symbol_data.columns:
            dates = symbol_data['ts_event_clean'].dt.date.unique()
        elif 'timestamp' in symbol_data.columns:
            dates = symbol_data['timestamp'].dt.date.unique()
        else:
            return []
        
        return sorted([str(date) for date in dates])
    
    def process_symbol_date(self, symbol: str, target_date: str) -> Tuple[List[Dict], Dict]:
        """
        Process a specific symbol for a specific date.
        
        Args:
            symbol: Stock symbol
            target_date: Date to process (YYYY-MM-DD format)
            
        Returns:
            Tuple of (features_list, summary_stats)
        """
        target_date_obj = pd.to_datetime(target_date).date()
        
        print(f"\n🔍 Processing {symbol} for {target_date}")
        
        # Filter data for the symbol and date
        symbol_intra_data = self._filter_intra_data(symbol, target_date_obj)
        symbol_daily_data = self._filter_daily_data(symbol, target_date_obj)

        # Keep a copy of full intraday data for buy signal evaluation
        full_symbol_intra_data_danger_zone_biased = symbol_intra_data.copy()
        
        if symbol_intra_data.empty:
            print(f"   ❌ No intraday data found for {symbol} on {target_date}")
            return [], {}
        
        print(f"   📈 Found {len(symbol_intra_data)} intraday records")
        print(f"   📊 Found {len(symbol_daily_data)} daily records")
        
        # Get unique minutes for the day
        minutes = self._get_minutes_for_day(symbol_intra_data)

        # Filter out non-market hours
        market_open = time(9, 30)   # 30 min after 9:30 open
        market_close = time(16, 0)  # 1 hour before 4:00 close
        
        # Filter minutes to only include market hours where we can actually trade
        market_hours_minutes = []
        for minute_time in minutes:
            minute_time_only = minute_time.time()
            if market_open <= minute_time_only <= market_close:
                market_hours_minutes.append(minute_time)
        
        minutes = market_hours_minutes
        
        print(f"   ⏰ Processing {len(minutes)} minutes...")
        
        features_list = []
        processed_count = 0
        error_count = 0
        
        # Process each minute
        for minute_time in tqdm(minutes, desc=f"   Processing {symbol}", leave=False):
            try:
                # Create the three dataframes for this minute
                real_time_df = self._create_real_time_df(symbol_intra_data, minute_time)
                intra_day_df = self._create_intra_day_df(symbol_intra_data, minute_time)
                daily_df = symbol_daily_data.copy()

                # Ensure no future bias
                time_col = 'ts_event_clean' if 'ts_event_clean' in intra_day_df.columns else 'timestamp'
                
                # Double-check to make sure no future data is included in intra_day_df
                if time_col in intra_day_df.columns:
                    intra_day_df = intra_day_df[intra_day_df[time_col].dt.floor('min') < minute_time]
                
                # Ensure daily_df only contains data up to the day before current date
                # Ensure daily_df only contains data up to the day before current date, 
                # EXCEPT for current day's OPEN price only
                current_date = pd.to_datetime(target_date).date()
                previous_day = current_date - pd.Timedelta(days=1)

                if 'date' in daily_df.columns:
                    # Filter to previous days + current day
                    daily_df = daily_df[daily_df['date'].dt.date <= current_date]
                    
                    # For current day: keep ONLY the opening price, null out everything else
                    current_day_mask = daily_df['date'].dt.date == current_date
                    if current_day_mask.any():
                        # Null out HIGH, LOW, CLOSE, VOLUME for current day (keep OPEN)
                        daily_df.loc[current_day_mask, ['high', 'low', 'close', 'volume']] = np.nan
                        # OPEN remains unchanged - this gives us access to official opening price
                    
                # Sort data to ensure chronological order
                if time_col in intra_day_df.columns and not intra_day_df.empty:
                    intra_day_df = intra_day_df.sort_values(time_col)
                    
                if 'date' in daily_df.columns and not daily_df.empty:
                    daily_df = daily_df.sort_values('date')
                    
                # Calculate features
                features = calculate_features(real_time_df, intra_day_df, daily_df)
                extra_features = calculate_additional_features(real_time_df, intra_day_df, daily_df)
                features.update(extra_features)

                # DEBUG: Check opening price calculation
                current_price = features.get('Current_Price', 0)
                return_since_open = features.get('Return_Since_Open', 0)  # FIXED: use features.get()
                return_enhanced = features.get('Return_Since_Open_Enhanced', 0)

                if abs(return_since_open) > 50:  # Flag extreme values
                    print(f"   🚨 DEBUG EXTREME RETURN: {minute_time}")
                    print(f"      Current Price: {current_price}")
                    print(f"      Return_Since_Open: {return_since_open:.3f}%")
                    print(f"      Return_Enhanced: {return_enhanced:.3f}%")
                    
                    # Check what opening price was used
                    if not daily_df.empty and 'open' in daily_df.columns:
                        official_open = daily_df['open'].iloc[-1]
                        print(f"      Official Daily Open: {official_open}")
                        if official_open > 0:
                            expected_return = ((current_price - official_open) / official_open) * 100
                            print(f"      Expected Return: {expected_return:.3f}%")
                    
                    # Check intraday first price
                    if not intra_day_df.empty and 'open_1min' in intra_day_df.columns:
                        intra_first_open = intra_day_df['open_1min'].iloc[0]
                        print(f"      Intraday First Open: {intra_first_open}")
                        if intra_first_open > 0:
                            wrong_calc = ((current_price - intra_first_open) / intra_first_open) * 100
                            print(f"      If using intraday first: {wrong_calc:.3f}%")
                    
                    # Check daily_df content
                    print(f"      Daily DF shape: {daily_df.shape}")
                    if not daily_df.empty:
                        print(f"      Daily DF columns: {daily_df.columns.tolist()}")
                        print(f"      Daily DF sample:")
                        print(daily_df.head().to_string())

                    
                    # Check daily_df content
                    print(f"      Daily DF shape: {daily_df.shape}")
                    if not daily_df.empty:
                        print(f"      Daily DF columns: {daily_df.columns.tolist()}")
                        print(f"      Daily DF sample:")
                        print(daily_df.head().to_string())
                
                # Add real-time data fields at the beginning
                #features['bidPrice'] = real_time_df['bidPrice'].iloc[0]
                #features['bidSize'] = real_time_df['bidSize'].iloc[0]
                #features['askPrice'] = real_time_df['askPrice'].iloc[0]
                #features['askSize'] = real_time_df['askSize'].iloc[0]
                #features['lastPrice'] = real_time_df['lastPrice'].iloc[0]
                #features['lastSize'] = real_time_df['lastSize'].iloc[0]
                #features['volume'] = real_time_df['volume'].iloc[0]
                
                # Add metadata
                #features['symbol'] = symbol
                #features['timestamp'] = minute_time
                #features['date'] = target_date

                # Evaluate buy signal (using full data to see future)
                signal_result = evaluate_buy_signal(
                    features, 
                    real_time_df, 
                    intra_day_df,  # Filtered data (for consistency)
                    daily_df,
                    full_symbol_intra_data_danger_zone_biased  # Full unfiltered data for future price lookup
                )


                # Define valid reasons (same as your analysis)
                valid_reasons = {
                    "end_of_day_no_2r",
                    "reached_2r_success",
                    "reached_3r_success", 
                    "reached_4r_success",
                    "reached_negative_1r"
                }

                # Add signal results to features
                features.update(signal_result)

                # Check both conditions: sufficient data AND valid reason
                if (len(intra_day_df) >= 15 and 
                    signal_result.get('reason') in valid_reasons):
                    
                    # 🧹 STEP 1: Clean and validate features first
                    cleaned_features = clean_and_validate_features(features)
                    
                    # 🚀 STEP 2: Optimize the clean features
                    optimized_features = optimize_features(cleaned_features, intra_day_df=intra_day_df)
    
                    features_list.append(optimized_features)
                    processed_count += 1

                
            except Exception as e:
                error_count += 1
                print(f"   ⚠️  Error processing minute {minute_time}: {e}")
                
        # Calculate summary statistics
        summary_stats = {
            'symbol': symbol,
            'date': target_date,
            'total_minutes': len(minutes),
            'processed_minutes': processed_count,
            'error_count': error_count,
            'success_rate': processed_count / len(minutes) * 100 if len(minutes) > 0 else 0
        }
        
        print(f"   ✅ Successfully processed {processed_count}/{len(minutes)} minutes ({summary_stats['success_rate']:.1f}%)")
        
        return features_list, summary_stats
    
    def _filter_intra_data(self, symbol: str, target_date) -> pd.DataFrame:
        """Filter intraday data for symbol and date."""
        # Filter by symbol
        symbol_col = 'symbol_price' if 'symbol_price' in self.intra_day_data.columns else 'symbol'
        data = self.intra_day_data[self.intra_day_data[symbol_col] == symbol].copy()
        
        # Filter by date
        time_col = 'ts_event_clean' if 'ts_event_clean' in data.columns else 'timestamp'
        data = data[data[time_col].dt.date == target_date]
        
        return data
    
    def _filter_daily_data(self, symbol: str, target_date) -> pd.DataFrame:
        """Filter daily data for symbol and dates up to target date."""
        # Filter by symbol
        data = self.daily_data[self.daily_data['symbol'] == symbol].copy()
        
        # Filter for dates up to and INCLUDING target date 
        data = data[data['date'].dt.date <= target_date]  # ✅ INCLUDES current day
        
        # For current day: keep ONLY the opening price, null out everything else
        current_day_mask = data['date'].dt.date == target_date
        if current_day_mask.any():
            # Null out HIGH, LOW, CLOSE, VOLUME for current day (keep OPEN)
            data.loc[current_day_mask, ['high', 'low', 'close', 'volume']] = np.nan
            # OPEN remains unchanged - this gives us access to official opening price
            print(f"   🔍 Including current day opening price for {symbol} on {target_date}")
        
        return data
    
    def _get_minutes_for_day(self, symbol_data: pd.DataFrame) -> List[pd.Timestamp]:
        """Get all unique minutes for the day."""
        time_col = 'ts_event_clean' if 'ts_event_clean' in symbol_data.columns else 'timestamp'
        
        # Round to minute and get unique values
        minutes = symbol_data[time_col].dt.floor('min').unique()
        return sorted(minutes)
    
    def _create_real_time_df(self, symbol_data: pd.DataFrame, minute_time: pd.Timestamp) -> pd.DataFrame:
        """Create real-time dataframe for the current minute."""
        time_col = 'ts_event_clean' if 'ts_event_clean' in symbol_data.columns else 'timestamp'
        
        # Get data for this exact minute
        minute_data = symbol_data[symbol_data[time_col].dt.floor('min') == minute_time]
        
        if minute_data.empty:
            raise ValueError(f"No data found for minute {minute_time}")
        
        # Take the last record of the minute as "real-time"
        last_record = minute_data.iloc[-1:].copy()
        
        # Proper volume handling - use the current minute's volume, not cumulative
        real_time_df = pd.DataFrame({
            'bidPrice': [last_record['bid_px_00'].iloc[0]] if 'bid_px_00' in last_record.columns else [0],
            'bidSize': [last_record['bid_sz_00'].iloc[0]] if 'bid_sz_00' in last_record.columns else [0],
            'askPrice': [last_record['ask_px_00'].iloc[0]] if 'ask_px_00' in last_record.columns else [0],
            'askSize': [last_record['ask_sz_00'].iloc[0]] if 'ask_sz_00' in last_record.columns else [0],
            'lastPrice': [last_record['close_1min'].iloc[0]] if 'close_1min' in last_record.columns else [0],
            'lastSize': [100],  # Use reasonable default instead of volume
            'volume': [last_record['volume_1min'].iloc[0]] if 'volume_1min' in last_record.columns else [0],
            'timestamp': [minute_time]
        })
        
        return real_time_df
    
    def _create_intra_day_df(self, symbol_data: pd.DataFrame, current_minute: pd.Timestamp) -> pd.DataFrame:
        """Create intraday dataframe with data up to (but not including) current minute."""
        time_col = 'ts_event_clean' if 'ts_event_clean' in symbol_data.columns else 'timestamp'
        
        # Get data up to current minute (exclusive)
        intra_day_data = symbol_data[symbol_data[time_col].dt.floor('min') < current_minute]
        
        return intra_day_data
    
    def save_features(self, features_list: List[Dict], output_file: str):
        """
        Save features to CSV file.
        
        Args:
            features_list: List of feature dictionaries
            output_file: Output CSV file path
        """
        if not features_list:
            print("   ❌ No features to save")
            return
        
        try:
            df = pd.DataFrame(features_list)
            
            # Reorder columns to put real-time fields first, then features, then metadata
            real_time_cols = ['bidPrice', 'bidSize', 'askPrice', 'askSize', 'lastPrice', 'lastSize', 'volume']
            metadata_cols = ['symbol', 'timestamp', 'date']
            
            # Get all columns that exist in the dataframe
            existing_real_time_cols = [col for col in real_time_cols if col in df.columns]
            existing_metadata_cols = [col for col in metadata_cols if col in df.columns]
            
            # Get feature columns (everything except real-time and metadata)
            feature_cols = [col for col in df.columns if col not in existing_real_time_cols + existing_metadata_cols]
            
            # Reorder columns: real-time first, then features, then metadata
            ordered_cols = existing_real_time_cols + feature_cols + existing_metadata_cols
            df = df[ordered_cols]
            
            df.to_csv(output_file, index=False)
            print(f"   💾 Saved {len(features_list)} feature records to: {output_file}")
            
        except Exception as e:
            print(f"   ❌ Error saving features: {e}")
    
    def print_summary(self, all_summaries: List[Dict]):
        """
        Print processing summary.
        
        Args:
            all_summaries: List of summary dictionaries
        """
        print("\n" + "="*60)
        print("📊 PROCESSING SUMMARY")
        print("="*60)
        
        total_minutes = sum(s['total_minutes'] for s in all_summaries)
        total_processed = sum(s['processed_minutes'] for s in all_summaries)
        total_errors = sum(s['error_count'] for s in all_summaries)
        
        print(f"Total symbols processed: {len(all_summaries)}")
        print(f"Total minutes processed: {total_processed:,}/{total_minutes:,}")
        print(f"Total errors: {total_errors:,}")
        print(f"Overall success rate: {total_processed/total_minutes*100:.1f}%" if total_minutes > 0 else "N/A")
        
        print("\nPer-symbol breakdown:")
        for summary in all_summaries:
            print(f"  {summary['symbol']} ({summary['date']}): "
                  f"{summary['processed_minutes']}/{summary['total_minutes']} "
                  f"({summary['success_rate']:.1f}%)")
        
        print("="*60)

def process_symbol_all_dates(args_tuple):
    """
    Process all dates for a single symbol, saving monthly files.
    This function will run in a separate process.
    """
    symbol, intraday_data, daily_data, overwrite = args_tuple
    
    # Create a processor instance with pre-loaded data
    processor = StockMinuteProcessor("", "")
    processor.intra_day_data = intraday_data
    processor.daily_data = daily_data
    
    # Process timestamps
    processor._process_timestamps()
    
    dates = processor.get_available_dates(symbol)
    print(f"🔄 Process {os.getpid()}: Processing {symbol} ({len(dates)} dates)")
    
    # Group dates by month
    monthly_dates = defaultdict(list)
    
    for date_str in dates:
        date_obj = pd.to_datetime(date_str).date()
        month_key = f"{date_obj.year}-{date_obj.month:02d}"
        monthly_dates[month_key].append(date_str)
    
    print(f"   📅 Found {len(monthly_dates)} months for {symbol}")
    
    symbol_summaries = []
    total_features = 0
    skipped_months = 0
    
    # Process each month separately
    for month_key, month_dates in monthly_dates.items():
        # Check if monthly file already exists
        monthly_file = f"monthly_{symbol}_{month_key}.csv"
        if os.path.exists(monthly_file) and not overwrite:
            print(f"   🔄 Skipping {symbol} - {month_key}: File already exists and --overwrite not specified")
            skipped_months += 1
            continue
            
        print(f"   🗓️  Processing {symbol} - {month_key} ({len(month_dates)} days)")
        
        month_features = []
        month_summaries = []
        
        # Process all dates in this month
        for date in month_dates:
            try:
                features_list, summary = processor.process_symbol_date(symbol, date)
                month_summaries.append(summary)
                month_features.extend(features_list)
            except Exception as e:
                print(f"❌ Process {os.getpid()}: Error processing {symbol} on {date}: {e}")
        
        # Save monthly file immediately
        if month_features:
            temp_processor = StockMinuteProcessor("", "")
            temp_processor.save_features(month_features, monthly_file)
            print(f"   💾 Monthly file saved: {monthly_file} ({len(month_features)} features)")
            
            total_features += len(month_features)
            symbol_summaries.extend(month_summaries)
    
    if skipped_months > 0:
        print(f"✅ Process {os.getpid()}: {symbol} - Skipped {skipped_months} months as files already exist")
    
    print(f"✅ Process {os.getpid()}: Completed {symbol} - {total_features} total features across {len(monthly_dates) - skipped_months} months")
    return symbol, total_features, symbol_summaries, skipped_months

def merge_monthly_files(pattern="monthly_*.csv", output="merged_features.csv"):
    """
    Merge all monthly files into one master file.
    """
    files = glob.glob(pattern)
    files.sort()
    
    print(f"🔄 Merging {len(files)} monthly files...")
    
    if not files:
        print("❌ No monthly files found!")
        return 0
    
    all_data = []
    for file in files:
        try:
            df = pd.read_csv(file)
            all_data.append(df)
            print(f"   ✅ Added {file}: {len(df):,} records")
        except Exception as e:
            print(f"   ❌ Error reading {file}: {e}")
    
    if all_data:
        merged_df = pd.concat(all_data, ignore_index=True)
        
        # Sort by symbol and timestamp for clean output
        if 'symbol' in merged_df.columns and 'timestamp' in merged_df.columns:
            merged_df = merged_df.sort_values(['symbol', 'timestamp'])
        
        merged_df.to_csv(output, index=False)
        print(f"💾 Merged file saved: {output} ({len(merged_df):,} total records)")
        
        # Print summary by symbol
        if 'symbol' in merged_df.columns:
            symbol_counts = merged_df['symbol'].value_counts().sort_index()
            print(f"\n📊 Records per symbol:")
            for symbol, count in symbol_counts.items():
                print(f"   {symbol}: {count:,} records")
    
    return len(all_data)

def list_monthly_files():
    """List all monthly files and their status."""
    files = glob.glob("monthly_*.csv")
    files.sort()
    
    print(f"📁 Found {len(files)} monthly files:")
    
    if not files:
        print("   No monthly files found.")
        return
    
    # Group files by symbol
    symbol_files = defaultdict(list)
    total_records = 0
    
    for file in files:
        # Extract symbol and month from filename
        parts = file.replace('monthly_', '').replace('.csv', '').split('_')
        if len(parts) >= 2:
            symbol = parts[0]
            month = '_'.join(parts[1:])
            symbol_files[symbol].append((month, file))
    
    # Print files by symbol
    symbols = sorted(symbol_files.keys())
    for symbol in symbols:
        print(f"\n   🔶 {symbol} ({len(symbol_files[symbol])} months):")
        
        # Sort months chronologically
        symbol_months = sorted(symbol_files[symbol])
        
        symbol_records = 0
        for month, file in symbol_months:
            try:
                df = pd.read_csv(file)
                records = len(df)
                symbol_records += records
                total_records += records
                
                print(f"      📄 {month}: {records:,} records")
            except Exception as e:
                print(f"      ❌ {month}: Error reading file ({e})")
        
        print(f"      📊 Total for {symbol}: {symbol_records:,} records")
    
    print(f"\n📊 Total: {total_records:,} records across {len(files)} files")
    print(f"📈 Symbols: {len(symbols)}")
    
    # Print any missing months in sequences
    print("\n🔍 Checking for missing months in sequences...")
    for symbol in symbols:
        months = [m[0] for m in symbol_files[symbol]]
        if len(months) <= 1:
            continue
            
        # Try to identify missing months in sequence
        try:
            # Parse months to datetime objects for easier comparison
            parsed_months = []
            for month in months:
                try:
                    # Handle different possible formats
                    if '-' in month:
                        parsed_months.append(pd.to_datetime(month))
                except:
                    pass
                    
            if parsed_months:
                parsed_months.sort()
                
                # Check for gaps
                has_gaps = False
                for i in range(len(parsed_months)-1):
                    curr = parsed_months[i]
                    next_month = parsed_months[i+1]
                    
                    # Check if there's more than a 1-month gap
                    if (next_month.year * 12 + next_month.month) - (curr.year * 12 + curr.month) > 1:
                        if not has_gaps:
                            print(f"   🔍 {symbol} has missing months:")
                            has_gaps = True
                            
                        # Generate missing months
                        missing_month = curr
                        while True:
                            # Move to next month
                            if missing_month.month == 12:
                                missing_month = pd.Timestamp(year=missing_month.year + 1, month=1, day=1)
                            else:
                                missing_month = pd.Timestamp(year=missing_month.year, month=missing_month.month + 1, day=1)
                                
                            # Check if we've reached the next existing month
                            if missing_month.year == next_month.year and missing_month.month == next_month.month:
                                break
                                
                            print(f"      ❓ Missing: {missing_month.strftime('%Y-%m')}")
        except Exception as e:
            # Skip any errors in missing month detection
            pass

def main():
    """Main function to run the stock minute processor."""
    # Apply optimizations first
    set_high_priority()
    optimize_libraries() 
    optimize_memory()
    
    parser = argparse.ArgumentParser(description="Process stock data minute by minute")
    parser.add_argument("--symbol", "-s", type=str, help="Stock symbol to process")
    parser.add_argument("--date", "-d", type=str, help="Date to process (YYYY-MM-DD)")
    parser.add_argument("--intraday-file", "-i", type=str, default="../data/intraday.csv", 
                        help="Path to intraday CSV file")
    parser.add_argument("--daily-file", "-f", type=str, default="../data/daily.csv", 
                        help="Path to daily CSV file")
    parser.add_argument("--output", "-o", type=str, help="Output CSV file path")
    parser.add_argument("--list-symbols", action="store_true", help="List available symbols")
    parser.add_argument("--list-dates", type=str, help="List available dates for a symbol")
    parser.add_argument("--list-monthly-files", action="store_true", help="List all existing monthly files")
    parser.add_argument("--all", action="store_true", help="Process ALL symbols and ALL dates")
    parser.add_argument("--parallel", action="store_true", help="Use parallel processing (only with --all)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing monthly files instead of skipping them")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of parallel worker processes (default: all CPU cores)")
    
    args = parser.parse_args()
    
    print("🚀 Stock Minute-by-Minute Feature Processor (OPTIMIZED)")
    print("=" * 50)
    
    # Start performance monitoring
    monitor_thread = monitor_performance()
    
    # Handle list monthly files command
    if args.list_monthly_files:
        list_monthly_files()
        return
    
    # Initialize processor
    processor = StockMinuteProcessor(args.intraday_file, args.daily_file)
    
    if not processor.load_data():
        sys.exit(1)
    
    # Handle list symbols command
    if args.list_symbols:
        symbols = processor.get_symbols()
        print(f"\n📋 Available symbols ({len(symbols)}):")
        for symbol in symbols:
            print(f"  - {symbol}")
        return
    
    # Handle list dates command
    if args.list_dates:
        dates = processor.get_available_dates(args.list_dates)
        print(f"\n📅 Available dates for {args.list_dates} ({len(dates)}):")
        for date in dates:
            print(f"  - {date}")
        return
    
    # Handle process all command
    if args.all:
        symbols = processor.get_symbols()
        print(f"\n🔄 Processing ALL symbols and dates ({len(symbols)} symbols)")
        
        start_time = time_module.time()
        
        if args.parallel:
            # Optimized parallel processing
            num_cores = mp.cpu_count() if args.num_workers is None else args.num_workers
            print(f"🏃‍♂️ Using {num_cores} process(es) (OMP/MKL/NUMEXPR threads per process: 1)")
            print(f"📊 Loading data once and sharing across processes...")
            
            # Load data once in main process
            intraday_data = processor.intra_day_data.copy()
            daily_data = processor.daily_data.copy()
            
            print(f"   ✅ Data loaded: {len(intraday_data):,} intraday + {len(daily_data):,} daily records")
            
            # Prepare arguments for each symbol
            symbol_args = [(symbol, intraday_data, daily_data, args.overwrite) for symbol in symbols]
            
            all_summaries = []
            total_features_processed = 0
            
            # Process symbols in parallel - each creates monthly files
            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                results = list(executor.map(process_symbol_all_dates, symbol_args))
            
            # Collect summary statistics
            for symbol, feature_count, summaries, skipped_months in results:
                total_features_processed += feature_count
                all_summaries.extend(summaries)
            
            processing_time = time_module.time() - start_time
            
            # Calculate total skipped months
            total_skipped_months = sum(result[3] for result in results)
            
            print(f"\n🎉 All processing complete!")
            print(f"📊 Total features processed: {total_features_processed:,}")
            print(f"🔄 Total months skipped (files already exist): {total_skipped_months}")
            print(f"⏱️  Total time: {processing_time:.2f} seconds")
            print(f"⚡ Features per second: {total_features_processed/processing_time:.0f}")
            print(f"📁 Monthly files created - use merge_monthly_files() to combine")
            
            # Auto-merge if requested
            if args.output:
                print(f"\n🔄 Auto-merging monthly files...")
                merge_monthly_files(output=args.output)
            else:
                print(f"\n💡 To merge all monthly files later, run:")
                print(f"   python -c \"from stock_minute_processor import merge_monthly_files; merge_monthly_files()\"")
            
            return
        
        else:
            # Sequential processing (original, now improved)
            all_summaries = []
            total_features_processed = 0
            skipped_months = 0
            symbols_processed = 0
            
            for symbol in symbols:
                dates = processor.get_available_dates(symbol)
                print(f"\n📊 Processing {symbol} ({len(dates)} dates)")
                
                # Group dates by month
                monthly_dates = defaultdict(list)
                for date_str in dates:
                    date_obj = pd.to_datetime(date_str).date()
                    month_key = f"{date_obj.year}-{date_obj.month:02d}"
                    monthly_dates[month_key].append(date_str)
                print(f"   📅 Found {len(monthly_dates)} months for {symbol}")
                
                for month_key, month_dates in monthly_dates.items():
                    monthly_file = f"monthly_{symbol}_{month_key}.csv"
                    if os.path.exists(monthly_file) and not args.overwrite:
                        print(f"   🔄 Skipping {symbol} - {month_key}: File already exists and --overwrite not specified")
                        skipped_months += 1
                        continue
                    print(f"   🗓️  Processing {symbol} - {month_key} ({len(month_dates)} days)")
                    month_features = []
                    month_summaries = []
                    for date in month_dates:
                        try:
                            features_list, summary = processor.process_symbol_date(symbol, date)
                            month_summaries.append(summary)
                            month_features.extend(features_list)
                        except Exception as e:
                            print(f"❌ Error processing {symbol} on {date}: {e}")
                    # Save monthly file immediately
                    if month_features:
                        temp_processor = StockMinuteProcessor("", "")
                        temp_processor.save_features(month_features, monthly_file)
                        print(f"   💾 Monthly file saved: {monthly_file} ({len(month_features)} features)")
                        total_features_processed += len(month_features)
                        all_summaries.extend(month_summaries)
                symbols_processed += 1
            processing_time = time_module.time() - start_time
            print(f"\n🎉 All processing complete!")
            print(f"📊 Total features processed: {total_features_processed:,}")
            print(f"🔄 Total months skipped (files already exist): {skipped_months}")
            print(f"⏱️  Total time: {processing_time:.2f} seconds")
            print(f"⚡ Features per second: {total_features_processed/processing_time:.0f}" if processing_time > 0 else "N/A")
            print(f"📁 Monthly files created - use merge_monthly_files() to combine")
            # Auto-merge if requested
            if args.output:
                print(f"\n🔄 Auto-merging monthly files...")
                merge_monthly_files(output=args.output)
            else:
                print(f"\n💡 To merge all monthly files later, run:")
                print(f"   python -c \"from stock_minute_processor import merge_monthly_files; merge_monthly_files()\"")
            return
    
    # Validate required arguments for single symbol/date processing
    if not args.symbol:
        print("❌ Symbol is required. Use --list-symbols to see available symbols or --all to process everything.")
        sys.exit(1)
    
    if not args.date:
        print("❌ Date is required. Use --list-dates SYMBOL to see available dates or --all to process everything.")
        sys.exit(1)
    
    # Process the symbol and date
    start_time = time_module.time()
    features_list, summary = processor.process_symbol_date(args.symbol, args.date)
    processing_time = time_module.time() - start_time
    
    # Save features if output file specified
    if args.output:
        processor.save_features(features_list, args.output)
    else:
        # Default output file
        output_file = f"features_{args.symbol}_{args.date.replace('-', '')}.csv"
        processor.save_features(features_list, output_file)
    
    # Print summary
    processor.print_summary([summary])
    
    print(f"\n⏱️  Total processing time: {processing_time:.2f} seconds")
    print(f"⚡ Average time per minute: {processing_time/len(features_list):.3f} seconds" if features_list else "N/A")

if __name__ == "__main__":
    main()