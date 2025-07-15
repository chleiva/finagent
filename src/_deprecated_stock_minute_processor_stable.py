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
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import argparse
from tqdm import tqdm
import time

# Add the src directory to the path so we can import our features module
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from features import calculate_features
from extra_features import calculate_additional_features


from buy_label import evaluate_buy_signal

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


         # 🆕 KEEP A COPY OF FULL INTRADAY DATA FOR BUY SIGNAL EVALUATION
        full_symbol_intra_data_danger_zone_biased = symbol_intra_data.copy()  # This contains ALL data for the day
    
        
        if symbol_intra_data.empty:
            print(f"   ❌ No intraday data found for {symbol} on {target_date}")
            return [], {}
        
        print(f"   📈 Found {len(symbol_intra_data)} intraday records")
        print(f"   📊 Found {len(symbol_daily_data)} daily records")
        
        # Get unique minutes for the day
        minutes = self._get_minutes_for_day(symbol_intra_data)

        # 🎯 ADD THIS FILTER RIGHT HERE - FILTER OUT NON-MARKET HOURS
        from datetime import time
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


                #Here: please make sure that the date passed to calculate_features is free of future bias, i.e. intra_day_df can only contain data upto real_time_df, and daily_df can only contain data upto the day before of intra_day_df
                
                # Ensure intra_day_df only contains data up to the current minute (already handled by _create_intra_day_df)
                time_col = 'ts_event_clean' if 'ts_event_clean' in intra_day_df.columns else 'timestamp'
                
                # Double-check to make sure no future data is included in intra_day_df
                if time_col in intra_day_df.columns:
                    intra_day_df = intra_day_df[intra_day_df[time_col].dt.floor('min') < minute_time]
                
                # Ensure daily_df only contains data up to the day before current date
                current_date = pd.to_datetime(target_date).date()
                previous_day = current_date - pd.Timedelta(days=1)
                
                if 'date' in daily_df.columns:
                    daily_df = daily_df[daily_df['date'].dt.date <= previous_day]
                    
                # Sort data to ensure chronological order
                if time_col in intra_day_df.columns and not intra_day_df.empty:
                    intra_day_df = intra_day_df.sort_values(time_col)
                    
                if 'date' in daily_df.columns and not daily_df.empty:
                    daily_df = daily_df.sort_values('date')
                    
                # Calculate features
                features = calculate_features(real_time_df, intra_day_df, daily_df)

                extra_features = calculate_additional_features(real_time_df, intra_day_df, daily_df)

                features.update(extra_features)
                
                # Add real-time data fields at the beginning
                features['bidPrice'] = real_time_df['bidPrice'].iloc[0]
                features['bidSize'] = real_time_df['bidSize'].iloc[0]
                features['askPrice'] = real_time_df['askPrice'].iloc[0]
                features['askSize'] = real_time_df['askSize'].iloc[0]
                features['lastPrice'] = real_time_df['lastPrice'].iloc[0]
                features['lastSize'] = real_time_df['lastSize'].iloc[0]
                features['volume'] = real_time_df['volume'].iloc[0]
                
                # Add metadata
                features['symbol'] = symbol
                features['timestamp'] = minute_time
                features['date'] = target_date


                # 🆕 EVALUATE BUY SIGNAL (using FULL data to see future)
                signal_result = evaluate_buy_signal(
                    features, 
                    real_time_df, 
                    intra_day_df,  # Filtered data (for consistency)
                    daily_df,
                    full_symbol_intra_data_danger_zone_biased  # 🎯 FULL unfiltered data for future price lookup
                )
        
                # Add signal results to features
                features.update(signal_result)


                if len(intra_day_df) >= 15:
                    features_list.append(features)
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
        """Filter daily data for symbol and previous dates."""
        # Filter by symbol
        data = self.daily_data[self.daily_data['symbol'] == symbol].copy()
        
        # Filter for dates before target date (for historical context)
        data = data[data['date'].dt.date < target_date]
        
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
        
        # FIXED: Proper volume handling - use the CURRENT minute's volume, not cumulative
        real_time_df = pd.DataFrame({
            'bidPrice': [last_record['bid_px_00'].iloc[0]] if 'bid_px_00' in last_record.columns else [0],
            'bidSize': [last_record['bid_sz_00'].iloc[0]] if 'bid_sz_00' in last_record.columns else [0],
            'askPrice': [last_record['ask_px_00'].iloc[0]] if 'ask_px_00' in last_record.columns else [0],
            'askSize': [last_record['ask_sz_00'].iloc[0]] if 'ask_sz_00' in last_record.columns else [0],
            'lastPrice': [last_record['close_1min'].iloc[0]] if 'close_1min' in last_record.columns else [0],
            'lastSize': [100],  # FIXED: Use reasonable default instead of volume
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


def main():
    """Main function to run the stock minute processor."""
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
    
    args = parser.parse_args()
    
    print("🚀 Stock Minute-by-Minute Feature Processor")
    print("=" * 50)
    
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
    
    # Validate required arguments
    if not args.symbol:
        print("❌ Symbol is required. Use --list-symbols to see available symbols.")
        sys.exit(1)
    
    if not args.date:
        print("❌ Date is required. Use --list-dates SYMBOL to see available dates.")
        sys.exit(1)
    
    # Process the symbol and date
    start_time = time.time()
    features_list, summary = processor.process_symbol_date(args.symbol, args.date)
    processing_time = time.time() - start_time
    
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