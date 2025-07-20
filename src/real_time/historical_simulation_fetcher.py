#!/usr/bin/env python3
"""
Local CSV Historical Simulation Data Fetcher
Reads 1-minute historical data from local CSV file for simulation purposes
"""

import sqlite3
import pandas as pd
from datetime import datetime, date
import os
from pathlib import Path

class LocalHistoricalSimulationFetcher:
    """Fetcher for historical 1-minute data from local CSV file"""
    
    def __init__(self, db_path="database/realtime_market_data.db", csv_path="src/feature_engineering/intraday.csv"):
        self.db_path = db_path
        self.csv_path = csv_path
        
        # Ensure database directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize the simulation table
        self.init_simulation_table()
        
        # Load data on initialization
        #self._load_csv_data()
    
    def init_simulation_table(self):
        """Initialize the historical simulation data table with ESSENTIAL columns only"""
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        cursor = conn.cursor()
        
        # Create table with ONLY essential columns to avoid "too many SQL variables" error
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS historical_simulation_intraday_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_event_price DATETIME NOT NULL,
                open_1min REAL,
                high_1min REAL,
                low_1min REAL,
                close_1min REAL,
                volume_1min INTEGER,
                symbol_price TEXT NOT NULL,
                bid_px_00 REAL,
                ask_px_00 REAL,
                bid_sz_00 REAL,
                ask_sz_00 REAL,
                simulation_date DATE NOT NULL,
                data_source TEXT DEFAULT 'local_csv',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol_price, ts_event_price, simulation_date)
            )
        ''')
        
        # Create indexes for fast querying
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_hist_sim_symbol_date_time 
            ON historical_simulation_intraday_data(symbol_price, simulation_date, ts_event_price)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_hist_sim_date 
            ON historical_simulation_intraday_data(simulation_date)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_hist_sim_symbol 
            ON historical_simulation_intraday_data(symbol_price)
        ''')
        
        conn.commit()
        conn.close()
        print("✅ Historical simulation table initialized with essential columns only")
    
    def _load_csv_data(self):
        """Load ESSENTIAL data from CSV file into the database"""
        if not os.path.exists(self.csv_path):
            print(f"❌ CSV file not found: {self.csv_path}")
            return False
        
        print(f"📊 Loading data from {self.csv_path}")
        
        try:
            # Read CSV file with only essential columns
            essential_columns = [
                'ts_event_price', 'open_1min', 'high_1min', 'low_1min', 'close_1min', 
                'volume_1min', 'symbol_price', 'bid_px_00', 'ask_px_00', 'bid_sz_00', 'ask_sz_00'
            ]
            
            df = pd.read_csv(self.csv_path, usecols=essential_columns)
            
            print(f"📊 CSV contains {len(df)} rows with {len(essential_columns)} essential columns")
            print(f"📊 Essential columns: {essential_columns}")
            
            # Convert timestamp column to datetime
            df['ts_event_price'] = pd.to_datetime(df['ts_event_price'], errors='coerce')
            
            # Extract date for simulation_date from ts_event_price
            df['simulation_date'] = df['ts_event_price'].dt.date


            # DEBUG: Check for duplicates
            duplicates = df.groupby(['symbol_price', 'ts_event_price', 'simulation_date']).size()
            duplicate_count = (duplicates > 1).sum()
            print(f"🐛 DEBUG: Found {duplicate_count} duplicate combinations of (symbol, timestamp, date)")
            if duplicate_count > 0:
                print(f"🐛 DEBUG: Sample duplicates:")
                print(duplicates[duplicates > 1].head())


            # Drop duplicates keeping the first occurrence
            df = df.drop_duplicates(subset=['symbol_price', 'ts_event_price'], keep='first')
            print(f"📊 After removing duplicates: {len(df)} rows remaining")
            
            # Add data_source column
            df['data_source'] = 'local_csv'
            
            # Check if data already exists to avoid duplicates
            conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM historical_simulation_intraday_data WHERE data_source = "local_csv"')
            existing_records = cursor.fetchone()[0]
            
            if existing_records > 0:
                print(f"⚠️ Found {existing_records} existing records from local_csv.")
                cursor.execute('DELETE FROM historical_simulation_intraday_data WHERE data_source = "local_csv"')
                conn.commit()
            
            # Insert data in smaller chunks to avoid "too many SQL variables" error
            chunk_size = 1000  # Process 1000 rows at a time
            total_inserted = 0
            
            for i in range(0, len(df), chunk_size):
                chunk = df.iloc[i:i+chunk_size]
                try:
                    chunk.to_sql('historical_simulation_intraday_data', conn, 
                                if_exists='append', index=False)
                    total_inserted += len(chunk)
                    if i % 10000 == 0:  # Progress update every 10k rows
                        print(f"  📊 Processed {total_inserted:,} rows...")
                except Exception as e:
                    print(f"❌ Error inserting chunk {i}-{i+chunk_size}: {e}")
                    conn.rollback()
                    conn.close()
                    return False
            
            conn.commit()
            
            # Get final count
            cursor.execute('SELECT COUNT(*) FROM historical_simulation_intraday_data WHERE data_source = "local_csv"')
            total_records = cursor.fetchone()[0]
            
            conn.close()
            
            print(f"✅ Successfully loaded {total_inserted:,} records into database (total: {total_records:,})")
            
            # Show sample of what was loaded
            print(f"📊 Sample data loaded:")
            print(f"   Date range: {df['simulation_date'].min()} to {df['simulation_date'].max()}")
            print(f"   Symbols: {df['symbol_price'].nunique()} unique symbols")
            print(f"   Symbol list: {sorted(df['symbol_price'].unique())}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading CSV data: {e}")
            return False
    
    def get_minute_data_by_date_and_symbol(self, symbol, target_date):
        """
        Get ALL minute data for a specific symbol and date
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            target_date: Date to fetch data for (datetime, date object, or string)
            
        Returns:
            List of dictionaries with all CSV columns
        """
        # Convert target_date to date object if needed
        if isinstance(target_date, str):
            target_date = datetime.strptime(target_date, '%Y-%m-%d').date()
        elif isinstance(target_date, datetime):
            target_date = target_date.date()
        
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        conn.row_factory = sqlite3.Row  # This allows us to access columns by name
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM historical_simulation_intraday_data
            WHERE symbol_price = ? AND simulation_date = ?
            ORDER BY ts_event_price ASC
        ''', (symbol, target_date))
        
        rows = cursor.fetchall()
        conn.close()
        
        # Convert to list of dictionaries for easier access
        return [dict(row) for row in rows]
    
    def get_basic_ohlcv_data(self, symbol, target_date):
        """
        Get basic OHLCV data for backward compatibility
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            target_date: Date to fetch data for (datetime, date object, or string)
            
        Returns:
            List of tuples: (bar_time, open, high, low, close, volume)
        """
        # Convert target_date to date object if needed
        if isinstance(target_date, str):
            target_date = datetime.strptime(target_date, '%Y-%m-%d').date()
        elif isinstance(target_date, datetime):
            target_date = target_date.date()
        
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT ts_event_price, open_1min, high_1min, low_1min, close_1min, volume_1min
            FROM historical_simulation_intraday_data
            WHERE symbol_price = ? AND simulation_date = ?
            ORDER BY ts_event_price ASC
        ''', (symbol, target_date))
        
        rows = cursor.fetchall()
        conn.close()
        
        return rows
    
    def has_data_for_date(self, symbol, target_date):
        """Check if we already have data for a symbol on a specific date"""
        # Convert target_date to date object if needed
        if isinstance(target_date, str):
            target_date = datetime.strptime(target_date, '%Y-%m-%d').date()
        elif isinstance(target_date, datetime):
            target_date = target_date.date()
        
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT COUNT(*) FROM historical_simulation_intraday_data
            WHERE symbol_price = ? AND simulation_date = ?
        ''', (symbol, target_date))
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count > 0
    
    def get_available_symbols(self):
        """Get list of all available symbols in the database"""
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT DISTINCT symbol_price FROM historical_simulation_intraday_data
            ORDER BY symbol_price
        ''')
        
        symbols = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        return symbols
    
    def get_available_dates(self, symbol=None):
        """Get list of all available dates (optionally for a specific symbol)"""
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        cursor = conn.cursor()
        
        if symbol:
            cursor.execute('''
                SELECT DISTINCT simulation_date FROM historical_simulation_intraday_data
                WHERE symbol_price = ?
                ORDER BY simulation_date
            ''', (symbol,))
        else:
            cursor.execute('''
                SELECT DISTINCT simulation_date FROM historical_simulation_intraday_data
                ORDER BY simulation_date
            ''')
        
        dates = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        return dates
    
    def get_minute_data_for_date_range(self, symbol, start_date, end_date):
        """
        Get ALL minute data for a symbol within a date range
        
        Args:
            symbol: Stock symbol
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            
        Returns:
            List of dictionaries with all CSV columns
        """
        # Convert dates if needed
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        elif isinstance(start_date, datetime):
            start_date = start_date.date()
            
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        elif isinstance(end_date, datetime):
            end_date = end_date.date()
        
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM historical_simulation_intraday_data
            WHERE symbol_price = ? AND simulation_date BETWEEN ? AND ?
            ORDER BY ts_event_price ASC
        ''', (symbol, start_date, end_date))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_data_summary(self):
        """Get a summary of the data in the database"""
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        cursor = conn.cursor()
        
        # Get total records
        cursor.execute('SELECT COUNT(*) FROM historical_simulation_intraday_data')
        total_records = cursor.fetchone()[0]
        
        # Get unique symbols count
        cursor.execute('SELECT COUNT(DISTINCT symbol_price) FROM historical_simulation_intraday_data')
        symbol_count = cursor.fetchone()[0]
        
        # Get date range
        cursor.execute('''
            SELECT MIN(simulation_date), MAX(simulation_date) 
            FROM historical_simulation_intraday_data
        ''')
        date_range = cursor.fetchone()
        
        # Get records per symbol
        cursor.execute('''
            SELECT symbol_price, COUNT(*) as record_count
            FROM historical_simulation_intraday_data
            GROUP BY symbol_price
            ORDER BY record_count DESC
        ''')
        symbol_stats = cursor.fetchall()
        
        # Get available columns
        cursor.execute('PRAGMA table_info(historical_simulation_intraday_data)')
        columns = [row[1] for row in cursor.fetchall()]
        
        conn.close()
        
        return {
            'total_records': total_records,
            'symbol_count': symbol_count,
            'date_range': date_range,
            'symbol_stats': symbol_stats,
            'available_columns': columns
        }


# Example usage and testing functions
def main():
    """Example usage of the LocalHistoricalSimulationFetcher"""
    
    # Initialize the fetcher
    fetcher = LocalHistoricalSimulationFetcher()
    
    # Get data summary
    summary = fetcher.get_data_summary()
    print(f"\n📊 Data Summary:")
    print(f"Total records: {summary['total_records']:,}")
    print(f"Unique symbols: {summary['symbol_count']}")
    print(f"Date range: {summary['date_range'][0]} to {summary['date_range'][1]}")
    print(f"Available columns: {len(summary['available_columns'])}")
    print(f"Columns: {', '.join(summary['available_columns'][:10])}...")
    
    # Show available symbols
    symbols = fetcher.get_available_symbols()
    print(f"\n📈 Available symbols: {', '.join(symbols[:10])}...")
    
    # Example: Get ALL data for AAPL on a specific date
    if symbols:
        test_symbol = symbols[0]  # Use first available symbol
        dates = fetcher.get_available_dates(test_symbol)
        
        if dates:
            test_date = dates[0]  # Use first available date
            
            # Get ALL data with all columns
            all_data = fetcher.get_minute_data_by_date_and_symbol(test_symbol, test_date)
            print(f"\n📊 Complete data for {test_symbol} on {test_date}:")
            print(f"Found {len(all_data)} minute bars with {len(all_data[0].keys()) if all_data else 0} columns each")
            


if __name__ == "__main__":
    main()