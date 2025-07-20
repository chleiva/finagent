#!/usr/bin/env python3
"""
Minimal IBKR WebSocket Data Collector
Fetches real-time data every 10 seconds and 1-minute historical data every 30 seconds
"""

import requests
import json
import sqlite3
import time
import threading
import websocket
import urllib3
from datetime import datetime, timezone, timedelta
import yfinance as yf
import pandas as pd

# Configuration
SYMBOLS = ["NVDA", "MSFT", "AAPL", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "NFLX", "COST"]
DB_PATH = "database/realtime_market_data.db"
BASE_URL = "https://localhost:15000/v1/api"
WS_URL = "wss://localhost:15000/v1/api/ws"

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class IBKRDataCollector:
    def __init__(self):
        self.session = requests.Session()
        self.session.verify = False
        self.contract_ids = {}
        self.ws = None
        self.session_id = None
        self.running = True
        self.initial_intraday_loaded = False  # Track if we've done initial full load
        
    def check_auth(self):
        """Check authentication and get session"""
        try:
            response = self.session.get(f"{BASE_URL}/iserver/auth/status", timeout=10)
            if response.status_code == 200 and response.json().get('authenticated'):
                # Get session ID
                tickle = self.session.get(f"{BASE_URL}/tickle", timeout=10)
                if tickle.status_code == 200:
                    self.session_id = tickle.json().get('session')
                    print(f"✅ Authenticated with session: {self.session_id[:16]}...")
                    return True
        except Exception as e:
            print(f"❌ Auth failed: {e}")
        return False
    
    def get_contract_ids(self):
        """Get contract IDs for symbols"""
        for symbol in SYMBOLS:
            try:
                response = self.session.get(
                    f"{BASE_URL}/trsrv/stocks",
                    params={'symbols': symbol},
                    timeout=10
                )
                if response.status_code == 200:
                    data = response.json()
                    if symbol in data and data[symbol]:
                        self.contract_ids[symbol] = data[symbol][0]['contracts'][0]['conid']
                        print(f"✅ {symbol}: Contract ID {self.contract_ids[symbol]}")
                time.sleep(0.2)  # Rate limiting
            except Exception as e:
                print(f"❌ Error getting contract for {symbol}: {e}")
    
    def fetch_realtime_data(self):
        """Fetch real-time data using snapshot API"""
        if not self.contract_ids:
            return
            
        for symbol, contract_id in self.contract_ids.items():
            try:
                response = self.session.get(
                    f"{BASE_URL}/iserver/marketdata/snapshot",
                    params={
                        'conids': str(contract_id),
                        'fields': '31,70,71,84,85,86,88,87,7059',
                        'since': '0'
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data and isinstance(data, list) and len(data) > 0:
                        item = data[0]
                        self.store_realtime_data(symbol, contract_id, item)
                        
            except Exception as e:
                print(f"❌ Error fetching realtime data for {symbol}: {e}")
            
            time.sleep(0.1)  # Small delay between symbols
    
    def store_realtime_data(self, symbol, contract_id, data):
        """Store real-time data in database"""
        try:
            # Extract server timestamp
            server_timestamp_epoch = data.get('_updated', 0) / 1000  # Convert ms to seconds
            received_timestamp_epoch = time.time()
            
            # Extract market data
            last_price = self.safe_float(data.get('31'))
            bid_price = self.safe_float(data.get('84'))
            ask_price = self.safe_float(data.get('86'))
            bid_size = self.safe_int(data.get('85'))
            ask_size = self.safe_int(data.get('88'))
            volume = self.safe_int(data.get('87'))
            high_price = self.safe_float(data.get('70'))
            low_price = self.safe_float(data.get('71'))
            close_price = self.safe_float(data.get('7059'))
            
            # Store in database
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO realtime_summary (
                    symbol, contract_id, last_update_server_epoch, last_update_received_epoch,
                    last_price, bid_price, ask_price, bid_size, ask_size,
                    volume, high_price, low_price, close_price
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol, contract_id, server_timestamp_epoch, received_timestamp_epoch,
                last_price, bid_price, ask_price, bid_size, ask_size,
                volume, high_price, low_price, close_price
            ))
            
            conn.commit()
            conn.close()
            
            if last_price:
                print(f"📊 {symbol}: ${last_price:.2f} | Vol: {volume:,}")
                
        except Exception as e:
            print(f"❌ Error storing realtime data for {symbol}: {e}")
    
    def fetch_yahoo_historical_data(self):
        """Fetch 30 days of 1-minute historical data from Yahoo Finance (one-time)"""
        print("\n📊 Fetching 7 days of historical data from Yahoo Finance...")
        
        for symbol in SYMBOLS:
            try:
                print(f"📈 Fetching Yahoo data for {symbol}...")
                
                # Calculate date range
                end_date = datetime.now(timezone.utc)
                start_date = end_date - timedelta(days=7)
                
                # Fetch data from yfinance
                ticker = yf.Ticker(symbol)
                data = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval='1m',
                    prepost=True  # Include pre/post market data
                )
                
                if data.empty:
                    print(f"⚠️ No Yahoo data for {symbol}")
                    continue
                
                # Store in database
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                
                records_added = 0
                for timestamp, row in data.iterrows():
                    try:
                        # Convert pandas timestamp to datetime
                        dt_timestamp = timestamp.to_pydatetime() if hasattr(timestamp, 'to_pydatetime') else timestamp
                        
                        # Extract values
                        open_val = float(row['Open']) if pd.notna(row['Open']) else None
                        high_val = float(row['High']) if pd.notna(row['High']) else None
                        low_val = float(row['Low']) if pd.notna(row['Low']) else None
                        close_val = float(row['Close']) if pd.notna(row['Close']) else None
                        volume_val = int(row['Volume']) if pd.notna(row['Volume']) else None
                        
                        cursor.execute('''
                            INSERT OR REPLACE INTO historical_data 
                            (symbol, timestamp, open_price, high_price, low_price, close_price, volume)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            symbol, dt_timestamp, open_val, high_val, low_val, close_val, volume_val
                        ))
                        records_added += 1
                    except Exception as e:
                        continue
                
                conn.commit()
                conn.close()
                
                print(f"✅ {symbol}: Stored {records_added} historical records")
                time.sleep(1)  # Rate limiting for Yahoo Finance
                
            except Exception as e:
                print(f"❌ Error fetching Yahoo data for {symbol}: {e}")
        
        print("✅ Yahoo Finance historical data fetch complete\n")
    
    def fetch_intraday_data(self):
        """Fetch today's 1-minute intraday data (full load first time, then last 5 minutes)"""
        if not self.contract_ids:
            return
            
        for symbol, contract_id in self.contract_ids.items():
            try:
                # Calculate lookback period
                if not self.initial_intraday_loaded:
                    # First time: get all data from market open
                    now = datetime.now()
                    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
                    minutes_since_open = int((now - market_open).total_seconds() / 60)
                    lookback_minutes = min(minutes_since_open, 390)  # Cap at full trading day
                else:
                    # Subsequent times: just get last 5 minutes
                    lookback_minutes = 5
                
                if lookback_minutes <= 0:
                    continue  # Before market open
                
                response = self.session.get(
                    f"{BASE_URL}/iserver/marketdata/history",
                    params={
                        'conid': contract_id,
                        'period': f'{lookback_minutes}min',
                        'bar': '1min',
                        'outsideRth': False
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    bars = data.get('data', [])
                    self.store_intraday_data(symbol, contract_id, bars)
                    
            except Exception as e:
                print(f"❌ Error fetching intraday data for {symbol}: {e}")
            
            time.sleep(0.5)  # Rate limiting
        
        # Mark initial load as complete
        if not self.initial_intraday_loaded:
            self.initial_intraday_loaded = True
            print("✅ Initial intraday data load complete\n")
    
    def store_intraday_data(self, symbol, contract_id, bars):
        """Store intraday 1-minute bars in database"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            stored_count = 0
            for bar in bars:
                bar_time = datetime.fromtimestamp(bar['t'] / 1000, tz=timezone.utc)
                
                cursor.execute('''
                    INSERT OR IGNORE INTO intraday_minute_data 
                    (symbol, contract_id, bar_time, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol, contract_id, bar_time,
                    bar.get('o'), bar.get('h'), bar.get('l'), bar.get('c'), bar.get('v', 0)
                ))
                stored_count += 1
            
            conn.commit()
            conn.close()
            
            print(f"📈 {symbol}: Stored {stored_count} minute bars")
            
        except Exception as e:
            print(f"❌ Error storing intraday data for {symbol}: {e}")
    
    def safe_float(self, value):
        """Safely convert to float"""
        if value is None:
            return None
        try:
            value_str = str(value).replace(',', '')
            if value_str.startswith('C'):  # Handle crypto format
                value_str = value_str[1:]
            return float(value_str)
        except:
            return None
    
    def safe_int(self, value):
        """Safely convert to int"""
        if value is None:
            return None
        try:
            value_str = str(value).replace(',', '')
            if 'M' in value_str:
                return int(float(value_str.replace('M', '')) * 1000000)
            elif 'K' in value_str:
                return int(float(value_str.replace('K', '')) * 1000)
            return int(float(value_str))
        except:
            return None
    
    def periodic_realtime_fetch(self):
        """Fetch real-time data every 10 seconds"""
        while self.running:
            try:
                print("\n🔄 Fetching real-time data...")
                self.fetch_realtime_data()
                time.sleep(10)
            except Exception as e:
                print(f"❌ Error in realtime fetch: {e}")
                time.sleep(10)
    
    def periodic_intraday_fetch(self):
        """Fetch intraday data every 30 seconds"""
        while self.running:
            try:
                if self.initial_intraday_loaded:
                    print("\n📊 Fetching last 5 minutes of intraday data...")
                else:
                    print("\n📊 Fetching full intraday data from market open...")
                self.fetch_intraday_data()
                time.sleep(30)
            except Exception as e:
                print(f"❌ Error in intraday fetch: {e}")
                time.sleep(30)
    
    def start(self):
        """Start data collection"""
        print("🚀 Starting IBKR Data Collector")
        print(f"📊 Symbols: {', '.join(SYMBOLS)}")
        print(f"🗄️ Database: {DB_PATH}")
        
        # Check authentication
        if not self.check_auth():
            print("❌ Not authenticated. Please login at https://localhost:15000")
            return
        
        # Get contract IDs
        self.get_contract_ids()
        if not self.contract_ids:
            print("❌ Failed to get contract IDs")
            return
        
        # One-time Yahoo Finance historical data fetch (30 days)
        self.fetch_yahoo_historical_data()
        
        # Start periodic fetch threads
        realtime_thread = threading.Thread(target=self.periodic_realtime_fetch, daemon=True)
        intraday_thread = threading.Thread(target=self.periodic_intraday_fetch, daemon=True)
        
        realtime_thread.start()
        intraday_thread.start()
        
        print("\n✅ Data collection started. Press Ctrl+C to stop.")
        
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n👋 Shutting down...")
            self.running = False

if __name__ == "__main__":
    collector = IBKRDataCollector()
    collector.start()