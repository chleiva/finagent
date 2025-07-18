#!/usr/bin/env python3
"""
Enhanced IBKR WebSocket Real-time Data Collector
Handles rate limits, connection timeouts, and automatic reconnection
"""

import requests
import json
import sqlite3
import time
import threading
import websocket
import urllib3
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import yfinance as yf
import pandas as pd
import pytz

# Market Configuration System
MARKET_CONFIGS = {
    'US': {
        'name': 'United States',
        'symbols':
         [
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
        ],
        'api_endpoint': '/trsrv/stocks',
        'api_method': 'GET',
        'api_params': lambda symbols: {'symbols': ','.join(symbols)},
        'response_parser': lambda data, symbol: data.get(symbol, [{}])[0].get('contracts', [{}])[0].get('conid') if data.get(symbol) else None,
        'websocket_format': 'smd+{conid}',
        'market_hours': {
            'start': '09:30',
            'end': '16:00',
            'timezone': 'US/Eastern'
        }
    },
    'UK': {
        'name': 'United Kingdom (LSE)',
        'symbols': [
            "AZN.L", "SHEL.L", "HSBA.L", "ULVR.L", "BP.L", "DGE.L", "GSK.L", "BATS.L",
            "RIO.L", "REL.L", "BARC.L", "LLOY.L", "LSEG.L", "PRU.L", "NG.L", "GLEN.L",
            "RKT.L", "CPG.L", "TSCO.L", "VOD.L"
        ],
        'api_endpoint': '/iserver/secdef/search',
        'api_method': 'POST',
        'api_params': lambda symbols: [{'symbol': symbol, 'name': True} for symbol in symbols],
        'response_parser': lambda data, symbol: next((contract.get('conid') for contract in data if isinstance(contract, dict) and contract.get('symbol') == symbol), None) if isinstance(data, list) else None,
        'websocket_format': 'smd+{conid}',
        'market_hours': {
            'start': '08:00',
            'end': '16:30',
            'timezone': 'Europe/London'
        }
    },
    'EU': {
        'name': 'European Union',
        'symbols': ['ASML.AS', 'SAP.DE', 'NESN.SW', 'NOVO-B.CO', 'ROCHE.SW'],
        'api_endpoint': '/iserver/secdef/search',
        'api_method': 'POST',
        'api_params': lambda symbols: [{'symbol': symbol, 'name': True} for symbol in symbols],
        'response_parser': lambda data, symbol: next((contract.get('conid') for contract in data if isinstance(contract, dict) and contract.get('symbol') == symbol), None) if isinstance(data, list) else None,
        'websocket_format': 'smd+{conid}',
        'market_hours': {
            'start': '09:00',
            'end': '17:30',
            'timezone': 'Europe/Paris'
        }
    },
    'CRYPTO': {
        'name': 'Cryptocurrency',
        'symbols': [
            "BTC", "ETH", "SOL", "ADA", "XRP"  # IBKR format: no -USD suffix
        ],
        'api_endpoint': '/iserver/secdef/search',
        'api_method': 'POST',
        'api_params': lambda symbols: [{'symbol': symbol, 'name': True, 'secType': 'CRYPTO'} for symbol in symbols],
        'response_parser': lambda data, symbol: next((contract.get('conid') for contract in data if isinstance(contract, dict) and contract.get('symbol') == symbol), None) if isinstance(data, list) else None,
        'websocket_format': 'smd+{conid}',
        'market_hours': {
            'start': '00:00',  # 24/7 trading
            'end': '23:59',
            'timezone': 'UTC'
        }
    }
}

# Default market (can be changed via command line or environment variable)
DEFAULT_MARKET = 'US'

# Symbol mapping for Yahoo Finance historical data
# Maps IBKR symbols to Yahoo Finance symbols for historical data fetching
YAHOO_FINANCE_SYMBOL_MAP = {
    # Crypto symbols: IBKR format -> Yahoo Finance format
    'BTC': 'BTC-USD',
    'ETH': 'ETH-USD', 
    'SOL': 'SOL-USD',
    'ADA': 'ADA-USD',
    'XRP': 'XRP-USD',
    'BNB': 'BNB-USD',
    'DOGE': 'DOGE-USD',
    'DOT': 'DOT-USD',
    'AVAX': 'AVAX-USD',
    'SHIB': 'SHIB-USD',
    'MATIC': 'MATIC-USD',
    'LTC': 'LTC-USD',
    'LINK': 'LINK-USD',
    'UNI': 'UNI-USD',
    'ATOM': 'ATOM-USD',
    'XLM': 'XLM-USD',
    'ALGO': 'ALGO-USD',
    'VET': 'VET-USD',
    # Add more mappings as needed
}

def get_yahoo_finance_symbol(symbol):
    """Convert IBKR symbol to Yahoo Finance symbol using the mapping."""
    return YAHOO_FINANCE_SYMBOL_MAP.get(symbol, symbol)

class EnhancedIBWebSocketCollector:
    def __init__(self, db_path="database/realtime_market_data.db", market="US"):
        # WebSocket and API setup
        self.base_url = "https://localhost:15000/v1/api"
        self.ws_url = "wss://localhost:15000/v1/api/ws"
        self.session = requests.Session()
        
        # Disable SSL warnings
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        self.session.verify = False
        
        # Market configuration
        self.market = market.upper()
        if self.market not in MARKET_CONFIGS:
            print(f"⚠️ Unknown market '{self.market}', using default '{DEFAULT_MARKET}'")
            self.market = DEFAULT_MARKET
        
        self.market_config = MARKET_CONFIGS[self.market]
        self.symbols = self.market_config['symbols']
        
        print(f"📊 Using market: {self.market_config['name']} ({self.market})")
        print(f"📈 Symbols: {', '.join(self.symbols)}")
        
        # Database setup
        self.db_path = db_path
        self.init_realtime_database()
        
        # Connection state
        self.authenticated = False
        self.ws_connected = False
        self.session_id = None
        self.ws = None
        self.running = False
        
        # Rate limiting and connection management
        self.last_subscription_time = 0
        self.subscription_delay = 2.0  # 2 seconds between subscriptions
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 30  # 30 seconds between reconnection attempts
        self.heartbeat_interval = 60  # Send heartbeat every 60 seconds
        self.last_heartbeat = 0

        self.contract_ids = {}
        self.subscribed_contracts = set()
        
        # Data tracking - Fixed type handling
        self.last_prices = defaultdict(dict)
        self.data_stats = {}
        self.excluded_symbols = defaultdict(int)  # Track symbols excluded due to missing server timestamps
        
        # Volume tracking for per-minute calculation
        self.last_daily_volumes = {}
        self.last_volume_timestamps = {}  # Track timestamp of last volume update
        
        # Connection tracking
        self.connection_attempts = 0
        self.last_connection_time = None
        self.total_reconnects = 0
        
        # Market hours validation
        self.enforce_market_hours = True  # Only collect during trading hours
        self.market_timezone = pytz.timezone('America/New_York')
        self.market_open_time = (9, 30)  # 9:30 AM
        self.market_close_time = (16, 0)  # 4:00 PM
        
    def is_market_open(self, check_time=None):
        """Check if the market is currently open (9:30 AM - 4:00 PM ET, Mon-Fri)"""
        if check_time is None:
            check_time = datetime.now(self.market_timezone)
        elif check_time.tzinfo is None:
            # If naive datetime, assume it's UTC and convert to market timezone
            check_time = check_time.replace(tzinfo=timezone.utc).astimezone(self.market_timezone)
        elif check_time.tzinfo != self.market_timezone:
            # Convert to market timezone
            check_time = check_time.astimezone(self.market_timezone)
        
        # Check if it's a weekday (Monday=0, Sunday=6)
        if check_time.weekday() >= 5:  # Saturday (5) or Sunday (6)
            return False
        
        # Check if within market hours
        market_open = check_time.replace(hour=self.market_open_time[0], minute=self.market_open_time[1], second=0, microsecond=0)
        market_close = check_time.replace(hour=self.market_close_time[0], minute=self.market_close_time[1], second=0, microsecond=0)
        
        return market_open <= check_time <= market_close
    
    def validate_ibkr_timestamp(self, server_timestamp_epoch):
        """Validate that IBKR timestamp is reasonable and during trading hours"""
        if not server_timestamp_epoch:
            return False, "No server timestamp provided"
        
        try:
            # Convert epoch to UTC datetime
            server_time_utc = datetime.fromtimestamp(server_timestamp_epoch, tz=timezone.utc)
            
            # Check if timestamp is reasonable (not too far in past/future)
            now_utc = datetime.now(timezone.utc)
            time_diff = abs((now_utc - server_time_utc).total_seconds())
            
            if time_diff > 86400:  # More than 24 hours difference
                return False, f"Timestamp too old/future: {server_time_utc} vs now {now_utc}"
            
            # Check if during market hours (if enforcement is enabled)
            if self.enforce_market_hours:
                if not self.is_market_open(server_time_utc):
                    return False, f"Outside market hours: {server_time_utc.astimezone(self.market_timezone)}"
            
            return True, "Valid"
            
        except Exception as e:
            return False, f"Invalid timestamp: {e}"
    
    def init_realtime_database(self):
        """Initialize enhanced SQLite database for real-time data"""
        conn = sqlite3.connect(self.db_path)
        
        # Fix SQLite datetime handling
        sqlite3.register_adapter(datetime, lambda dt: dt.isoformat())
        sqlite3.register_converter("DATETIME", lambda s: datetime.fromisoformat(s.decode()))
        
        cursor = conn.cursor()
        
        # Check if tables exist and have correct schema
        cursor.execute("PRAGMA table_info(realtime_ticks)")
        existing_columns = [col[1] for col in cursor.fetchall()]
        
        # FIXED: Don't return early - create tables regardless of schema state
        if existing_columns and 'server_timestamp_epoch' not in existing_columns:
            print("⚠️ Database schema is outdated. Recreating tables with new schema...")
            # Drop old tables to recreate with new schema
            cursor.execute("DROP TABLE IF EXISTS realtime_ticks")
            cursor.execute("DROP TABLE IF EXISTS realtime_summary")
            cursor.execute("DROP TABLE IF EXISTS ws_connection_log")
            cursor.execute("DROP TABLE IF EXISTS historical_data")
        elif not existing_columns:
            print("📁 Creating new database tables...")
        else:
            print("✅ Database schema is up to date")
        
        # Enhanced real-time market data table with epoch timestamps
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS realtime_ticks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                contract_id INTEGER,
                server_timestamp_epoch REAL NOT NULL,  -- UTC epoch timestamp in seconds
                received_timestamp_epoch REAL NOT NULL,  -- UTC epoch timestamp in seconds
                tick_type TEXT NOT NULL,
                price REAL,
                size INTEGER,
                bid_price REAL,
                ask_price REAL,
                bid_size INTEGER,
                ask_size INTEGER,
                last_price REAL,
                volume INTEGER,
                high_price REAL,
                low_price REAL,
                close_price REAL,
                data_source TEXT DEFAULT 'websocket',
                raw_data TEXT
            )
        ''')
        
        # Create indexes separately
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_symbol_server_timestamp 
            ON realtime_ticks(symbol, server_timestamp_epoch)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_contract_server_timestamp 
            ON realtime_ticks(contract_id, server_timestamp_epoch)
        ''')
        
        # Real-time summary table (latest values per symbol) with epoch timestamps
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS realtime_summary (
                symbol TEXT PRIMARY KEY,
                contract_id INTEGER,
                last_update_server_epoch REAL,  -- UTC epoch timestamp in seconds
                last_update_received_epoch REAL,  -- UTC epoch timestamp in seconds
                last_price REAL,
                bid_price REAL,
                ask_price REAL,
                bid_size INTEGER,
                ask_size INTEGER,
                volume INTEGER,
                high_price REAL,
                low_price REAL,
                close_price REAL,
                total_ticks INTEGER DEFAULT 0,
                data_quality TEXT DEFAULT 'fresh'
            )
        ''')
        
        # Enhanced connection log
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ws_connection_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                event_type TEXT,
                message TEXT,
                session_id TEXT,
                reconnect_attempt INTEGER DEFAULT 0
            )
        ''')
        
        # Historical data table for yfinance data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS historical_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                open_price REAL,
                high_price REAL,
                low_price REAL,
                close_price REAL,
                volume INTEGER,
                data_source TEXT DEFAULT 'yfinance',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp)
            )
        ''')
        
        # Create indexes for historical data
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_historical_symbol_timestamp 
            ON historical_data(symbol, timestamp)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_historical_timestamp 
            ON historical_data(timestamp)
        ''')
        
        # --- New tables for clean separation ---
        # Intraday minute bars (from IBKR WebSocket and REST)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS intraday_minute_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                contract_id INTEGER,
                bar_time DATETIME NOT NULL,  -- UTC time for the minute bar
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                data_source TEXT DEFAULT 'ibkr',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, bar_time)
            )
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_intraday_symbol_time 
            ON intraday_minute_data(symbol, bar_time)
        ''')
        # Daily summary bars (from yfinance)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_summary_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                bar_date DATE NOT NULL,  -- Date for the daily bar
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                data_source TEXT DEFAULT 'yfinance',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, bar_date)
            )
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_daily_symbol_date 
            ON daily_summary_data(symbol, bar_date)
        ''')
        
        conn.commit()
        conn.close()
        print(f"📁 Enhanced real-time database initialized: {self.db_path}")

    def clear_realtime_data(self):
        """Clear all real-time data to ensure no corrupted timestamps remain"""
        try:
            print("🗑️ Clearing all real-time data to ensure timestamp integrity...")
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Clear real-time tables (preserve historical data and connection logs)
            cursor.execute("DELETE FROM realtime_ticks")
            ticks_deleted = cursor.rowcount
            
            cursor.execute("DELETE FROM realtime_summary") 
            summary_deleted = cursor.rowcount
            
            # Optionally clear connection logs if they might contain corrupted data
            cursor.execute("DELETE FROM ws_connection_log")
            logs_deleted = cursor.rowcount
            
            # Keep historical data since it comes from yfinance, not IBKR timestamps
            # cursor.execute("DELETE FROM historical_data")  # Commented out - keep historical data
            
            conn.commit()
            conn.close()
            
            print(f"✅ Database cleaned:")
            print(f"   🗑️ Deleted {ticks_deleted} real-time ticks")
            print(f"   🗑️ Deleted {summary_deleted} summary records")  
            print(f"   🗑️ Deleted {logs_deleted} connection log entries")
            print(f"   💾 Preserved historical data (yfinance)")
            print(f"   🔄 Fresh collection will repopulate with accurate timestamps")
            
            return True
            
        except Exception as e:
            print(f"❌ Error clearing database: {e}")
            return False

    def check_auth(self):
        """Check authentication and get session with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"🔐 Checking authentication (attempt {attempt + 1}/{max_retries})...")
                response = self.session.get(f"{self.base_url}/iserver/auth/status", timeout=15)
                if response.status_code == 200:
                    status = response.json()
                    self.authenticated = status.get('authenticated', False)
                    
                    if self.authenticated:
                        # Get session for WebSocket
                        print("📡 Getting session ID...")
                        tickle_response = self.session.get(f"{self.base_url}/tickle", timeout=15)
                        if tickle_response.status_code == 200:
                            tickle_data = tickle_response.json()
                            self.session_id = tickle_data.get('session')
                            print(f"✅ Authenticated! Session: {self.session_id[:16]}...")
                            return True
                        else:
                            print(f"⚠️ Failed to get session: {tickle_response.status_code}")
                    else:
                        print("❌ Not authenticated. Please login at https://localhost:15000")
                        return False
                
            except requests.exceptions.Timeout:
                print(f"⏰ Authentication timeout (attempt {attempt + 1}) - IBKR Gateway not responding")
                if attempt < max_retries - 1:
                    time.sleep(5)
            except requests.exceptions.ConnectionError:
                print(f"🔌 Connection failed (attempt {attempt + 1}) - Cannot reach IBKR Gateway")
                if attempt < max_retries - 1:
                    time.sleep(5)
            except Exception as e:
                print(f"❌ Auth check failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
        
        return False
    
    def get_contract_ids(self):
        """Get contract IDs for symbols with rate limiting"""
        try:
            print("📋 Getting contract IDs...")
            
            # Process symbols in smaller batches to avoid rate limits
            batch_size = 4
            for i in range(0, len(self.symbols), batch_size):
                batch = self.symbols[i:i + batch_size]
                
                for symbol in batch:
                    try:
                        if self.market == 'CRYPTO':
                            # Enhanced crypto symbol search with multiple formats
                            contract_found = False
                            
                            # Try different crypto symbol formats for IBKR
                            crypto_formats = [
                                symbol,           # BTC
                                f"{symbol}USD",   # BTCUSD  
                                f"{symbol}.USD",  # BTC.USD
                                f"{symbol}/USD"   # BTC/USD
                            ]
                            
                            for crypto_symbol in crypto_formats:
                                print(f"🔍 Trying crypto symbol format: {crypto_symbol}")
                                
                                # Method 1: Standard secdef search with CRYPTO secType
                                response = self.session.post(
                                    f"{self.base_url}/iserver/secdef/search",
                                    json={
                                        'symbol': crypto_symbol, 
                                        'name': True, 
                                        'secType': 'CRYPTO'
                                    },
                                    timeout=15
                                )
                                
                                if response.status_code == 200:
                                    data = response.json()
                                    print(f"🔍 CRYPTO search response for {crypto_symbol}: {data}")
                                    
                                    if isinstance(data, list) and len(data) > 0:
                                        # Look for any crypto contract
                                        for contract in data:
                                            if isinstance(contract, dict) and contract.get('conid'):
                                                # Accept any crypto contract found
                                                self.contract_ids[symbol] = contract['conid']
                                                print(f"✅ {symbol}: Found crypto contract ID {self.contract_ids[symbol]} (format: {crypto_symbol})")
                                                print(f"   Contract details: {contract}")
                                                contract_found = True
                                                break
                                        
                                        if contract_found:
                                            break
                                
                                time.sleep(0.3)  # Brief delay between format attempts
                            
                            # If standard search fails, try alternative method for crypto
                            if not contract_found:
                                print(f"⚠️ Standard crypto search failed for {symbol}, trying alternative method...")
                                
                                # Method 2: General search without secType restriction
                                response = self.session.post(
                                    f"{self.base_url}/iserver/secdef/search",
                                    json={'symbol': symbol, 'name': True},
                                    timeout=15
                                )
                                
                                if response.status_code == 200:
                                    data = response.json()
                                    print(f"🔍 General search response for {symbol}: {data}")
                                    
                                    if isinstance(data, list) and len(data) > 0:
                                        for contract in data:
                                            if isinstance(contract, dict):
                                                # Look for crypto-related contracts
                                                contract_symbol = contract.get('symbol', '')
                                                sec_type = contract.get('secType', '')
                                                
                                                if (symbol.upper() in contract_symbol.upper() or 
                                                    sec_type == 'CRYPTO' or
                                                    'USD' in contract_symbol):
                                                    self.contract_ids[symbol] = contract['conid']
                                                    print(f"✅ {symbol}: Found alternative contract ID {self.contract_ids[symbol]}")
                                                    print(f"   Contract details: {contract}")
                                                    contract_found = True
                                                    break
                            
                            if not contract_found:
                                print(f"❌ Could not find crypto contract for {symbol}")
                                print(f"💡 Try checking IBKR permissions for crypto trading")
                        
                        elif symbol.endswith('.L'):
                            # For LSE symbols, use secdef search
                            response = self.session.post(
                                f"{self.base_url}/iserver/secdef/search",
                                json={'symbol': symbol, 'name': True},
                                timeout=15
                            )
                            
                            if response.status_code == 200:
                                data = response.json()
                                print(f"🔍 LSE search response for {symbol}: {data}")
                                
                                if isinstance(data, list) and len(data) > 0:
                                    for contract in data:
                                        if isinstance(contract, dict) and contract.get('symbol') == symbol:
                                            self.contract_ids[symbol] = contract['conid']
                                            print(f"📋 {symbol}: Contract ID {self.contract_ids[symbol]}")
                                            break
                                    else:
                                        print(f"⚠️ No exact match found for {symbol} in LSE response")
                                        print(f"⚠️ Available contracts: {[c.get('symbol') for c in data if isinstance(c, dict)]}")
                                else:
                                    print(f"⚠️ Empty LSE response for {symbol}")
                        else:
                            # For US symbols, use trsrv/stocks
                            response = self.session.get(
                                f"{self.base_url}/trsrv/stocks",
                                params={'symbols': symbol},
                                timeout=15
                            )
                            
                            if response.status_code == 200:
                                data = response.json()
                                print(f"🔍 US stocks response for {symbol}: {data}")
                                
                                if symbol in data and data[symbol]:
                                    self.contract_ids[symbol] = data[symbol][0]['contracts'][0]['conid']
                                    print(f"📋 {symbol}: Contract ID {self.contract_ids[symbol]}")
                        
                        # Rate limiting between individual symbols
                        time.sleep(0.5)
                        
                    except Exception as e:
                        print(f"❌ Error getting contract ID for {symbol}: {e}")
                        continue
                
                # Rate limiting between batches
                if i + batch_size < len(self.symbols):
                    time.sleep(1)
            
            print(f"📋 Found {len(self.contract_ids)} contract IDs out of {len(self.symbols)} symbols")
            return len(self.contract_ids) > 0
                
        except Exception as e:
            print(f"❌ Error getting contract IDs: {e}")
            return False
    
    def log_ws_event(self, event_type, message):
        """Log WebSocket events to database"""
        try:
            conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO ws_connection_log (event_type, message, session_id, reconnect_attempt)
                VALUES (?, ?, ?, ?)
            ''', (event_type, message, self.session_id, self.connection_attempts))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"⚠️ Failed to log event: {e}")
    
    def on_ws_open(self, ws):
        """WebSocket connection opened"""
        print("🔗 WebSocket connected!")
        self.ws_connected = True
        self.connection_attempts = 0
        self.last_connection_time = datetime.now()
        self.log_ws_event("CONNECT", "WebSocket connection established")
        
        # Send session authentication - CRITICAL for receiving data
        if self.session_id:
            auth_message = {"session": self.session_id}
            try:
                ws.send(json.dumps(auth_message))
                print(f"🔐 Sent authentication with session: {self.session_id[:16]}...")
            except Exception as e:
                print(f"❌ Failed to send authentication: {e}")
        else:
            print("❌ No session ID available for authentication")
    
    def on_ws_message(self, ws, message):
        """Handle WebSocket messages"""
        try:
            data = json.loads(message)
            
            # Handle authentication response
            if data.get('topic') == 'sts':
                if data.get('args', {}).get('authenticated'):
                    print("✅ WebSocket authenticated!")
                    self.log_ws_event("AUTHENTICATED", "WebSocket authentication successful")
                    
                    # Start subscribing to market data
                    threading.Thread(target=self.subscribe_to_market_data, daemon=True).start()
                else:
                    print("❌ WebSocket authentication failed")
                    self.log_ws_event("AUTH_FAILED", "WebSocket authentication failed")
            
            # Handle system messages
            elif data.get('topic') == 'system':
                print(f"🖥️ System message: {data}")
                self.log_ws_event("SYSTEM", str(data))
            
            # Handle market data
            elif data.get('topic') == 'smd':  # Streaming market data
                self.process_market_data(data)
            
            # Handle other topics
            else:
                print(f"📨 Received: {data}")
                self.log_ws_event("MESSAGE", str(data))
                
        except json.JSONDecodeError:
            print(f"⚠️ Invalid JSON received: {message}")
        except Exception as e:
            print(f"⚠️ Error processing message: {e}")
    
    def on_ws_error(self, ws, error):
        """Handle WebSocket errors"""
        print(f"❌ WebSocket error: {error}")
        self.ws_connected = False
        self.log_ws_event("ERROR", str(error))
    
    def on_ws_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection close"""
        print(f"🔌 WebSocket closed: {close_status_code} - {close_msg}")
        self.ws_connected = False
        self.log_ws_event("CLOSE", f"Status: {close_status_code}, Message: {close_msg}")
        
        # Attempt reconnection if still running
        if self.running:
            self.schedule_reconnection()
    
    def schedule_reconnection(self):
        """Schedule reconnection attempt"""
        if self.connection_attempts < self.max_reconnect_attempts:
            self.connection_attempts += 1
            delay = self.reconnect_delay * self.connection_attempts  # Exponential backoff
            print(f"🔄 Scheduling reconnection attempt {self.connection_attempts}/{self.max_reconnect_attempts} in {delay} seconds...")
            
            threading.Timer(delay, self.reconnect).start()
        else:
            print("❌ Max reconnection attempts reached. Stopping collector.")
            self.running = False
    
    def reconnect(self):
        """Attempt to reconnect WebSocket"""
        if not self.running:
            return
            
        print(f"🔄 Attempting reconnection {self.connection_attempts}/{self.max_reconnect_attempts}...")
        
        # Check authentication first
        if not self.check_auth():
            print("❌ Reconnection failed - not authenticated")
            self.schedule_reconnection()
            return
        
        # Connect WebSocket
        self.connect_websocket()
        self.total_reconnects += 1
    
    def subscribe_to_market_data(self):
        """Subscribe to market data using the working approach from old code"""
        if not self.ws_connected or not self.running:
            return
        
        print("📡 Starting market data subscriptions...")
        
        # Wait a moment for authentication to complete
        time.sleep(2)
        
        for symbol, contract_id in self.contract_ids.items():
            try:
                # Use the snapshot endpoint to initialize streaming (WORKING APPROACH)
                response = self.session.get(
                    f"{self.base_url}/iserver/marketdata/snapshot",
                    params={
                        'conids': str(contract_id),
                        'fields': '31,70,71,84,85,86,88,87,7059',  # Comprehensive field list
                        'since': '0',  # Force fresh data, not cached
                        'format': 'json'  # Explicit format
                    },
                    timeout=10,
                    headers={
                        'Cache-Control': 'no-cache',  # Prevent caching
                        'Pragma': 'no-cache'
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"🔍 Snapshot response for {symbol} ({contract_id}): {data}")
                    
                    if data and isinstance(data, list) and len(data) > 0:
                        print(f"✅ Initialized {symbol} ({contract_id}) market data stream")
                        self.subscribed_contracts.add(contract_id)
                        self.log_ws_event("SUBSCRIBED", f"Symbol: {symbol}, Contract: {contract_id}")
                        
                        # Process initial data if available
                        item = data[0]
                        if 'conid' in item:
                            fake_ws_data = {
                                'topic': 'smd',
                                'conid': item['conid'],
                                '_updated': item.get('_updated'),
                                **{k: v for k, v in item.items() if k.isdigit()}
                            }
                            self.process_market_data(fake_ws_data)
                    else:
                        print(f"⚠️ Empty snapshot response for {symbol} ({contract_id})")
                        print(f"   This might indicate missing crypto market data permissions")
                    
                else:
                    print(f"⚠️ Failed to initialize {symbol}: {response.status_code}")
                    print(f"Response: {response.text[:200]}...")
                    
                    # For crypto, also try alternative field list
                    if self.market == 'CRYPTO':
                        print(f"🔄 Trying alternative crypto fields for {symbol}...")
                        alt_response = self.session.get(
                            f"{self.base_url}/iserver/marketdata/snapshot",
                            params={
                                'conids': str(contract_id),
                                'fields': '31,84,86',  # Just last, bid, ask for crypto
                                'since': '0',
                                'format': 'json'
                            },
                            timeout=10,
                            headers={
                                'Cache-Control': 'no-cache',
                                'Pragma': 'no-cache'
                            }
                        )
                        
                        if alt_response.status_code == 200:
                            alt_data = alt_response.json()
                            print(f"🔍 Alternative crypto response for {symbol}: {alt_data}")
                            
                            if alt_data and isinstance(alt_data, list) and len(alt_data) > 0:
                                print(f"✅ Alternative method worked for {symbol}")
                                self.subscribed_contracts.add(contract_id)
                                self.log_ws_event("SUBSCRIBED", f"Symbol: {symbol}, Contract: {contract_id} (alternative)")
                                
                                item = alt_data[0]
                                if 'conid' in item:
                                    fake_ws_data = {
                                        'topic': 'smd',
                                        'conid': item['conid'],
                                        '_updated': item.get('_updated'),
                                        **{k: v for k, v in item.items() if k.isdigit()}
                                    }
                                    self.process_market_data(fake_ws_data)
                        else:
                            print(f"❌ Alternative method also failed for {symbol}: {alt_response.status_code}")
                            print(f"💡 Possible causes:")
                            print(f"   • Missing crypto market data permissions in IBKR")
                            print(f"   • Crypto contracts require different subscription method")
                            print(f"   • IBKR account doesn't support real-time crypto data")
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                print(f"❌ Error subscribing to {symbol}: {e}")
        
        # Now send WebSocket subscription message for streaming updates
        if self.subscribed_contracts and self.ws:
            try:
                # IBKR WebSocket requires specific format for streaming market data
                for contract_id in self.subscribed_contracts:
                    # Method 1: Try individual subscriptions
                    ws_subscription = f"smd+{contract_id}"
                    self.ws.send(ws_subscription)
                    time.sleep(0.1)
                
                print(f"📡 Sent WebSocket streaming subscriptions for {len(self.subscribed_contracts)} contracts")
                
                # Note: IBKR only supports individual string subscriptions, not JSON format
                # Removed JSON subscription that was causing "Topic unknown" errors
                
            except Exception as e:
                print(f"❌ Error sending WebSocket subscription: {e}")
    
    def send_heartbeat(self):
        """Send heartbeat to keep connection alive"""
        if self.ws and self.ws_connected and self.running:
            try:
                current_time = time.time()
                if current_time - self.last_heartbeat > self.heartbeat_interval:
                    # Send a ping or keep-alive message
                    self.ws.send("ping")
                    self.last_heartbeat = current_time
                    print("💓 Heartbeat sent")
            except Exception as e:
                print(f"⚠️ Heartbeat failed: {e}")
    
    def periodic_data_refresh(self):
        """Periodically refresh market data since IBKR WebSocket doesn't provide true streaming"""
        print("🔄 Starting periodic data refresh (every 30 seconds)...")
        
        while self.running:
            try:
                time.sleep(30)  # Refresh every 30 seconds
                
                if not self.running:
                    break
                
                print("🔄 Refreshing market data...")
                
                # Refresh data for each symbol
                for symbol, contract_id in self.contract_ids.items():
                    try:
                        # Get fresh snapshot data
                        response = self.session.get(
                            f"{self.base_url}/iserver/marketdata/snapshot",
                            params={
                                'conids': str(contract_id),
                                'fields': '31,70,71,84,85,86,88,87,7059',
                                'since': '0',  # Force fresh data
                                'format': 'json'
                            },
                            timeout=10,
                            headers={
                                'Cache-Control': 'no-cache',
                                'Pragma': 'no-cache'
                            }
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            if data and isinstance(data, list) and len(data) > 0:
                                item = data[0]
                                if 'conid' in item:
                                    # Process as WebSocket data
                                    fake_ws_data = {
                                        'topic': 'smd',
                                        'conid': item['conid'],
                                        '_updated': item.get('_updated'),
                                        **{k: v for k, v in item.items() if k.isdigit()}
                                    }
                                    self.process_market_data(fake_ws_data)
                        
                        time.sleep(0.2)  # Small delay between symbols
                        
                    except Exception as e:
                        print(f"⚠️ Error refreshing {symbol}: {e}")
                
                print("✅ Market data refresh complete")
                
            except Exception as e:
                print(f"⚠️ Error in periodic refresh: {e}")
                time.sleep(5)  # Wait before retrying
    
    def process_market_data(self, data):
        """Process market data using the working approach from old code"""
        try:
            contract_id = data.get('conid')
            if not contract_id:
                return
            
            # Find symbol for this contract
            symbol = None
            for sym, cid in self.contract_ids.items():
                if cid == contract_id:
                    symbol = sym
                    break
            
            if not symbol:
                return
            
            # Extract server timestamp as epoch seconds (this is the key!)
            server_timestamp_epoch = None
            server_time_ms = data.get('_updated')  # IBKR server timestamp in milliseconds
            
            if server_time_ms:
                server_timestamp_epoch = server_time_ms / 1000  # Convert ms to seconds
            else:
                # NO FALLBACK! If there's no server timestamp, we cannot trust the data timing
                self.excluded_symbols[symbol] += 1
                print(f"❌ No server timestamp for {symbol}, excluding from processing (data integrity)")
                print(f"   This usually indicates: market closed, no data permissions, or stale cached data")
                print(f"   Total exclusions for {symbol}: {self.excluded_symbols[symbol]}")
                return  # Skip processing entirely - don't corrupt timing data!
            
            received_timestamp_epoch = time.time()  # Current time as epoch seconds
            
            # Validate timestamp and enforce market hours
            is_valid, reason = self.validate_ibkr_timestamp(server_timestamp_epoch)
            if not is_valid:
                print(f"⚠️ {symbol}: Invalid timestamp ({server_timestamp_epoch}) - {reason}. Excluding.")
                self.excluded_symbols[symbol] += 1
                return
            
            # Extract daily cumulative volume for per-minute calculation
            daily_cumulative_volume = self.safe_volume(data.get('87'))  # Daily total volume
            
            # Calculate per-minute volume from daily cumulative changes
            per_minute_volume = self.calculate_per_minute_volume(symbol, daily_cumulative_volume, server_timestamp_epoch)
            
            # Extract market data fields
            tick_data = {
                'last_price': self.safe_float(data.get('31')),      # Last price
                'bid_price': self.safe_float(data.get('84')),       # Bid price
                'ask_price': self.safe_float(data.get('86')),       # Ask price
                'bid_size': self.safe_int(data.get('85')),          # Bid size
                'ask_size': self.safe_int(data.get('88')),          # Ask size
                'volume': per_minute_volume,                        # Per-minute volume (calculated from daily changes)
                'daily_volume': daily_cumulative_volume,            # Store daily total for reference
                'high_price': self.safe_float(data.get('70')),      # High
                'low_price': self.safe_float(data.get('71')),       # Low
                'close_price': self.safe_float(data.get('7059'))    # Previous close
            }
            
            # Store in database
            self.store_tick_data(symbol, contract_id, server_timestamp_epoch, received_timestamp_epoch, tick_data, data)
            
            # Update statistics - Fixed type handling
            if symbol not in self.data_stats:
                self.data_stats[symbol] = {'count': 0, 'last_update': None}
            
            stats = self.data_stats[symbol]
            current_count = stats.get('count', 0)
            stats['count'] = current_count + 1
            # Store as datetime object for proper display handling
            server_timestamp_dt = datetime.fromtimestamp(server_timestamp_epoch)
            stats['last_update'] = server_timestamp_dt  # Store datetime, not string!
            
            # Display real-time update
            if tick_data['last_price']:
                age_seconds = received_timestamp_epoch - server_timestamp_epoch
                server_time_str = datetime.fromtimestamp(server_timestamp_epoch).strftime('%H:%M:%S.%f')[:-3]
                
                # Enhanced display with volume information
                volume_str = f"vol={tick_data['volume']:,}" if tick_data['volume'] is not None else "vol=N/A"
                daily_vol_str = f"daily={tick_data['daily_volume']:,}" if tick_data['daily_volume'] is not None else "daily=N/A"
                print(f"🔴 {symbol:6} | ${tick_data['last_price']:7.2f} | {volume_str} | {daily_vol_str} | Server: {server_time_str} | Age: {age_seconds:.1f}s")
            
        except Exception as e:
            print(f"⚠️ Error processing market data: {e}")
    
    def safe_float(self, value):
        """Safely convert to float - handles IBKR crypto 'C' prefix"""
        if value is None:
            return None
        try:
            # Handle IBKR crypto price format: 'C113413.50' -> '113413.50'
            value_str = str(value).replace(',', '')
            if value_str.startswith('C'):
                value_str = value_str[1:]  # Remove 'C' prefix
            return float(value_str)
        except (ValueError, TypeError):
            return None
    
    def safe_int(self, value):
        """Safely convert to int"""
        if value is None:
            return None
        try:
            return int(str(value).replace(',', ''))
        except (ValueError, TypeError):
            return None
    
    def safe_volume(self, value):
        """Safely convert volume (handles K/M notation)"""
        if value is None:
            return None
        
        try:
            value_str = str(value).replace(',', '')
            if 'M' in value_str:
                return int(float(value_str.replace('M', '')) * 1000000)
            elif 'K' in value_str:
                return int(float(value_str.replace('K', '')) * 1000)
            else:
                return int(float(value_str))
        except (ValueError, TypeError):
            return None
    
    def calculate_per_minute_volume(self, symbol, current_daily_volume, current_timestamp):
        """Calculate per-minute volume from daily cumulative volume changes"""
        if current_daily_volume is None:
            return None
        
        # Get last known daily volume for this symbol
        last_daily_volume = self.last_daily_volumes.get(symbol)
        last_timestamp = self.last_volume_timestamps.get(symbol)
        
        # If this is the first volume reading, store it and return None (no per-minute calculation possible)
        if last_daily_volume is None or last_timestamp is None:
            self.last_daily_volumes[symbol] = current_daily_volume
            self.last_volume_timestamps[symbol] = current_timestamp
            print(f"🔄 {symbol}: Initializing volume tracking - daily_volume={current_daily_volume:,}")
            return None
        
        # Calculate volume difference (should be positive as daily volume is cumulative)
        volume_difference = current_daily_volume - last_daily_volume
        
        # Calculate time difference in minutes
        time_difference_seconds = current_timestamp - last_timestamp
        time_difference_minutes = time_difference_seconds / 60.0
        
        # Update stored values
        self.last_daily_volumes[symbol] = current_daily_volume
        self.last_volume_timestamps[symbol] = current_timestamp
        
        # If volume decreased or stayed the same, return 0 (market might have reset or no new trades)
        if volume_difference <= 0:
            print(f"⚠️ {symbol}: Volume difference={volume_difference:,} (time_diff={time_difference_minutes:.1f}min) - returning 0")
            return 0
        
        # If time difference is too small or too large, return the raw difference
        if time_difference_minutes < 0.5 or time_difference_minutes > 10:
            print(f"🕐 {symbol}: Time difference={time_difference_minutes:.1f}min out of range - using raw volume difference={volume_difference:,}")
            return volume_difference
        
        # For normal time differences, return the volume difference as per-minute volume
        per_minute_volume = volume_difference
        print(f"✅ {symbol}: Per-minute volume={per_minute_volume:,} (daily_change={volume_difference:,}, time_diff={time_difference_minutes:.1f}min)")
        
        return per_minute_volume
    
    def store_tick_data(self, symbol, contract_id, server_timestamp, received_timestamp, tick_data, raw_data):
        """Store tick data in database with server timestamp as epoch"""
        try:
            conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
            cursor = conn.cursor()
            
            # Convert timestamps to epoch if they aren't already
            server_timestamp_epoch = server_timestamp.timestamp() if isinstance(server_timestamp, datetime) else float(server_timestamp)
            received_timestamp_epoch = received_timestamp.timestamp() if isinstance(received_timestamp, datetime) else float(received_timestamp)
            
            try:
                # Add daily_volume to raw_data for storage
                enhanced_raw_data = raw_data.copy()
                enhanced_raw_data['daily_volume_processed'] = tick_data['daily_volume']
                enhanced_raw_data['per_minute_volume_calculated'] = tick_data['volume']
                
                # Store detailed tick with epoch timestamps
                cursor.execute('''
                    INSERT INTO realtime_ticks (
                        symbol, contract_id, server_timestamp_epoch, received_timestamp_epoch,
                        tick_type, last_price, bid_price, ask_price, bid_size, ask_size,
                        volume, high_price, low_price, close_price, raw_data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol, contract_id, server_timestamp_epoch, received_timestamp_epoch,
                    'market_data', tick_data['last_price'], tick_data['bid_price'], 
                    tick_data['ask_price'], tick_data['bid_size'], tick_data['ask_size'],
                    tick_data['volume'], tick_data['high_price'], tick_data['low_price'],
                    tick_data['close_price'], json.dumps(enhanced_raw_data)
                ))
                
                # Calculate age in seconds directly from epoch timestamps
                age_seconds = received_timestamp_epoch - server_timestamp_epoch
                data_quality = 'fresh' if age_seconds < 10 else 'stale' if age_seconds < 120 else 'old'
                
                # Update summary table with epoch timestamps
                cursor.execute('''
                    INSERT OR REPLACE INTO realtime_summary (
                        symbol, contract_id, last_update_server_epoch, last_update_received_epoch,
                        last_price, bid_price, ask_price, bid_size, ask_size,
                        volume, high_price, low_price, close_price, 
                        total_ticks, data_quality
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                             COALESCE((SELECT total_ticks FROM realtime_summary WHERE symbol = ?) + 1, 1), ?)
                ''', (
                    symbol, contract_id, server_timestamp_epoch, received_timestamp_epoch,
                    tick_data['last_price'], tick_data['bid_price'], tick_data['ask_price'],
                    tick_data['bid_size'], tick_data['ask_size'], tick_data['volume'],
                    tick_data['high_price'], tick_data['low_price'], tick_data['close_price'],
                    symbol, data_quality
                ))
                
            except sqlite3.OperationalError as e:
                if "no column named server_timestamp_epoch" in str(e):
                    print(f"⚠️ Database schema error detected. Reinitializing database...")
                    conn.close()
                    # Reinitialize the database
                    self.init_realtime_database()
                    # Retry the operation
                    return self.store_tick_data(symbol, contract_id, server_timestamp, received_timestamp, tick_data, raw_data)
                else:
                    raise e
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"⚠️ Database error for {symbol}: {e}")
    
    def display_latest_prices_table(self):
        """Display a table with latest prices and server timestamp latency"""
        try:
            conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
            cursor = conn.cursor()
            
            # Get latest data for each symbol (using epoch timestamps)
            cursor.execute('''
                SELECT 
                    symbol,
                    last_price,
                    bid_price,
                    ask_price,
                    last_update_server_epoch,
                    last_update_received_epoch,
                    data_quality,
                    total_ticks
                FROM realtime_summary 
                WHERE last_price IS NOT NULL
                ORDER BY symbol
            ''')
            
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                print("📊 Latest Prices: No data available")
                return
            
            # Calculate current time for latency
            now = datetime.now()
            
            # Print table header
            print("\n📊 Latest Prices & Latency:")
            print("=" * 85)
            print(f"{'Symbol':<8} {'Last':<10} {'Bid':<10} {'Ask':<10} {'Server Time':<12} {'Latency':<8} {'Quality':<8} {'Ticks':<6}")
            print("-" * 85)
            
            for row in rows:
                symbol, last_price, bid_price, ask_price, server_time_epoch, received_time_epoch, quality, ticks = row
                
                if server_time_epoch and received_time_epoch:
                    # Ensure we have numeric values for timestamps
                    server_time_epoch = float(server_time_epoch)
                    received_time_epoch = float(received_time_epoch)
                    
                    # Calculate latency in milliseconds directly from epoch timestamps
                    latency_ms = int((received_time_epoch - server_time_epoch) * 1000)
                    latency_str = f"{latency_ms}ms"
                    
                    # Convert epoch to datetime for display
                    server_time = datetime.fromtimestamp(server_time_epoch)
                    server_time_str = server_time.strftime('%H:%M:%S.%f')[:-3]  # Remove microseconds, keep milliseconds
                    
                    # Calculate quality based on actual latency (not stored quality)
                    if latency_ms < 10000:  # 10 seconds in ms
                        quality_emoji = "🟢"
                        quality_display = "🟢 fresh"
                    elif latency_ms < 120000:  # 120 seconds in ms
                        quality_emoji = "🟡"
                        quality_display = "🟡 stale"
                    else:
                        quality_emoji = "🔴"
                        quality_display = "🔴 old"
                    
                    # Format prices
                    last_str = f"${last_price:.2f}" if last_price else "N/A"
                    bid_str = f"${bid_price:.2f}" if bid_price else "N/A"
                    ask_str = f"${ask_price:.2f}" if ask_price else "N/A"
                    
                    print(f"{symbol:<8} {last_str:<10} {bid_str:<10} {ask_str:<10} {server_time_str:<12} {latency_str:<8} {quality_display:<8} {ticks:<6}")
                else:
                    print(f"{symbol:<8} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<12} {'N/A':<8} {'N/A':<8} {ticks:<6}")
            
            print("=" * 85)
            
        except Exception as e:
            print(f"⚠️ Error displaying price table: {e}")
    
    def check_historical_data_coverage(self):
        """Check if we have 7 days of historical data for all symbols"""
        try:
            conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
            cursor = conn.cursor()
            
            # Use timezone-aware datetime for comparison
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=7)
            
            for symbol in self.symbols:
                cursor.execute('''
                    SELECT MIN(timestamp), MAX(timestamp), COUNT(*) 
                    FROM historical_data 
                    WHERE symbol = ?
                ''', (symbol,))
                
                result = cursor.fetchone()
                if result and result[0] and result[1]:
                    earliest, latest, count = result
                    # Convert string to datetime if needed
                    if isinstance(earliest, str):
                        earliest = datetime.fromisoformat(earliest)
                    if isinstance(latest, str):
                        latest = datetime.fromisoformat(latest)
                    
                    # Ensure both datetimes are timezone-aware for comparison
                    if earliest.tzinfo is None:
                        earliest = earliest.replace(tzinfo=timezone.utc)
                    if latest.tzinfo is None:
                        latest = latest.replace(tzinfo=timezone.utc)
                    
                    if earliest > cutoff_date:
                        print(f"⚠️ {symbol}: Missing data before {earliest.strftime('%Y-%m-%d')}")
                        return False
                    print(f"✅ {symbol}: {earliest.strftime('%Y-%m-%d')} to {latest.strftime('%Y-%m-%d')} ({count} records)")
                else:
                    print(f"❌ {symbol}: No historical data found")
                    return False
            
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Error checking historical data: {e}")
            return False
    
    def fetch_historical_data(self, symbol, days=7):
        """Fetch historical 1-minute data from yfinance"""
        # Map IBKR symbol to Yahoo Finance symbol
        yahoo_symbol = get_yahoo_finance_symbol(symbol)
        
        try:
            print(f"📊 Fetching {days} days of 1-minute data for {symbol} (Yahoo: {yahoo_symbol})...")
            
            # Calculate date range with timezone-aware datetimes
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)
            
            # Fetch data from yfinance (1-minute data to match training data and IBKR)
            ticker = yf.Ticker(yahoo_symbol)
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval='1m',
                prepost=True  # Include pre/post market data
            )
                
            if data.empty:
                print(f"⚠️ No data received for {symbol} (Yahoo: {yahoo_symbol})")
                return False
            
            print(f"📈 {symbol}: Retrieved {len(data)} 1-minute records")
            
            # Store in database
            conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
            cursor = conn.cursor()
            
            records_added = 0
            for timestamp, row in data.iterrows():
                try:
                    # Convert pandas timestamp to datetime - Fixed type handling
                    if hasattr(timestamp, 'to_pydatetime'):
                        dt_timestamp = timestamp.to_pydatetime()
                    else:
                        dt_timestamp = timestamp
                    
                    # Fixed pandas Series handling - convert to scalar values
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
                        symbol,
                        dt_timestamp,
                        open_val,
                        high_val,
                        low_val,
                        close_val,
                        volume_val
                    ))
                    records_added += 1
                except Exception as e:
                    print(f"⚠️ Error inserting {symbol} data for {timestamp}: {e}")
                    continue
            
            conn.commit()
            conn.close()
            
            print(f"✅ {symbol}: Stored {records_added} records in database")
            return True
            
        except Exception as e:
            print(f"❌ Error fetching historical data for {symbol} (Yahoo: {yahoo_symbol}): {e}")
            return False
    
    def populate_historical_data(self):
        """Populate database with 7 days of historical data for all symbols"""
        print("🔄 Checking historical data coverage...")
        
        if self.check_historical_data_coverage():
            print("✅ Historical data coverage is complete (7 days)")
            return True
        
        print("📊 Populating historical data from yfinance...")
        print("=" * 60)
        
        success_count = 0
        for symbol in self.symbols:
            if self.fetch_historical_data(symbol, days=7):
                success_count += 1
            else:
                print(f"❌ Failed to fetch data for {symbol}")
            
            # Rate limiting between symbols
            time.sleep(1)
        
        print("=" * 60)
        print(f"📊 Historical data population complete: {success_count}/{len(self.symbols)} symbols")
        
        # Verify coverage
        if self.check_historical_data_coverage():
            print("✅ All symbols have complete 7-day historical data")
            return True
        else:
            print("⚠️ Some symbols may have incomplete historical data")
            return False
    
    def backfill_intraday_minute_data(self, from_market_open=True):
        """Backfill intraday minute data using IBKR REST API
        
        Args:
            from_market_open (bool): If True, fetch from market open (9:30 AM ET) to now.
                                   If False, fetch last 60 minutes only.
        """
        try:
            current_time = datetime.now(self.market_timezone)
            
            if from_market_open:
                # Calculate market open time for today
                market_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
                
                # If current time is before market open, use yesterday's market open
                if current_time < market_open:
                    market_open = market_open - timedelta(days=1)
                    
                # Calculate minutes from market open to now
                time_diff = current_time - market_open
                lookback_minutes = int(time_diff.total_seconds() / 60)
                
                # Cap at 390 minutes (6.5 hours = full trading day)
                lookback_minutes = min(lookback_minutes, 390)
                
                print(f"🔄 Backfilling ENTIRE trading day from market open...")
                print(f"📅 Market open: {market_open.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                print(f"🕐 Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                print(f"⏱️ Fetching {lookback_minutes} minutes of 1-minute bars...")
            else:
                lookback_minutes = 60
                print(f"🔄 Backfilling intraday minute data for the last {lookback_minutes} minutes...")
            
            success_count = 0
            total_bars = 0
            
            for symbol, contract_id in self.contract_ids.items():
                # Fetch 1-min bars from market open (or last N minutes)
                params = {
                    'conid': contract_id,
                    'period': f'{lookback_minutes}min',
                    'bar': '1min',
                    'outsideRth': False  # Only regular trading hours
                }
                response = self.session.get(f"{self.base_url}/iserver/marketdata/history", params=params)
                if response.status_code == 200:
                    data = response.json()
                    bars = data.get('data', [])
                    symbol_bars = 0
                    
                    for bar in bars:
                        try:
                            # Store each bar in the database, avoiding duplicates
                            bar_time = datetime.fromtimestamp(bar['t'] / 1000, tz=timezone.utc)
                            open_price = bar.get('o')
                            high_price = bar.get('h')
                            low_price = bar.get('l')
                            close_price = bar.get('c')
                            volume = bar.get('v', 0)  # Default to 0 if no volume
                            
                            conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
                            cursor = conn.cursor()
                            cursor.execute('''
                                INSERT OR IGNORE INTO intraday_minute_data 
                                (symbol, contract_id, bar_time, open, high, low, close, volume)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            ''', (symbol, contract_id, bar_time, open_price, high_price, low_price, close_price, volume))
                            conn.commit()
                            conn.close()
                            symbol_bars += 1
                        except Exception as e:
                            print(f"⚠️ Error storing bar for {symbol} at {bar_time}: {e}")
                    
                    print(f"✅ {symbol}: {symbol_bars} bars stored")
                    success_count += 1
                    total_bars += symbol_bars
                else:
                    print(f"❌ Failed to fetch data for {symbol}: {response.status_code} - {response.text}")
                    
                # Rate limiting between symbols to avoid IBKR limits
                time.sleep(0.5)
            
            print("=" * 60)
            print(f"📊 Intraday backfill complete: {success_count}/{len(self.symbols)} symbols")
            print(f"📈 Total 1-minute bars stored: {total_bars}")
            
            if from_market_open and total_bars > 0:
                print(f"🎯 Successfully fetched {lookback_minutes} minutes from market open")
                print(f"⏰ Time range: {market_open.strftime('%H:%M')} ET to {current_time.strftime('%H:%M')} ET")
            
            return success_count > 0
            
        except Exception as e:
            print(f"❌ Error during backfill: {e}")
            return False
    
    def connect_websocket(self):
        """Establish WebSocket connection"""
        print(f"🔗 Connecting to WebSocket: {self.ws_url}")
        
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_open=self.on_ws_open,
            on_message=self.on_ws_message,
            on_error=self.on_ws_error,
            on_close=self.on_ws_close
        )
        
        # Run WebSocket in background thread
        ws_thread = threading.Thread(target=self.ws.run_forever, kwargs={'sslopt': {"cert_reqs": 0}}, daemon=True)
        ws_thread.start()
        
        return ws_thread
    
    def periodic_yfinance_refresh(self):
        """Periodically refresh daily summary data from yfinance"""
        while self.running:
            try:
                print("🔄 Refreshing daily summary data from yfinance...")
                for symbol in self.symbols:
                    self.fetch_historical_data(symbol, days=1)  # Fetch only the latest day
                print("✅ Daily summary data refresh complete")
            except Exception as e:
                print(f"⚠️ Error during yfinance refresh: {e}")
            time.sleep(3600)  # Refresh every hour

    def aggregate_ticks_to_minute_bars(self):
        """Aggregate realtime_ticks into 1-minute bars and store in intraday_minute_data"""
        try:
            conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
            cursor = conn.cursor()
            for symbol in self.symbols:
                cursor.execute('''
                    SELECT 
                        strftime('%Y-%m-%d %H:%M:00', datetime(server_timestamp_epoch, 'unixepoch')) as bar_time,
                        MIN(server_timestamp_epoch),
                        MAX(server_timestamp_epoch),
                        MIN(last_price),
                        MAX(last_price),
                        MIN(bid_price),
                        MAX(ask_price),
                        MIN(low_price),
                        MAX(high_price),
                        SUM(volume)
                    FROM realtime_ticks
                    WHERE symbol = ?
                    GROUP BY bar_time
                ''', (symbol,))
                for row in cursor.fetchall():
                    bar_time, _, _, open_, close_, low_, high_, _, _, volume = row
                    cursor.execute('''
                        INSERT OR IGNORE INTO intraday_minute_data 
                        (symbol, bar_time, open, close, low, high, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (symbol, bar_time, open_, close_, low_, high_, volume))
            conn.commit()
            conn.close()
            print("✅ Aggregated ticks to intraday_minute_data")
        except Exception as e:
            print(f"❌ Error aggregating ticks: {e}")

    def fetch_and_store_daily_bars(self, symbol, days=60):
        """Fetch last N daily bars from yfinance and store in daily_summary_data"""
        yahoo_symbol = get_yahoo_finance_symbol(symbol)
        try:
            ticker = yf.Ticker(yahoo_symbol)
            data = ticker.history(period=f'{days}d', interval='1d')
            if not data.empty:
                for timestamp, row in data.iterrows():
                    import datetime
                    if isinstance(timestamp, (pd.Timestamp, datetime.datetime)):
                        bar_date = timestamp.date()
                    elif isinstance(timestamp, (str, int, float)):
                        bar_date = pd.to_datetime(timestamp).date()
                    else:
                        bar_date = timestamp
                    def get_scalar(val):
                        if pd.api.types.is_scalar(val):
                            return val
                        if hasattr(val, 'item'):
                            return val.item()
                        if hasattr(val, 'iloc'):
                            return val.iloc[0]
                        if hasattr(val, 'values'):
                            return val.values[0]
                        return float(val)
                    def is_valid_scalar(val):
                        return val is not None and isinstance(val, (int, float)) and not pd.isna(val)
                    open_val = get_scalar(row['Open'])
                    high_val = get_scalar(row['High'])
                    low_val = get_scalar(row['Low'])
                    close_val = get_scalar(row['Close'])
                    volume_val = get_scalar(row['Volume'])
                    open_ = float(open_val) if is_valid_scalar(open_val) else None
                    high_ = float(high_val) if is_valid_scalar(high_val) else None
                    low_ = float(low_val) if is_valid_scalar(low_val) else None
                    close_ = float(close_val) if is_valid_scalar(close_val) else None
                    volume = int(volume_val) if is_valid_scalar(volume_val) else None
                    conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT OR REPLACE INTO daily_summary_data
                        (symbol, bar_date, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (symbol, bar_date, open_, high_, low_, close_, volume))
                    conn.commit()
                    conn.close()
                print(f"✅ Stored {len(data)} daily bars for {symbol} in daily_summary_data")
            else:
                print(f"⚠️ No daily data for {symbol}")
        except Exception as e:
            print(f"❌ Error fetching/storing daily bars for {symbol}: {e}")

    def periodic_migration_tasks(self):
        """Periodically aggregate ticks and refresh daily bars into new tables"""
        while self.running:
            self.aggregate_ticks_to_minute_bars()
            for symbol in self.symbols:
                self.fetch_and_store_daily_bars(symbol, days=60)
            time.sleep(300)  # Run every 5 minutes

    def detect_and_fill_intraday_gaps(self):
        """Detect gaps in minute-by-minute data and backfill missing periods"""
        try:
            current_time = datetime.now(self.market_timezone)
            
            # Only check during market hours
            if not self.is_market_open(current_time):
                return
            
            # Calculate expected market open time for today
            market_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
            if current_time < market_open:
                market_open = market_open - timedelta(days=1)
            
            # Calculate expected number of minutes from market open to now
            time_diff = current_time - market_open
            expected_minutes = int(time_diff.total_seconds() / 60)
            
            print(f"🔍 Gap detection: Expected {expected_minutes} minutes from {market_open.strftime('%H:%M')} to {current_time.strftime('%H:%M')}")
            
            conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
            cursor = conn.cursor()
            
            gaps_found = False
            
            for symbol in self.symbols:
                # Get all minute timestamps for this symbol from today's market open
                cursor.execute('''
                    SELECT COUNT(DISTINCT strftime('%Y-%m-%d %H:%M', bar_time)) as minute_count
                    FROM intraday_minute_data 
                    WHERE symbol = ? 
                    AND datetime(bar_time) >= datetime(?)
                    AND datetime(bar_time) <= datetime(?)
                ''', (symbol, 
                      market_open.astimezone(timezone.utc).isoformat(),
                      current_time.astimezone(timezone.utc).isoformat()))
                
                actual_minutes = cursor.fetchone()[0]
                missing_minutes = expected_minutes - actual_minutes
                
                if missing_minutes > 0:
                    print(f"⚠️ {symbol}: Missing {missing_minutes} minutes of data (have {actual_minutes}/{expected_minutes})")
                    gaps_found = True
                else:
                    print(f"✅ {symbol}: Complete data ({actual_minutes} minutes)")
            
            conn.close()
            
            # If gaps found, do a complete backfill from market open
            if gaps_found:
                print("🔄 Gaps detected, performing complete backfill from market open...")
                self.backfill_intraday_minute_data(from_market_open=True)
            else:
                print("✅ No gaps detected, all symbols have complete minute data")
                
        except Exception as e:
            print(f"❌ Error in gap detection: {e}")

    def periodic_gap_detection_and_backfill(self):
        """Periodically check for gaps and backfill missing data"""
        while self.running:
            try:
                time.sleep(300)  # Check every 5 minutes
                
                if not self.running:
                    break
                    
                self.detect_and_fill_intraday_gaps()
                
            except Exception as e:
                print(f"⚠️ Error in periodic gap detection: {e}")
                time.sleep(60)  # Wait before retrying


    def start_realtime_collection(self, clear_existing_data=False):
        """Start real-time data collection via WebSocket with enhanced reliability"""
        print("🚀 Enhanced IBKR WebSocket Real-time Data Collector")
        print("=" * 60)
        print("🛡️ Features: Rate limiting, auto-reconnection, heartbeat")
        print("📊 Historical data: 7 days (1-minute) from yfinance")
        print("🔐 Timestamp integrity: No fake server timestamps")
        print("⏰ Market hours enforcement: 9:30 AM - 4:00 PM ET, Mon-Fri")
        print("=" * 60)
        
        # Check if market is currently open
        if self.enforce_market_hours:
            current_time = datetime.now(self.market_timezone)
            if not self.is_market_open(current_time):
                print(f"\n❌ MARKET IS CURRENTLY CLOSED")
                print(f"Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                print(f"Market hours: {self.market_open_time[0]:02d}:{self.market_open_time[1]:02d} - {self.market_close_time[0]:02d}:{self.market_close_time[1]:02d} ET")
                print(f"Weekend: {'Yes' if current_time.weekday() >= 5 else 'No'}")
                
                if current_time.weekday() < 5:  # Weekday
                    market_open_today = current_time.replace(hour=self.market_open_time[0], minute=self.market_open_time[1], second=0, microsecond=0)
                    if current_time < market_open_today:
                        wait_time = (market_open_today - current_time).total_seconds()
                        print(f"⏰ Market opens in {wait_time/3600:.1f} hours at {market_open_today.strftime('%H:%M ET')}")
                    else:
                        next_open = market_open_today + timedelta(days=1)
                        wait_time = (next_open - current_time).total_seconds()
                        print(f"⏰ Market opens tomorrow in {wait_time/3600:.1f} hours at {next_open.strftime('%H:%M ET')}")
                else:
                    # Weekend - find next Monday
                    days_until_monday = 7 - current_time.weekday()  # Monday = 0
                    next_monday = current_time + timedelta(days=days_until_monday)
                    next_open = next_monday.replace(hour=self.market_open_time[0], minute=self.market_open_time[1], second=0, microsecond=0)
                    wait_time = (next_open - current_time).total_seconds()
                    print(f"⏰ Market opens Monday in {wait_time/3600/24:.1f} days at {next_open.strftime('%Y-%m-%d %H:%M ET')}")
                
                print("\n💡 To collect data outside market hours, set enforce_market_hours=False")
                print("⚠️  WARNING: Outside market hours, IBKR data may be stale/cached")
                return False
        
        # Clear existing data if requested (removes any corrupted timestamp data)
        if clear_existing_data:
            print("\n🗑️ Clearing existing real-time data for timestamp integrity...")
            if not self.clear_realtime_data():
                print("❌ Failed to clear data, continuing anyway...")
                
        self.running = True
        
        # Initialize data_stats for all symbols
        for symbol in self.symbols:
            self.data_stats[symbol] = {'count': 0, 'last_update': None}
        
        # Populate historical data first
        print("\n📊 Checking and populating historical data...")
        if not self.populate_historical_data():
            print("⚠️ Historical data population had issues, but continuing...")
        
        print(f"\n🔐 Starting real-time data collection during market hours...")
        print(f"Current market time: {datetime.now(self.market_timezone).strftime('%Y-%m-%d %H:%M:%S %Z')}")
        
        # Check authentication
        if not self.check_auth():
            print("❌ Not authenticated. Please login at https://localhost:15000")
            return False
        
        # Get contract IDs
        if not self.get_contract_ids():
            print("❌ Failed to get contract IDs")
            return False
        
        # Backfill the entire trading day from market open
        print("\n📈 Fetching complete trading day from market open...")
        if not self.backfill_intraday_minute_data(from_market_open=True):
            print("⚠️ Failed to backfill trading day data, but continuing with real-time collection...")
        
        # Connect WebSocket
        ws_thread = self.connect_websocket()
        
        print(f"📡 Starting enhanced real-time data collection...")
        print(f"🗄️ Database: {self.db_path}")
        print(f"📊 Symbols: {', '.join(self.symbols)}")
        print(f"⏰ Data will only be collected during market hours (9:30 AM - 4:00 PM ET)")
        
        # Start background threads
        heartbeat_thread = threading.Thread(target=self.send_heartbeat, daemon=True)
        heartbeat_thread.start()
        
        status_thread = threading.Thread(target=self.display_latest_prices_table, daemon=True)
        status_thread.start()
        
        # Start periodic data refresh (since IBKR WebSocket is not true streaming)
        refresh_thread = threading.Thread(target=self.periodic_data_refresh, daemon=True)
        refresh_thread.start()
        
        # Start task for migrating data to new structured tables
        migration_thread = threading.Thread(target=self.periodic_migration_tasks, daemon=True)
        migration_thread.start()
        
        # Start gap detection and backfill thread
        gap_detection_thread = threading.Thread(target=self.periodic_gap_detection_and_backfill, daemon=True)
        gap_detection_thread.start()
        
        print("\n🔄 Data collection threads started. Press Ctrl+C to stop.")
        
        try:
            # Keep main thread alive and monitor market hours
            while self.running:
                time.sleep(60)  # Check every minute
                
                # If market hours enforcement is on, check if market closed
                if self.enforce_market_hours and not self.is_market_open():
                    print(f"\n⏰ Market has closed. Stopping data collection...")
                    print(f"Current time: {datetime.now(self.market_timezone).strftime('%Y-%m-%d %H:%M:%S %Z')}")
                    break
                    
        except KeyboardInterrupt:
            print("\n👋 Shutting down gracefully...")
        finally:
            self.running = False
            if self.ws:
                self.ws.close()
            print("✅ Data collection stopped.")
            
        return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced IBKR WebSocket Real-time Data Collector")
    parser.add_argument("--market", "-m", choices=["US", "UK", "EU", "CRYPTO"], default="US",
                       help="Market to collect data for (US, UK, EU, CRYPTO)")
    parser.add_argument("--db", "-d", default="database/realtime_market_data.db",
                       help="Database file path")
    parser.add_argument("--clear-data", "-c", action="store_true",
                       help="Clear all real-time data before starting (ensures no fake timestamps)")
    parser.add_argument("--clear-only", action="store_true",
                       help="Only clear real-time data and exit (don't start collection)")
    parser.add_argument("--no-market-hours", action="store_true",
                       help="Disable market hours enforcement (allows collection outside 9:30 AM - 4:00 PM ET)")
    
    args = parser.parse_args()
    
    print("🚀 Enhanced IBKR WebSocket Real-time Data Collector")
    print("=" * 55)
    print("🛡️ Handles rate limits, timeouts, and auto-reconnection")
    print("=" * 55)
    
    # Initialize collector with market parameter
    collector = EnhancedIBWebSocketCollector(args.db, args.market)
    
    # Disable market hours enforcement if requested
    if args.no_market_hours:
        collector.enforce_market_hours = False
        print("⚠️  Market hours enforcement DISABLED - will collect data outside trading hours")
        print("⚠️  WARNING: Data outside market hours may be stale/cached from IBKR")
    
    # Handle clear-only mode
    if args.clear_only:
        print("\n🗑️ Clear-only mode: Removing all real-time data...")
        if collector.clear_realtime_data():
            print("✅ Database cleared successfully!")
        else:
            print("❌ Failed to clear database")
        return
    
    print(f"\n💡 Enhanced features:")
    print(f"   • Rate limiting to avoid IBKR limits")
    print(f"   • Automatic reconnection on timeout")
    print(f"   • Heartbeat to keep connection alive")
    print(f"   • Reduced symbol list for stability")
    print(f"   • Enhanced error handling and logging")
    print(f"   • Timestamp integrity (no fake timestamps)")
    print(f"   • Market hours enforcement: {'DISABLED' if args.no_market_hours else 'ENABLED (9:30 AM - 4:00 PM ET)'}")
    if args.clear_data:
        print(f"   • 🗑️ Clearing existing data for clean start")
    print("")
    
    # Start collection
    collector.start_realtime_collection(clear_existing_data=args.clear_data)

if __name__ == "__main__":
    main()