import sqlite3
import time
import sys
import os
import argparse
from datetime import datetime, timedelta
import pandas as pd
from tabulate import tabulate
import pytz

# Add parent directories to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import technical assessment function
from feature_engineering.buy_label import meets_basic_buy_conditions

# Import model inference adapter
from inference.model_inference_adapter import ModelInferenceAdapter

# Import feature optimization
from feature_engineering.feature_optimizer import optimize_features

DB_PATH = 'database/realtime_market_data.db'
SYMBOLS = [
    "NVDA", "MSFT", "AAPL", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "NFLX", "COST"
]

FETCH_INTERVAL = 30  # seconds

# Global variable to store time offset in minutes
TIME_OFFSET_MINUTES = 0

def get_simulated_current_time():
    """Get the current time adjusted by the time offset for simulation"""
    try:
        from zoneinfo import ZoneInfo
        ny_tz = ZoneInfo("America/New_York")
    except ImportError:
        # Fallback for Python <3.9
        ny_tz = pytz.timezone("America/New_York")
    
    actual_now = datetime.now(ny_tz)
    simulated_now = actual_now - timedelta(minutes=TIME_OFFSET_MINUTES)
    return simulated_now, actual_now

def datetime_to_utc_string(dt):
    """Convert timezone-aware datetime to UTC string format for database queries"""
    from datetime import timezone
    
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    
    # Format to match database: '2025-07-17T19:59:00+00:00'
    return dt.isoformat()

def get_market_open_time(date):
    """Get market open time for a given date (9:30 AM ET)"""
    try:
        from zoneinfo import ZoneInfo
        ny_tz = ZoneInfo("America/New_York")
    except ImportError:
        # Fallback for Python <3.9
        import pytz
        ny_tz = pytz.timezone("America/New_York")
    
    market_open = datetime.combine(date, datetime.min.time()).replace(hour=9, minute=30, tzinfo=ny_tz)
    
    return market_open

# Initialize model adapter
model_adapter = ModelInferenceAdapter(model_index_path="model_artifacts/model_index.csv")



def fetch_real_time_data(symbol):
    """Fetch real-time data for a symbol from the database"""
    simulated_now, actual_now = get_simulated_current_time()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if TIME_OFFSET_MINUTES > 0:
        # In simulation mode, get the closest intraday data to the simulated time
        cursor.execute('''
            SELECT 
                close, open, high, low, volume, bar_time
            FROM intraday_minute_data
            WHERE symbol = ? AND bar_time <= ?
            ORDER BY bar_time DESC
            LIMIT 1
        ''', (symbol, datetime_to_utc_string(simulated_now)))

        
        row = cursor.fetchone()
        if not row:
            conn.close()
            raise ValueError(f"No simulated real-time data found for {symbol} at {simulated_now}")


        # Debug: Print the full row data
        #print(f"🔍 RAW DB ROW for {symbol}: {row}")
        #print(f"🔍 ROW TYPES: {[type(x) for x in row] if row else 'None'}")

        
        close, open_price, high, low, volume, bar_time = row
        
        # Create simulated real-time data based on the intraday bar
        # Use close as last price, and create synthetic bid/ask around it
        spread_pct = 0.001  # 0.1% spread
        last_price = close
        bid_price = last_price * (1 - spread_pct / 2)
        ask_price = last_price * (1 + spread_pct / 2)
        
        #debug print volume
        print(f"DEBUB  VOLUME = {volume}")

        # Create real-time DataFrame
        real_time_df = pd.DataFrame({
            'bidPrice': [bid_price],
            'bidSize': [100],  # Default bid size
            'askPrice': [ask_price],
            'askSize': [100],  # Default ask size
            'lastPrice': [last_price],
            'volume': [float(volume) if volume is not None else 0.0],
            'timestamp': [simulated_now]
        })
        
        print(f"📊 {symbol} simulated real-time data from {bar_time}: last=${last_price:.2f}, bid=${bid_price:.2f}, ask=${ask_price:.2f}")
        
    else:
        # Original real-time data fetching
        cursor.execute('''
            SELECT 
                last_price, bid_price, ask_price, bid_size, ask_size, volume,
                high_price, low_price, close_price,
                last_update_server_epoch, last_update_received_epoch
            FROM realtime_summary
            WHERE symbol = ?
        ''', (symbol,))
        
        row = cursor.fetchone()
        if not row:
            conn.close()
            raise ValueError(f"No real-time data found for {symbol}")
        
        (last_price, bid_price, ask_price, bid_size, ask_size, volume,
         high_price, low_price, close_price, server_epoch, received_epoch) = row
        
        # Check if we have essential data
        if last_price is None or bid_price is None or ask_price is None:
            raise ValueError(f"Missing essential price data for {symbol}")
        
        if bid_size is None or ask_size is None:
            raise ValueError(f"Missing bid/ask size data for {symbol}")
        
        # Create real-time DataFrame
        real_time_df = pd.DataFrame({
            'bidPrice': [bid_price],
            'bidSize': [bid_size],
            'askPrice': [ask_price],
            'askSize': [ask_size],
            'lastPrice': [last_price],
            'volume': [volume] if volume else [0],
            'timestamp': [datetime.fromtimestamp(server_epoch) if server_epoch else datetime.now()]
        })
    
    conn.close()
    return real_time_df




def fetch_intraday_data(symbol, lookback_minutes=60):
    """Fetch intraday minute data for a symbol from the database"""
    simulated_now, actual_now = get_simulated_current_time()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if TIME_OFFSET_MINUTES > 0:
        # In simulation mode, get data from market open until simulated time minus 1 minute
        market_open = get_market_open_time(simulated_now)
        
        # Get data from market open until simulated time minus 1 minute
        end_time = simulated_now - timedelta(minutes=1)

        print(f"📈 {symbol} fetching intraday data from {market_open.strftime('%H:%M')} to {end_time.strftime('%H:%M')} (simulated time: {simulated_now.strftime('%H:%M')})")

        # Use datetime() function in SQL for proper timestamp comparison
        cursor.execute('''
            SELECT 
                bar_time, open, high, low, close, volume
            FROM intraday_minute_data
            WHERE symbol = ? 
            AND datetime(bar_time) >= datetime(?) 
            AND datetime(bar_time) <= datetime(?)
            ORDER BY bar_time ASC
        ''', (symbol, datetime_to_utc_string(market_open), datetime_to_utc_string(end_time)))
        
    else:
        # Original intraday data fetching
        cutoff_time = datetime.now() - pd.Timedelta(minutes=lookback_minutes)
        
        cursor.execute('''
            SELECT 
                bar_time, open, high, low, close, volume
            FROM intraday_minute_data
            WHERE symbol = ? AND datetime(bar_time) >= datetime(?)
            ORDER BY bar_time DESC
            LIMIT ?
        ''', (symbol, cutoff_time.strftime('%Y-%m-%d %H:%M:%S'), lookback_minutes))
    
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        raise ValueError(f"No intraday data found for {symbol} - query returned 0 rows")
    
    # Create intraday DataFrame
    data = []
    for row in rows:
        bar_time, open_price, high, low, close, volume = row
        data.append({
            'close_1min': close,
            'open_1min': open_price,
            'high_1min': high,
            'low_1min': low,
            'volume_1min': volume if volume else 0,
            'ts_event_clean': bar_time,
            'symbol_price': symbol
        })
    
    intra_day_df = pd.DataFrame(data)
    
    # Sort by timestamp (oldest first for simulation, newest first for real-time)
    if TIME_OFFSET_MINUTES > 0:
        intra_day_df = intra_day_df.sort_values('ts_event_clean')
    else:
        intra_day_df = intra_day_df.sort_values('ts_event_clean')
    
    print(f"📊 {symbol} fetched {len(intra_day_df)} intraday records")
    return intra_day_df





def fetch_daily_data(symbol, lookback_days=60):
    """Fetch daily data for a symbol from the database (unchanged for simulation)"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT 
            bar_date, open, high, low, close, volume
        FROM daily_summary_data
        WHERE symbol = ?
        ORDER BY bar_date DESC
        LIMIT ?
    ''', (symbol, lookback_days))
    
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        raise ValueError(f"No daily data found for {symbol}")
    
    # Create daily DataFrame
    data = []
    for row in rows:
        bar_date, open_price, high, low, close, volume = row
        data.append({
            'date': bar_date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume if volume else 0,
            'symbol': symbol
        })
    
    daily_df = pd.DataFrame(data)
    
    # Sort by date (oldest first)
    daily_df = daily_df.sort_values('date')
    
    return daily_df

def get_model_scores(symbol_data):
    """Get model scores using real data from the database (supports simulation mode)"""
    results = []
    
    for row in symbol_data:
        symbol = row['symbol']
        try:
            # Fetch data from database (automatically handles simulation mode)
            real_time_df = fetch_real_time_data(symbol)
            intra_day_df = fetch_intraday_data(symbol)
            daily_df = fetch_daily_data(symbol)
            
            # Calculate base features first
            features = model_adapter.compute_features(real_time_df, intra_day_df, daily_df)
            print(f"🔍 FEATURES DEBUG {row['symbol']}: Volume_Percentile_Intraday = {features.get('Volume_Percentile_Intraday', 'NOT_FOUND')}")

            
            
            # Debug: Check intraday data structure
            print(f"🔍 Debug {symbol}: intra_day_df shape={intra_day_df.shape}, columns={list(intra_day_df.columns)}")
            if not intra_day_df.empty:
                print(f"   Sample data: high_1min={intra_day_df['high_1min'].iloc[-1]}, low_1min={intra_day_df['low_1min'].iloc[-1]}")
            
            # Apply feature optimization to create missing features
            optimized_features = optimize_features(features, intra_day_df=intra_day_df)

            # Debug: Check what features were created
            impulse_features = {k: v for k, v in optimized_features.items() if 'Impulse' in k}
            
            # Get the model and run inference with optimized features
            model = model_adapter.get_model(symbol)
            model_features = model.features
            
            # Prepare feature vector in correct order using optimized features
            feature_vector = []
            missing_features = []
            for feat in model_features:
                if feat in optimized_features:
                    feature_vector.append(optimized_features[feat])
                else:
                    feature_vector.append(0.0)
                    missing_features.append(feat)
            
            X = pd.DataFrame([feature_vector], columns=model_features)

            
            # Run model prediction
            threshold = 0.5
            prediction, probability = model.predict(X, threshold=threshold)
            
            # Create result
            result = {
                'symbol': symbol,
                'prediction': int(prediction[0]),
                'probability': float(probability[0]),
                'features': optimized_features,
                'missing_features': missing_features,
                'model_info': model.get_info(),
                'threshold': threshold
            }
            results.append(result)
            
        except Exception as e:
            print(f"❌ Model inference failed for {symbol}: {e}")
            # Return failure result
            results.append({
                'symbol': symbol,
                'prediction': 0,
                'probability': 0.0,
                'missing_features': [str(e)],
                'threshold': 0.5,
                'error': str(e)
            })
    
    return results

def fetch_latest_prices():
    """Fetch latest prices from the database"""
    simulated_now, actual_now = get_simulated_current_time()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    data = []
    
    for symbol in SYMBOLS:
        if TIME_OFFSET_MINUTES > 0:
            # In simulation mode, get the closest intraday data to the simulated time
            cursor.execute('''
                SELECT 
                    close, open, high, low, volume, bar_time
                FROM intraday_minute_data
                WHERE symbol = ? AND bar_time <= ?
                ORDER BY bar_time DESC
                LIMIT 1
            ''', (symbol, datetime_to_utc_string(simulated_now)))
            
            row = cursor.fetchone()
            if row:
                close, open_price, high, low, volume, bar_time = row
                
                # Create simulated real-time data based on the intraday bar
                spread_pct = 0.001  # 0.1% spread
                last_price = close
                bid_price = last_price * (1 - spread_pct / 2)
                ask_price = last_price * (1 + spread_pct / 2)
                
                # Parse bar_time to datetime (database stores UTC, convert to NY time)
                #last_update = pd.to_datetime(bar_time).tz_localize('UTC').tz_convert('America/New_York')
                last_update = pd.to_datetime(bar_time)  # Already in UTC from database


                data.append({
                    'symbol': symbol, 
                    'last_price': last_price, 
                    'bid_price': bid_price,
                    'ask_price': ask_price,
                    'bid_size': 100,  # Default bid size
                    'ask_size': 100,  # Default ask size
                    'volume': volume,
                    'last_update': last_update
                })
            else:
                data.append({
                    'symbol': symbol, 
                    'last_price': None, 
                    'bid_price': None,
                    'ask_price': None,
                    'bid_size': None,
                    'ask_size': None,
                    'volume': None,
                    'last_update': None
                })
        else:
            # Original real-time data fetching
            # Get New York timezone
            try:
                from zoneinfo import ZoneInfo
                ny_tz = ZoneInfo("America/New_York")
            except ImportError:
                # Fallback for Python <3.9
                ny_tz = pytz.timezone("America/New_York")
            
            cursor.execute('''
                SELECT 
                    last_price, 
                    bid_price, 
                    ask_price, 
                    bid_size, 
                    ask_size, 
                    volume,
                    last_update_server_epoch
                FROM realtime_summary
                WHERE symbol = ?
            ''', (symbol,))
            
            row = cursor.fetchone()
            if row:
                (last_price, bid_price, ask_price, bid_size, ask_size, volume, server_epoch) = row
                if server_epoch:
                    # Convert UTC timestamp to New York timezone
                    utc_time = datetime.utcfromtimestamp(server_epoch)
                    last_update = utc_time.replace(tzinfo=pytz.UTC).astimezone(ny_tz)
                else:
                    last_update = None
                data.append({
                    'symbol': symbol, 
                    'last_price': last_price, 
                    'bid_price': bid_price,
                    'ask_price': ask_price,
                    'bid_size': bid_size,
                    'ask_size': ask_size,
                    'volume': volume,
                    'last_update': last_update
                })
            else:
                data.append({
                    'symbol': symbol, 
                    'last_price': None, 
                    'bid_price': None,
                    'ask_price': None,
                    'bid_size': None,
                    'ask_size': None,
                    'volume': None,
                    'last_update': None
                })
    
    conn.close()
    return data

def check_data_freshness():
    """Check the freshness of real-time, historical, and intraday data"""
    simulated_now, actual_now = get_simulated_current_time()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if TIME_OFFSET_MINUTES > 0:
        # In simulation mode, check data freshness relative to simulated time
        current_time = simulated_now
        
        # Check simulated real-time data freshness (latest intraday data up to simulated time)
        cursor.execute('''

        
            SELECT MAX(bar_time) FROM intraday_minute_data
            WHERE bar_time <= ?
        ''', (datetime_to_utc_string(simulated_now),))
        latest_realtime_str = cursor.fetchone()[0]
        if latest_realtime_str:
            latest_realtime = pd.to_datetime(latest_realtime_str).tz_convert('America/New_York')
        else:
            latest_realtime = None
        
        # Check historical data freshness (unchanged)
        cursor.execute('''
            SELECT MAX(bar_date) FROM daily_summary_data
        ''')
        latest_historical_date = cursor.fetchone()[0]
        
        # Check intraday data freshness up to simulated time
        cursor.execute('''
            SELECT MAX(bar_time) FROM intraday_minute_data
            WHERE bar_time <= ?
        ''', (datetime_to_utc_string(simulated_now),))
        latest_intraday_time_str = cursor.fetchone()[0]
        latest_intraday_time = None
        if latest_intraday_time_str:
            try:
                latest_intraday_time = pd.to_datetime(latest_intraday_time_str)
            except:
                latest_intraday_time = None
        
        # Calculate latency for simulated real-time data
        latency_seconds = None
        if latest_realtime:
            latency_seconds = (current_time - latest_realtime).total_seconds()
        
    else:
        # Original data freshness checking
        # Get New York timezone
        try:
            from zoneinfo import ZoneInfo
            ny_tz = ZoneInfo("America/New_York")
        except ImportError:
            # Fallback for Python <3.9
            ny_tz = pytz.timezone("America/New_York")
        
        current_time = datetime.now(ny_tz)
        
        # Check real-time data freshness
        cursor.execute('''
            SELECT MAX(last_update_server_epoch) FROM realtime_summary
        ''')
        latest_realtime_epoch = cursor.fetchone()[0]
        if latest_realtime_epoch:
            # Convert UTC timestamp to New York timezone
            utc_time = datetime.utcfromtimestamp(latest_realtime_epoch)
            latest_realtime = utc_time.replace(tzinfo=pytz.UTC).astimezone(ny_tz)
        else:
            latest_realtime = None
        
        # Check historical data freshness
        cursor.execute('''
            SELECT MAX(bar_date) FROM daily_summary_data
        ''')
        latest_historical_date = cursor.fetchone()[0]
        
        # Check intraday data freshness
        cursor.execute('''
            SELECT MAX(bar_time) FROM intraday_minute_data
        ''')
        latest_intraday_time_str = cursor.fetchone()[0]
        latest_intraday_time = None
        if latest_intraday_time_str:
            try:
                latest_intraday_time = pd.to_datetime(latest_intraday_time_str)
            except:
                latest_intraday_time = None
        
        # Calculate latency for real-time data
        latency_seconds = None
        if latest_realtime:
            latency_seconds = (current_time - latest_realtime).total_seconds()
    
    conn.close()
    
    return {
        'current_time': current_time,
        'latest_realtime': latest_realtime,
        'latest_historical_date': latest_historical_date,
        'latest_intraday_time': latest_intraday_time,
        'latency_seconds': latency_seconds,
        'actual_time': actual_now if TIME_OFFSET_MINUTES > 0 else None
    }

def run_technical_assessment(symbol_data):
    """Run technical assessment for each symbol using real calculated features"""
    results = []
    debug_features_nvda = None
    debug_reason_nvda = None
    
    # Get current time (simulated or actual)
    simulated_now, actual_now = get_simulated_current_time()
    current_time = simulated_now
    
    for row in symbol_data:
        symbol = row['symbol']
        try:
            # Fetch real data for feature calculation
            real_time_df = fetch_real_time_data(symbol)
            intra_day_df = fetch_intraday_data(symbol)
            daily_df = fetch_daily_data(symbol)
            
            # Calculate features and optimize
            features = model_adapter.compute_features(real_time_df, intra_day_df, daily_df)
        
        
            # Use optimized features for technical assessment
            ts = pd.Timestamp(current_time)
            if bool(pd.isnull(ts)) or not isinstance(ts, pd.Timestamp):
                ts = pd.Timestamp.now(tz=ny_tz)
            
            #This must always be called with features without optimize!!!!
            assessment_passed, assessment_reason = meets_basic_buy_conditions(features, ts)
            results.append({
                'symbol': symbol,
                'technical_assessment': assessment_passed,
                'assessment_reason': assessment_reason
            })
            


        except Exception as e:
            print(f"❌ Technical assessment failed for {symbol}: {e}")
            results.append({
                'symbol': symbol,
                'technical_assessment': False,
                'assessment_reason': f'error: {str(e)}'
            })
    
    # Attach debug info for NVDA to the results for printing after the table
    results.append({'_debug_nvda_features': debug_features_nvda, '_debug_nvda_reason': debug_reason_nvda})
    return results

def main():
    global TIME_OFFSET_MINUTES
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Trading Simulation with Time Offset Support")
    parser.add_argument('--time_offset', type=int, default=0, 
                        help='Time offset in minutes to simulate trading in the past (0 = real-time mode)')
    
    args = parser.parse_args()
    TIME_OFFSET_MINUTES = args.time_offset
    
    # Display mode information
    if TIME_OFFSET_MINUTES > 0:
        print(f"\n🕐 Trading Simulation (EXPERIMENTATION MODE - {TIME_OFFSET_MINUTES} minutes offset)")
        print("=" * 80)
        simulated_now, actual_now = get_simulated_current_time()
        print(f"Actual time: {actual_now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"Simulated time: {simulated_now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"Time offset: {TIME_OFFSET_MINUTES} minutes")
        print(f"Symbols: {', '.join(SYMBOLS)}")
        print("Using historical intraday data for simulation. Press Ctrl+C to stop.\n")
    else:
        print("\n🚦 Trading Simulation (REAL DATA MODE)")
        print("=" * 60)
        print(f"Symbols: {', '.join(SYMBOLS)}")
        print("Fetching real data every 30 seconds. Press Ctrl+C to stop.\n")
    
    while True:
        try:
            latest_data = fetch_latest_prices()
            
            # Check data freshness
            data_freshness = check_data_freshness()
            
            # Run technical assessment first
            technical_results = run_technical_assessment(latest_data)
            
            # Filter out debug objects from technical results
            technical_results_filtered = [r for r in technical_results if 'symbol' in r]
            
            # Only run model inference for symbols that pass technical assessment
            symbols_passed_assessment = [r['symbol'] for r in technical_results_filtered if r['technical_assessment']]
            print(f"✅ Technical assessment passed for: {', '.join(symbols_passed_assessment) if symbols_passed_assessment else 'None'}")
            
            # Run model inference only for symbols that passed technical assessment
            model_results = []
            for row, tech_result in zip(latest_data, technical_results_filtered):
                if tech_result['technical_assessment']:
                    try:
                        # Fetch real data for model inference
                        real_time_df = fetch_real_time_data(row['symbol'])
                        intra_day_df = fetch_intraday_data(row['symbol'])
                        daily_df = fetch_daily_data(row['symbol'])
                        
                        # Calculate base features first
                        features = model_adapter.compute_features(real_time_df, intra_day_df, daily_df)
                        
                        # Debug: Check intraday data structure
                        print(f"🔍 Debug {row['symbol']}: intra_day_df shape={intra_day_df.shape}, columns={list(intra_day_df.columns)}")
                        if not intra_day_df.empty:
                            print(f"   Sample data: high_1min={intra_day_df['high_1min'].iloc[-1]}, low_1min={intra_day_df['low_1min'].iloc[-1]}")
                        
                        # Apply feature optimization to create missing features
                        optimized_features = optimize_features(features, intra_day_df=intra_day_df)
                        #features must never be optimized for BUY signal
                        
                        # Debug: Check what features were created
                        impulse_features = {k: v for k, v in optimized_features.items() if 'Impulse' in k}
                        print(f"   Impulse features created: {len(impulse_features)} - {list(impulse_features.keys())}")
                        
                        # Get the model and run inference with optimized features
                        model = model_adapter.get_model(row['symbol'])
                        model_features = model.features
                        
                        # Prepare feature vector in correct order using optimized features
                        feature_vector = []
                        missing_features = []
                        for feat in model_features:
                            if feat in optimized_features:
                                feature_vector.append(optimized_features[feat])
                            else:
                                feature_vector.append(0.0)
                                missing_features.append(feat)
                        
                        X = pd.DataFrame([feature_vector], columns=model_features)
                        
                        # Run model prediction
                        threshold = 0.5
                        prediction, probability = model.predict(X, threshold=threshold)
                        
                        # Create result
                        result = {
                            'symbol': row['symbol'],
                            'prediction': int(prediction[0]),
                            'probability': float(probability[0]),
                            'features': optimized_features,
                            'missing_features': missing_features,
                            'model_info': model.get_info(),
                            'threshold': threshold
                        }
                        model_results.append(result)
                    except Exception as e:
                        print(f"❌ Model inference failed for {row['symbol']}: {e}")
                        model_results.append({
                            'symbol': row['symbol'],
                            'prediction': 0,
                            'probability': 0.0,
                            'missing_features': [str(e)],
                            'threshold': 0.5,
                            'error': str(e)
                        })
                else:
                    # Add placeholder for symbols that didn't pass technical assessment
                    model_results.append({
                        'symbol': row['symbol'],
                        'prediction': 0,
                        'probability': 0.0,
                        'missing_features': [],
                        'threshold': 0.5
                    })
            
            # Merge results for display
            table = []
            for row, tech_result, model_result in zip(latest_data, technical_results_filtered, model_results):
                prediction_text = "BUY" if model_result.get('prediction', 0) == 1 else "NO BUY"
                tech_status = "✅ PASS" if tech_result['technical_assessment'] else f"❌ {tech_result['assessment_reason']}"
                
                # Get volume percentile for display
                volume_percentile = "N/A"
                if tech_result['technical_assessment'] and 'features' in model_result:
                    #volume_percentile = f"{model_result['features'].get('Volume_Percentile_Intraday', 0):.3f}"
                    volume_percentile = f"{features.get('Volume_Percentile_Intraday', 0):.3f}"
                elif not tech_result['technical_assessment']:
                    # Try to get volume percentile even for failed assessments
                    try:
                        real_time_df = fetch_real_time_data(row['symbol'])
                        intra_day_df = fetch_intraday_data(row['symbol'])
                        daily_df = fetch_daily_data(row['symbol'])
                        features = model_adapter.compute_features(real_time_df, intra_day_df, daily_df)
                        optimized_features = optimize_features(features, intra_day_df=intra_day_df)
                        volume_percentile = f"{features.get('Volume_Percentile_Intraday', 0):.3f}"
                        
                        # DEBUG: Print volume information
                        current_volume = real_time_df['volume'].iloc[0] if 'volume' in real_time_df.columns else 0
                        volume_col = 'volume_1min' if 'volume_1min' in intra_day_df.columns else 'volume'
                        if volume_col in intra_day_df.columns:
                            day_volumes = intra_day_df[volume_col].dropna()
                            print(f"🔍 DEBUG {row['symbol']}: current_volume={current_volume}, day_volumes_count={len(day_volumes)}, day_volumes_range=[{day_volumes.min():.0f}, {day_volumes.max():.0f}]")
                    except:
                        volume_percentile = "N/A"

                                # Convert timezone just for display
                def format_last_update_print(last_update):
                    if not last_update:
                        return 'N/A'
                    if last_update.tzinfo:
                        # Convert UTC to NY time for display
                        ny_time = last_update.tz_convert('America/New_York')
                        return ny_time.strftime('%H:%M:%S')
                    else:
                        return last_update.strftime('%H:%M:%S')
                
                table.append([
                    row['symbol'],
                    f"${row['last_price']:.2f}" if row['last_price'] is not None else 'N/A',
                    format_last_update_print(row['last_update']) if row['last_update'] else 'N/A',
                    tech_status,
                    prediction_text if tech_result['technical_assessment'] else 'N/A',
                    f"{model_result.get('probability', 0):.4f}" if tech_result['technical_assessment'] and 'probability' in model_result else 'N/A',
                    volume_percentile,
                    len(model_result.get('missing_features', [])) if tech_result['technical_assessment'] else 'N/A',
                    f"{model_result.get('threshold', 0):.3f}" if tech_result['technical_assessment'] and 'threshold' in model_result else 'N/A'
                ])
            
            # Display current time (simulated or actual)
            current_time_display = data_freshness['current_time']
            print("\n" + current_time_display.strftime('%Y-%m-%d %H:%M:%S %Z'))
            if TIME_OFFSET_MINUTES > 0:
                print(f"(Simulated time - actual time: {data_freshness['actual_time'].strftime('%Y-%m-%d %H:%M:%S %Z')})")
            
            print(tabulate(table, headers=["Symbol", "Last Price", "Last Update", "Tech Assessment", "Prediction", "Probability", "Vol %", "Missing Features", "Threshold"], tablefmt="fancy_grid"))

            # After the table, print missing feature warnings
            for row, tech_result, model_result in zip(latest_data, technical_results_filtered, model_results):
                if tech_result['technical_assessment'] and model_result.get('missing_features'):
                    missing = model_result['missing_features']
                    if missing:
                        print(f"⚠️  {row['symbol']} missing features: {', '.join(str(f) for f in missing)}")

            # --- SUMMARY SECTION ---
            buy_signals = [r['symbol'] for r, t in zip(model_results, technical_results_filtered) if t['technical_assessment'] and r.get('prediction', 0) == 1]
            num_tech_pass = sum(1 for t in technical_results_filtered if t['technical_assessment'])
            num_buy_signals = len(buy_signals)
            symbols_with_missing = [r['symbol'] for r in model_results if r.get('missing_features') and len(r['missing_features']) > 0]
            
            print("\n================ SUMMARY ================")
            print(f"BUY SIGNALS: {', '.join(buy_signals) if buy_signals else 'None'}")
            print(f"Number of symbols passing technical assessment: {num_tech_pass}")
            print(f"Number of BUY signals: {num_buy_signals}")
            print(f"Symbols with missing features: {', '.join(symbols_with_missing) if symbols_with_missing else 'None'}")
            print("\n--- DATA FRESHNESS ---")
            if TIME_OFFSET_MINUTES > 0:
                print(f"Simulation mode: {TIME_OFFSET_MINUTES} minutes offset")
                print(f"Actual time: {data_freshness['actual_time'].strftime('%Y-%m-%d %H:%M:%S %Z')}")
                print(f"Simulated time: {data_freshness['current_time'].strftime('%Y-%m-%d %H:%M:%S %Z')}")
            else:
                print(f"Current time: {data_freshness['current_time'].strftime('%Y-%m-%d %H:%M:%S %Z')}")
            print(f"Last real-time update: {data_freshness['latest_realtime'].strftime('%Y-%m-%d %H:%M:%S %Z') if data_freshness['latest_realtime'] else 'N/A'}")
            print(f"Real-time latency: {data_freshness['latency_seconds']:.1f}s" if data_freshness['latency_seconds'] is not None else "Real-time latency: N/A")
            print(f"Last historical data: {data_freshness['latest_historical_date'] if data_freshness['latest_historical_date'] else 'N/A'}")
            print(f"Last intraday data: {data_freshness['latest_intraday_time'].strftime('%Y-%m-%d %H:%M:%S %Z') if data_freshness['latest_intraday_time'] else 'N/A'}")
            print("========================================\n")

            time.sleep(FETCH_INTERVAL)
            
        except KeyboardInterrupt:
            print("\n🛑 Simulation stopped by user.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(FETCH_INTERVAL)

if __name__ == "__main__":
    main() 