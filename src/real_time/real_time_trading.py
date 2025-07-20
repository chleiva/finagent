import sqlite3
import time
import sys
import os
from datetime import datetime
import pandas as pd
from tabulate import tabulate
import pytz

# Add parent directories to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import technical assessment function
from feature_engineering.buy_label import meets_basic_buy_conditions
from inference.model_inference_adapter import ModelInferenceAdapter
from feature_engineering.feature_optimizer import optimize_features

DB_PATH = 'database/realtime_market_data.db'
SYMBOLS = ["NVDA", "MSFT", "AAPL", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "NFLX", "COST"]
FETCH_INTERVAL = 30  # seconds

# Initialize model adapter - REAL MODELS ONLY
model_adapter = ModelInferenceAdapter(model_index_path="model_artifacts/model_index.csv")

def get_current_time():
    """Get the current REAL time in NY timezone"""
    try:
        from zoneinfo import ZoneInfo
        ny_tz = ZoneInfo("America/New_York")
    except ImportError:
        ny_tz = pytz.timezone("America/New_York")
    
    return datetime.now(ny_tz)

def fetch_real_time_data(symbol):
    """Fetch REAL-TIME data for a symbol from the database - REAL DATA ONLY"""
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Fetch the latest REAL data from IBKR
        cursor.execute('''
            SELECT 
                last_price, bid_price, ask_price, bid_size, ask_size, volume,
                high_price, low_price, close_price,
                last_update_server_epoch, last_update_received_epoch
            FROM realtime_summary
            WHERE symbol = ?
            ORDER BY last_update_server_epoch DESC
            LIMIT 1
        ''', (symbol,))
        
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"No real-time data found for {symbol}")
        
        (last_price, bid_price, ask_price, bid_size, ask_size, volume,
         high_price, low_price, close_price, server_epoch, received_epoch) = row

        # Validate essential REAL data exists
        if last_price is None or bid_price is None or ask_price is None:
            raise ValueError(f"Missing REAL price data for {symbol}")
        
        if bid_size is None or ask_size is None:
            raise ValueError(f"Missing REAL bid/ask size data for {symbol}")
        
        # Parse timestamp from REAL data
        server_epoch_pandas = pd.to_datetime(server_epoch, unit='s', utc=True)
        
        # Create DataFrame with REAL IBKR data only
        real_time_df = pd.DataFrame({
            'bidPrice': [float(bid_price)],
            'bidSize': [float(bid_size)],
            'askPrice': [float(ask_price)],
            'askSize': [float(ask_size)],
            'lastPrice': [float(last_price)],
            'volume': [float(volume) if volume is not None else 0.0],
            'timestamp': [server_epoch_pandas]
        })
        
        print(f"📊 {symbol} REAL IBKR data: last=${last_price:.2f}, bid=${bid_price:.2f}, ask=${ask_price:.2f}, bid_size={bid_size}, ask_size={ask_size}")
    
    finally:
        conn.close()
    
    return real_time_df

def fetch_intraday_data(symbol, lookback_minutes=60):
    """Fetch REAL intraday minute data for a symbol from the database"""
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Get last lookback_minutes of REAL data
        cursor.execute('''
            SELECT 
                bar_time, open, high, low, close, volume
            FROM intraday_minute_data
            WHERE symbol = ?
            ORDER BY bar_time DESC
            LIMIT ?
        ''', (symbol, lookback_minutes))
        
        rows = cursor.fetchall()
        
        if not rows:
            raise ValueError(f"No intraday data found for {symbol}")
        
        # Create intraday DataFrame with REAL data
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
        intra_day_df = intra_day_df.sort_values('ts_event_clean')
        
        print(f"📊 {symbol} fetched {len(intra_day_df)} REAL intraday records")
        return intra_day_df
    
    finally:
        conn.close()

def fetch_daily_data(symbol, lookback_days=60):
    """Fetch REAL daily data for a symbol from the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            SELECT 
                bar_date, open, high, low, close, volume
            FROM daily_summary_data
            WHERE symbol = ?
            ORDER BY bar_date DESC
            LIMIT ?
        ''', (symbol, lookback_days))
        
        rows = cursor.fetchall()
        
        if not rows:
            raise ValueError(f"No daily data found for {symbol}")
        
        # Create daily DataFrame with REAL data
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
        daily_df = daily_df.sort_values('date')
        
        return daily_df
    
    finally:
        conn.close()

def fetch_latest_prices():
    """Fetch latest REAL prices from the database - REAL-TIME DATA ONLY"""
    
    try:
        from zoneinfo import ZoneInfo
        ny_tz = ZoneInfo("America/New_York")
    except ImportError:
        ny_tz = pytz.timezone("America/New_York")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    data = []
    
    try:
        for symbol in SYMBOLS:
            # ONLY fetch from realtime_summary - REAL IBKR Level 1 data
            cursor.execute('''
                SELECT 
                    last_price, bid_price, ask_price, bid_size, ask_size, volume,
                    last_update_server_epoch
                FROM realtime_summary
                WHERE symbol = ?
                ORDER BY last_update_server_epoch DESC
                LIMIT 1
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
                    
                # REAL Level 1 market data - NO SYNTHETIC VALUES
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
                # No real-time data available
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
    
    finally:
        conn.close()
    
    return data

def check_data_freshness():
    """Check the freshness of REAL data"""
    
    try:
        from zoneinfo import ZoneInfo
        ny_tz = ZoneInfo("America/New_York")
    except ImportError:
        ny_tz = pytz.timezone("America/New_York")
    
    current_time = datetime.now(ny_tz)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Check real-time data freshness
        cursor.execute('''
            SELECT MAX(last_update_server_epoch) FROM realtime_summary
        ''')
        latest_realtime_epoch = cursor.fetchone()[0]
        
        if latest_realtime_epoch:
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
    
    finally:
        conn.close()
    
    return {
        'current_time': current_time,
        'latest_realtime': latest_realtime,
        'latest_historical_date': latest_historical_date,
        'latest_intraday_time': latest_intraday_time,
        'latency_seconds': latency_seconds
    }

def run_technical_assessment(symbol_data):
    """Run technical assessment for each symbol using REAL calculated features"""
    results = []
    current_time = get_current_time()
    
    for row in symbol_data:
        symbol = row['symbol']
        try:
            print(f"\n📊 Processing {symbol}...")
            
            # Fetch REAL data for feature calculation
            real_time_df = fetch_real_time_data(symbol)
            intra_day_df = fetch_intraday_data(symbol)
            daily_df = fetch_daily_data(symbol)
            
            # Calculate features using REAL data
            features = model_adapter.compute_features(real_time_df, intra_day_df, daily_df)
            
            # Run technical assessment with REAL features
            assessment_result, assessment_reason = meets_basic_buy_conditions(features, current_time)
            
            results.append({
                'symbol': symbol,
                'technical_assessment': assessment_result,
                'assessment_reason': assessment_reason
            })

        except Exception as e:
            print(f"❌ Technical assessment failed for {symbol}: {e}")
            results.append({
                'symbol': symbol,
                'technical_assessment': False,
                'assessment_reason': f'error: {str(e)}'
            })
    
    return results


    

def main():
    print("\n🚦 Trading System (REAL DATA MODE ONLY)")
    print("=" * 60)
    print(f"Symbols: {', '.join(SYMBOLS)}")
    print("Fetching REAL data every 30 seconds. Press Ctrl+C to stop.\n")
    
    while True:
        try:
            # Fetch REAL latest prices
            latest_data = fetch_latest_prices()
            
            # Check REAL data freshness
            data_freshness = check_data_freshness()
            
            # Run technical assessment with REAL data
            technical_results = run_technical_assessment(latest_data)
            
            # Filter results
            technical_results_filtered = [r for r in technical_results if 'symbol' in r]
            
            # Only run model inference for symbols that pass technical assessment
            symbols_passed_assessment = [r['symbol'] for r in technical_results_filtered if r['technical_assessment']]
            print(f"✅ Technical assessment passed for: {', '.join(symbols_passed_assessment) if symbols_passed_assessment else 'None'}")
            
            # Run model inference only for symbols that passed technical assessment
            model_results = []
            for row, tech_result in zip(latest_data, technical_results_filtered):
                if tech_result['technical_assessment']:
                    try:
                        # Fetch REAL data for model inference
                        real_time_df = fetch_real_time_data(row['symbol'])
                        intra_day_df = fetch_intraday_data(row['symbol'])
                        daily_df = fetch_daily_data(row['symbol'])
                        
                        # Calculate base features with REAL data
                        features = model_adapter.compute_features(real_time_df, intra_day_df, daily_df)
                        
                        # Apply feature optimization to create missing features
                        optimized_features = optimize_features(features, intra_day_df=intra_day_df)
                        
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
                    volume_percentile = f"{model_result['features'].get('Volume_Percentile_Intraday', 0):.3f}"

                # Format last update time
                def format_last_update_print(last_update):
                    if not last_update:
                        return 'N/A'
                    return last_update.strftime('%H:%M:%S')
                
                table.append([
                    row['symbol'],
                    f"${row['last_price']:.2f}" if row['last_price'] is not None else 'N/A',
                    format_last_update_print(row['last_update']),
                    tech_status,
                    prediction_text if tech_result['technical_assessment'] else 'N/A',
                    f"{model_result.get('probability', 0):.4f}" if tech_result['technical_assessment'] and 'probability' in model_result else 'N/A',
                    volume_percentile,
                    len(model_result.get('missing_features', [])) if tech_result['technical_assessment'] else 'N/A',
                    f"{model_result.get('threshold', 0):.3f}" if tech_result['technical_assessment'] and 'threshold' in model_result else 'N/A'
                ])
            
            # Display current REAL time
            current_time_display = data_freshness['current_time']
            print("\n" + current_time_display.strftime('%Y-%m-%d %H:%M:%S %Z'))
            
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
            print("\n--- REAL DATA FRESHNESS ---")
            print(f"Current time: {data_freshness['current_time'].strftime('%Y-%m-%d %H:%M:%S %Z')}")
            print(f"Last real-time update: {data_freshness['latest_realtime'].strftime('%Y-%m-%d %H:%M:%S %Z') if data_freshness['latest_realtime'] else 'N/A'}")
            print(f"Real-time latency: {data_freshness['latency_seconds']:.1f}s" if data_freshness['latency_seconds'] is not None else "Real-time latency: N/A")
            print(f"Last historical data: {data_freshness['latest_historical_date'] if data_freshness['latest_historical_date'] else 'N/A'}")
            print(f"Last intraday data: {data_freshness['latest_intraday_time'].strftime('%Y-%m-%d %H:%M:%S %Z') if data_freshness['latest_intraday_time'] else 'N/A'}")
            print("========================================\n")

            time.sleep(FETCH_INTERVAL)
            
        except KeyboardInterrupt:
            print("\n🛑 Trading system stopped by user.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(FETCH_INTERVAL)

if __name__ == "__main__":
    main()