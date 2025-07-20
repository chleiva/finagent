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

# Import historical simulation fetcher
from real_time.historical_simulation_fetcher import LocalHistoricalSimulationFetcher

DB_PATH = 'database/realtime_market_data.db'
SYMBOLS = [
    "NVDA", "MSFT", "AAPL", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "NFLX", "COST"
]

FETCH_INTERVAL = 5  # seconds

# Global variables for simulation mode
SIMULATION_DATE_NY = None

# Get timezone objects
try:
    from zoneinfo import ZoneInfo
    NY_TZ = ZoneInfo("America/New_York")
    UTC_TZ = ZoneInfo("UTC")
except ImportError:
    # Fallback for Python <3.9
    NY_TZ = pytz.timezone("America/New_York")
    UTC_TZ = pytz.UTC

def parse_simulation_date(date_str):
    """Parse simulation date string in format YYYYMMDDHH24MI to datetime object in NY timezone"""
    if len(date_str) != 12:
        raise ValueError("Simulation date must be in format YYYYMMDDHH24MI (12 digits)")
    
    try:
        year = int(date_str[0:4])
        month = int(date_str[4:6])
        day = int(date_str[6:8])
        hour = int(date_str[8:10])
        minute = int(date_str[10:12])
        
        # Create datetime object with NY timezone (this is what user inputs)
        simulation_dt_ny = datetime(year, month, day, hour, minute, tzinfo=NY_TZ)
        return simulation_dt_ny
    except ValueError as e:
        raise ValueError(f"Invalid simulation date format: {e}")

def ny_to_utc(dt_ny):
    """Convert NY timezone datetime to UTC"""
    if dt_ny.tzinfo is None:
        dt_ny = dt_ny.replace(tzinfo=NY_TZ)
    return dt_ny.astimezone(UTC_TZ)

def utc_to_ny(dt_utc):
    """Convert UTC datetime to NY timezone"""
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=UTC_TZ)
    return dt_utc.astimezone(NY_TZ)

def datetime_to_utc_string(dt):
    """Convert timezone-aware datetime to UTC string format for database queries"""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC_TZ)
    else:
        dt = dt.astimezone(UTC_TZ)
    
    # Format to match database: '2025-07-18T14:00:00+00:00'
    return dt.isoformat()

def get_market_open_time(date_ny):
    """Get market open time for a given date (9:30 AM ET) in NY timezone"""
    if isinstance(date_ny, datetime):
        date_only = date_ny.date()
    else:
        date_only = date_ny
    
    market_open_ny = datetime.combine(date_only, datetime.min.time()).replace(hour=9, minute=30, tzinfo=NY_TZ)
    return market_open_ny

def get_market_close_time(date_ny):
    """Get market close time for a given date (4:00 PM ET) in NY timezone"""
    if isinstance(date_ny, datetime):
        date_only = date_ny.date()
    else:
        date_only = date_ny
    
    market_close_ny = datetime.combine(date_only, datetime.min.time()).replace(hour=16, minute=0, tzinfo=NY_TZ)
    return market_close_ny

def is_market_hours(dt_ny):
    """Check if given NY time is during market hours"""
    market_open = get_market_open_time(dt_ny)
    market_close = get_market_close_time(dt_ny)
    
    # Check if it's a weekday (Monday=0, Sunday=6)
    if dt_ny.weekday() >= 5:  # Saturday or Sunday
        return False
    
    return market_open <= dt_ny <= market_close

# Initialize model adapter and historical fetcher
model_adapter = ModelInferenceAdapter(model_index_path="model_artifacts/model_index.csv")
historical_fetcher = LocalHistoricalSimulationFetcher(db_path=DB_PATH)



def fetch_intraday_data(symbol, simulated_now_utc):
    """Fetch intraday minute data for a symbol from the database using historical simulation"""
    simulation_date = simulated_now_utc.date()
    
    print(f"\n🔄 DEBUG: Fetching intraday data for {symbol}")
    print(f"  - Simulated time UTC: {simulated_now_utc}")
    print(f"  - Simulation date: {simulation_date}")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get data from start of day until simulated time minus 1 minute (to avoid lookahead bias)
    # This ensures we only use data that would have been available at simulation time
    end_time_utc = simulated_now_utc - timedelta(minutes=1)
    
    print(f"  - End time UTC (simulation - 1 min): {end_time_utc}")
    
    # Query only data up to simulation time - no future data allowed
    cursor.execute('''
        SELECT 
            ts_event_price, open_1min, high_1min, low_1min, close_1min, volume_1min,
            bid_px_00, ask_px_00, bid_sz_00, ask_sz_00
        FROM historical_simulation_intraday_data
        WHERE symbol_price = ? AND simulation_date = ? AND datetime(ts_event_price) <= datetime(?)
        ORDER BY datetime(ts_event_price) ASC
    ''', (symbol, simulation_date, end_time_utc.isoformat()))
    
    rows = cursor.fetchall()

    print(f"  - Found {len(rows)} rows up to simulation time")
    if rows:
        last_bar_utc = pd.to_datetime(rows[-1][0], utc=True)
        last_bar_ny = utc_to_ny(last_bar_utc)
        print(f"  - Last bar time: {last_bar_ny.strftime('%H:%M:%S %Z')} (NY)")

    conn.close()
    
    if not rows:
        raise ValueError(f"No intraday data found for {symbol} up to simulation time - query returned 0 rows")
    
    # Create intraday DataFrame
    data = []
    for row in rows:
        ts_event_price, open_price, high, low, close_1min, volume, bid_px_00, ask_px_00, bid_sz_00, ask_sz_00 = row
        data.append({
        'close_1min': close_1min,
        'close': close_1min,  # Add backward compatibility
        'open_1min': open_price,
        'open': open_price,   # Add backward compatibility
        'high_1min': high,
        'high': high,         # Add backward compatibility
        'low_1min': low,
        'low': low,           # Add backward compatibility
        'volume_1min': float(volume) if volume is not None else 0.0,
        'volume': float(volume) if volume is not None else 0.0,
        'ts_event_clean': ts_event_price,
        'symbol_price': symbol,
        'bidPrice': float(bid_px_00) if bid_px_00 is not None else None,
        'askPrice': float(ask_px_00) if ask_px_00 is not None else None,
        'bidSize': float(bid_sz_00) if bid_sz_00 is not None else None,
        'askSize': float(ask_sz_00) if ask_sz_00 is not None else None,
        'lastPrice': close_1min,
        'timestamp': ts_event_price
        })
    
    intra_day_df = pd.DataFrame(data)
    
    # Filter out future data after simulation time
    intra_day_df['ts_event_clean'] = pd.to_datetime(intra_day_df['ts_event_clean'], utc=True)
    intra_day_df = intra_day_df[intra_day_df['ts_event_clean'] <= end_time_utc]

    # Sort by timestamp (oldest first for simulation)
    intra_day_df = intra_day_df.sort_values('ts_event_clean')
    
    print(f"📊 {symbol} fetched {len(intra_day_df)} intraday records (up to simulation time)")
    return intra_day_df

    


def fetch_daily_data(symbol, lookback_days=60):
    """Fetch daily data for a symbol from the database"""
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
        bar_date, open_price, high, low, close_1min, volume = row
        data.append({
            'date': bar_date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_1min,
            'volume': float(volume) if volume is not None else 0.0,
            'symbol': symbol
        })
    
    daily_df = pd.DataFrame(data)
    
    # Sort by date (oldest first)
    daily_df = daily_df.sort_values('date')
    
    return daily_df


def fetch_latest_prices(simulated_now_utc):
  """Fetch latest prices from the database using historical simulation"""
  data = []
  
  for symbol in SYMBOLS:
      try:
          intra_day_df = fetch_intraday_data(symbol, simulated_now_utc)
          if not intra_day_df.empty:
              latest = intra_day_df.iloc[-1]
              data.append({
                  'symbol': symbol,
                  'last_price': latest['close_1min'],
                  'bid_price': latest.get('bidPrice'),
                  'ask_price': latest.get('askPrice'),
                  'bid_size': latest.get('bidSize'),
                  'ask_size': latest.get('askSize'),
                  'volume': latest['volume_1min'],
                  'last_update': latest['ts_event_clean']
              })
          else:
              data.append({'symbol': symbol, 'last_price': None, 'bid_price': None, 'ask_price': None, 'bid_size': None, 'ask_size': None, 'volume': None, 'last_update': None})
      except Exception as e:
          data.append({'symbol': symbol, 'last_price': None, 'bid_price': None, 'ask_price': None, 'bid_size': None, 'ask_size': None, 'volume': None, 'last_update': None})
          print(f"General Error: {e}")
  return data


def run_technical_assessment(symbol_data, simulated_now_utc):
    """Run technical assessment for each symbol using real calculated features"""
    results = []
    
    # Convert to NY time for market hours check
    current_time_ny = utc_to_ny(simulated_now_utc)
    

    for row in symbol_data:
        print(f"DEBUG LINE 464: {row}")
        symbol = row['symbol']
        try:
            print(f"\n📊 Processing {symbol}...")
            
            # Check if we're in market hours first
            if not is_market_hours(current_time_ny):
                print(f"  - Outside market hours for {symbol}")
                results.append({
                    'symbol': symbol,
                    'technical_assessment': False,
                    'assessment_reason': 'outside_market_hours'
                })
                continue
            
            
            # Fetch intraday data
            intra_day_df = fetch_intraday_data(symbol, simulated_now_utc)
            print(f"  - Intraday data shape: {intra_day_df.shape}")
            

            # Fetch real data for feature calculation
            real_time_df = intra_day_df.iloc[[-1]] if not intra_day_df.empty else pd.DataFrame()

            print(f"DEBUG line 395: {real_time_df}")

            # Fetch daily data
            daily_df = fetch_daily_data(symbol)
            print(f"  - Daily data shape: {daily_df.shape}")
            
            # Calculate features
            features = model_adapter.compute_features(real_time_df, intra_day_df, daily_df)
            print(f"  - Calculated features: {list(features.keys())}")
            
            # Convert UTC timestamp to NY timestamp for technical assessment
            ts_ny = pd.Timestamp(current_time_ny)
            
            # Run technical assessment
            assessment_result, assessment_reason = meets_basic_buy_conditions(features, ts_ny)
            print(f"  - Technical assessment result: {assessment_result}")
            
            results.append({
                'symbol': symbol,
                'technical_assessment': assessment_result,
                'assessment_reason': assessment_reason,
                'features': features  # Store features for later use
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
    global SIMULATION_DATE_NY
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Trading Simulation with Exact Date (Historical Mode Only)")
    parser.add_argument('--simulation_date', type=str, required=True,
                       help='Simulation date and time in format YYYYMMDDHH24MI (e.g., 202507181400 for July 18, 2025 14:00 NY time)')
    
    args = parser.parse_args()

    # Parse the simulation date as NY time (what user inputs)
    simulated_now_ny = parse_simulation_date(args.simulation_date)
    SIMULATION_DATE_NY = simulated_now_ny
    
    # Convert to UTC for all database operations
    simulated_now_utc = ny_to_utc(simulated_now_ny)
    
    # Display mode information
    print(f"\n🕐 Trading Simulation (HISTORICAL MODE - EXACT DATE)")
    print("=" * 80)
    print(f"Simulated time New York: {simulated_now_ny.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Simulated time UTC: {simulated_now_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Market hours check: {is_market_hours(simulated_now_ny)}")
    print(f"Symbols: {', '.join(SYMBOLS)}")
    print("Using historical intraday data for simulation. Press Ctrl+C to stop.\n")
    
    # Ensure historical data is available for the simulation date
    print("🔍 Checking historical data availability... (bypassed because local file assumed to be complete)")

    while True:
        try:
            latest_data = fetch_latest_prices(simulated_now_utc)

            # Run technical assessment first
            technical_results = run_technical_assessment(latest_data, simulated_now_utc)
            
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
                        intra_day_df = fetch_intraday_data(row['symbol'], simulated_now_utc)

                        real_time_df = intra_day_df.iloc[[-1]] if not intra_day_df.empty else pd.DataFrame()

                        daily_df = fetch_daily_data(row['symbol'])
                        
                        # Calculate base features first
                        features = model_adapter.compute_features(real_time_df, intra_day_df, daily_df)
                        
                        # Debug: Check intraday data structure
                        print(f"🔍 Debug {row['symbol']}: intra_day_df shape={intra_day_df.shape}, columns={list(intra_day_df.columns)}")
                        if not intra_day_df.empty:
                            print(f"   Sample data: high_1min={intra_day_df['high_1min'].iloc[-1]}, low_1min={intra_day_df['low_1min'].iloc[-1]}")
                        
                        # Apply feature optimization to create missing features
                        optimized_features = optimize_features(features, intra_day_df=intra_day_df)
                        
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
                if tech_result['technical_assessment'] and 'features' in tech_result:
                    volume_percentile = f"{tech_result['features'].get('Volume_Percentile_Intraday', 0):.3f}"
                elif not tech_result['technical_assessment'] and 'features' in tech_result:
                    # Try to get volume percentile even for failed assessments
                    volume_percentile = f"{tech_result['features'].get('Volume_Percentile_Intraday', 0):.3f}"

                # Convert last_update from UTC to NY for display
                def format_last_update_print(last_update):
                    if not last_update:
                        return 'N/A'
                    try:
                        if last_update.tzinfo:
                            # Convert UTC to NY time for display
                            ny_time = utc_to_ny(last_update)
                            return ny_time.strftime('%H:%M:%S %Z')
                        else:
                            # If no timezone info, assume UTC and convert
                            utc_time = last_update.replace(tzinfo=UTC_TZ)
                            ny_time = utc_to_ny(utc_time)
                            return ny_time.strftime('%H:%M:%S %Z')
                    except Exception as e:
                        print(f"⚠️ Timezone conversion error for {last_update}: {e}")
                        return str(last_update.time()) if hasattr(last_update, 'time') else 'N/A'

                format_last_update_time = lambda ts: utc_to_ny(pd.to_datetime(ts, utc=True)).strftime('%H:%M:%S %Z') if ts else 'N/A'

                
                table.append([
                    row['symbol'],
                    f"${row['last_price']:.2f}" if row['last_price'] is not None else 'N/A',
                    format_last_update_time(row['last_update']),
                    tech_status,
                    prediction_text if tech_result['technical_assessment'] else 'N/A',
                    f"{model_result.get('probability', 0):.4f}" if tech_result['technical_assessment'] and 'probability' in model_result else 'N/A',
                    volume_percentile,
                    len(model_result.get('missing_features', [])) if tech_result['technical_assessment'] else 'N/A',
                    f"{model_result.get('threshold', 0):.3f}" if tech_result['technical_assessment'] and 'threshold' in model_result else 'N/A'
                ])
            
            # Display current time (simulated) in NY timezone
            print("(Historical Simulation Mode)")
            
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
            print(f"Historical simulation mode: Exact date ({simulated_now_ny.strftime('%Y-%m-%d %H:%M:%S %Z')})")
            print(f"Market hours: {is_market_hours(simulated_now_ny)}")
            print("========================================\n")

            time.sleep(FETCH_INTERVAL)
            simulated_now_utc += timedelta(seconds=55) 
            
        except KeyboardInterrupt:
            print("\n🛑 Simulation stopped by user.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(FETCH_INTERVAL)

if __name__ == "__main__":
    main()