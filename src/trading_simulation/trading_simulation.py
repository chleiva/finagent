import sqlite3
import time
import sys
import os
import argparse
from datetime import datetime, timedelta
import pandas as pd
from tabulate import tabulate
import pytz
from buy_sell import PositionManager
import uuid



simulation_id = str(uuid.uuid4())  # Generate unique simulation ID
current_cash = 10000.0  # Starting balance

# Add parent directories to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import technical assessment function
from feature_engineering.buy_label import meets_basic_buy_conditions
from feature_engineering.strict_higher_swing_lows import strict_higher_swing_lows
from feature_engineering.strict_higher_swing_lows import adaptive_higher_swing_lows

from inference.model_inference_adapter import ModelInferenceAdapter
from feature_engineering.feature_optimizer import optimize_features

DB_PATH = 'database/realtime_market_data.db'
SYMBOLS = ["NVDA", "MSFT", "AAPL", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "NFLX", "COST"]
FETCH_INTERVAL = 0


position_manager = PositionManager(DB_PATH)

# Get timezone objects
try:
    from zoneinfo import ZoneInfo
    NY_TZ = ZoneInfo("America/New_York")
    UTC_TZ = ZoneInfo("UTC")
except ImportError:
    NY_TZ = pytz.timezone("America/New_York")
    UTC_TZ = pytz.UTC

class TimeManager:
    @staticmethod
    def parse_simulation_date(date_str):
        """Parse simulation date string in format YYYYMMDDHH24MI to datetime object in NY timezone"""
        if len(date_str) != 12:
            raise ValueError("Simulation date must be in format YYYYMMDDHH24MI (12 digits)")
        
        try:
            year, month, day, hour, minute = int(date_str[0:4]), int(date_str[4:6]), int(date_str[6:8]), int(date_str[8:10]), int(date_str[10:12])
            return datetime(year, month, day, hour, minute, tzinfo=NY_TZ)
        except ValueError as e:
            raise ValueError(f"Invalid simulation date format: {e}")

    @staticmethod
    def convert_time(dt, target_tz='UTC'):
        """Universal timezone converter"""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=NY_TZ if target_tz == 'UTC' else UTC_TZ)
        
        if target_tz == 'UTC':
            return dt.astimezone(UTC_TZ)
        elif target_tz == 'NY':
            return dt.astimezone(NY_TZ)
        else:
            return dt

    @staticmethod
    def is_market_hours(dt_ny):
        """Check if given NY time is during market hours (9:30 AM - 4:00 PM ET, weekdays)"""
        if dt_ny.weekday() >= 5:  # Weekend
            return False
        
        market_open = dt_ny.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = dt_ny.replace(hour=16, minute=0, second=0, microsecond=0)
        return market_open <= dt_ny <= market_close

class DataManager:
    def __init__(self, db_path):
        self.db_path = db_path
    
    def _execute_query(self, query, params):
        """Execute database query and return results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        return rows
    
    def fetch_intraday_data(self, symbol, simulated_now_utc):
        """Fetch intraday minute data for a symbol"""
        simulation_date = simulated_now_utc.date()
        end_time_utc = simulated_now_utc - timedelta(minutes=1)
        
        query = '''
            SELECT ts_event_price, open_1min, high_1min, low_1min, close_1min, volume_1min,
                   bid_px_00, ask_px_00, bid_sz_00, ask_sz_00
            FROM historical_simulation_intraday_data
            WHERE symbol_price = ? AND simulation_date = ? AND datetime(ts_event_price) <= datetime(?)
            ORDER BY datetime(ts_event_price) ASC
        '''
        
        rows = self._execute_query(query, (symbol, simulation_date, end_time_utc.isoformat()))
        
        if not rows:
            raise ValueError(f"No intraday data found for {symbol}")
        
        return self._create_intraday_dataframe(rows, symbol, end_time_utc)
    
    def fetch_daily_data(self, symbol, lookback_days=60):
        """Fetch daily data for a symbol"""
        query = '''
            SELECT bar_date, open, high, low, close, volume
            FROM daily_summary_data
            WHERE symbol = ?
            ORDER BY bar_date DESC
            LIMIT ?
        '''
        
        rows = self._execute_query(query, (symbol, lookback_days))
        
        if not rows:
            raise ValueError(f"No daily data found for {symbol}")
        
        return self._create_daily_dataframe(rows, symbol)
    
    def _create_intraday_dataframe(self, rows, symbol, end_time_utc):
        """Create intraday DataFrame from database rows"""
        data = []
        for row in rows:
            ts_event_price, open_price, high, low, close_1min, volume, bid_px_00, ask_px_00, bid_sz_00, ask_sz_00 = row
            data.append({
                'close_1min': close_1min, 'close': close_1min,
                'open_1min': open_price, 'open': open_price,
                'high_1min': high, 'high': high,
                'low_1min': low, 'low': low,
                'volume_1min': float(volume) if volume else 0.0,
                'volume': float(volume) if volume else 0.0,
                'ts_event_clean': ts_event_price,
                'symbol_price': symbol,
                'bidPrice': float(bid_px_00) if bid_px_00 else None,
                'askPrice': float(ask_px_00) if ask_px_00 else None,
                'bidSize': float(bid_sz_00) if bid_sz_00 else None,
                'askSize': float(ask_sz_00) if ask_sz_00 else None,
                'lastPrice': close_1min,
                'timestamp': ts_event_price
            })
        
        df = pd.DataFrame(data)
        df['ts_event_clean'] = pd.to_datetime(df['ts_event_clean'], utc=True)
        df = df[df['ts_event_clean'] <= end_time_utc]
        return df.sort_values('ts_event_clean')
    
    def _create_daily_dataframe(self, rows, symbol):
        """Create daily DataFrame from database rows"""
        data = []
        for row in rows:
            bar_date, open_price, high, low, close_1min, volume = row
            data.append({
                'date': bar_date, 'open': open_price, 'high': high, 'low': low,
                'close': close_1min, 'volume': float(volume) if volume else 0.0,
                'symbol': symbol
            })
        
        df = pd.DataFrame(data)
        return df.sort_values('date')

class ResultBuilder:
    @staticmethod
    def build_result(symbol, **kwargs):
        """Standard result format"""
        return {'symbol': symbol, **kwargs}
    
    @staticmethod
    def build_error_result(symbol, error):
        """Standard error format"""
        return ResultBuilder.build_result(symbol, technical_assessment=False, 
                                         assessment_reason=f'error: {str(error)}')

class TradingProcessor:
    def __init__(self, data_manager, model_adapter):
        self.data_manager = data_manager
        self.model_adapter = model_adapter
        self.time_manager = TimeManager()
    
    
    def process_symbols(self, symbols, simulated_now_utc):
        global current_cash
        global simulation_id
        """Process all symbols - technical assessment and model inference combined with swing low detection"""
        current_time_ny = self.time_manager.convert_time(simulated_now_utc, 'NY')
        
        # Fetch latest prices for all symbols
        latest_data = self._fetch_latest_prices(symbols, simulated_now_utc)
        
        results = []
        for symbol_data in latest_data:
            symbol = symbol_data['symbol']
            
            try:
                # Check market hours
                if not self.time_manager.is_market_hours(current_time_ny):
                    results.append(ResultBuilder.build_result(
                        symbol, technical_assessment=False, 
                        assessment_reason='outside_market_hours', model_result={}))
                    continue
                
                # Get all data for this symbol
                intra_day_df = self.data_manager.fetch_intraday_data(symbol, simulated_now_utc)
                daily_df = self.data_manager.fetch_daily_data(symbol)
                real_time_df = intra_day_df.iloc[[-1]] if not intra_day_df.empty else pd.DataFrame()
                
                # Calculate features
                features = self.model_adapter.compute_features(real_time_df, intra_day_df, daily_df)

                #print(f"DEBUG features: {features}")

                # ===== SWING LOW DETECTION =====
                swing_low_minutes = []
                is_swing_low = False
                swing_low_match_info = "not_detected"
                
                # Perform swing low detection if we have sufficient intraday data
                if not intra_day_df.empty and len(intra_day_df) > 100:  # Need sufficient data points
                    try:
                        print(f"\n🔍 PERFORMING SWING LOW DETECTION FOR {symbol}...")
                        
                        # Prepare data for swing detection (ensure proper datetime index and columns)
                        swing_detection_df = intra_day_df.copy()
                        
                        # Ensure we have required columns - adapt to your actual column names
                        price_col = 'close'  # Adjust based on your data structure
                        vwap_col = 'vwap'    # Adjust based on your data structure
                        
                        # Check if columns exist, if not try common alternatives
                        if price_col not in swing_detection_df.columns:
                            for alt_col in ['Close', 'close_price', 'last_price', 'price']:
                                if alt_col in swing_detection_df.columns:
                                    price_col = alt_col
                                    break
                        
                        if vwap_col not in swing_detection_df.columns:
                            for alt_col in ['VWAP', 'vwap_price', 'volume_weighted_price']:
                                if alt_col in swing_detection_df.columns:
                                    vwap_col = alt_col
                                    break
                        
                        # Only proceed if we have the required columns
                        if price_col in swing_detection_df.columns:
                            print(f"   📊 Using price column: {price_col}")
                            if vwap_col in swing_detection_df.columns:
                                print(f"   📊 Using VWAP column: {vwap_col}")
                            else:
                                print(f"   ⚠️  VWAP column not found, using price-only detection")
                                vwap_col = None
                            
                            # Run STRICT higher swing low detection
                            print(f"   🔍 Detecting STRICT higher swing lows...")
                            strict_swing_lows_df = strict_higher_swing_lows(
                                swing_detection_df,
                                price_col=price_col,
                                window=4,
                                max_gap_minutes=40,
                                vwap_col=vwap_col,
                                vwap_tolerance=0.5,
                                debug=False
                            )
                            
                            # Run ADAPTIVE higher swing low detection
                            print(f"   🔍 Detecting ADAPTIVE higher swing lows...")
                            adaptive_swing_lows_df = adaptive_higher_swing_lows(
                                swing_detection_df,
                                price_col=price_col,
                                window=4,
                                max_gap_minutes=40,
                                vwap_col=vwap_col,
                                vwap_tolerance=0.5,
                                debug=False
                            )
                            
                            # Merge results from both methods
                            print(f"   📊 Merging results from both methods...")
                            all_swing_lows_df = pd.DataFrame()
                            
                            if not strict_swing_lows_df.empty:
                                strict_copy = strict_swing_lows_df.copy()
                                strict_copy['detection_method'] = 'strict'
                                all_swing_lows_df = pd.concat([all_swing_lows_df, strict_copy])
                                
                            if not adaptive_swing_lows_df.empty:
                                adaptive_copy = adaptive_swing_lows_df.copy()
                                adaptive_copy['detection_method'] = 'adaptive'
                                all_swing_lows_df = pd.concat([all_swing_lows_df, adaptive_copy])

                            # Remove duplicate timestamps and merge detection methods
                            if not all_swing_lows_df.empty:
                                # Sort by timestamp
                                all_swing_lows_df = all_swing_lows_df.sort_index()
                                
                                # Group by timestamp and combine detection methods
                                merged_swing_lows = []
                                current_timestamp = None
                                current_group = []
                                
                                for timestamp, row in all_swing_lows_df.iterrows():
                                    if current_timestamp is None or timestamp == current_timestamp:
                                        current_timestamp = timestamp
                                        current_group.append(row)
                                    else:
                                        # Process the previous group
                                        if current_group:
                                            merged_row = current_group[0].copy()
                                            methods = [r['detection_method'] for r in current_group]
                                            merged_row['detection_method'] = '+'.join(sorted(set(methods)))
                                            merged_swing_lows.append((current_timestamp, merged_row))
                                        
                                        # Start new group
                                        current_timestamp = timestamp
                                        current_group = [row]
                                
                                # Process the last group
                                if current_group:
                                    merged_row = current_group[0].copy()
                                    methods = [r['detection_method'] for r in current_group]
                                    merged_row['detection_method'] = '+'.join(sorted(set(methods)))
                                    merged_swing_lows.append((current_timestamp, merged_row))
                                
                                # Create final DataFrame
                                if merged_swing_lows:
                                    final_swing_lows_df = pd.DataFrame([row for _, row in merged_swing_lows], 
                                                                    index=[ts for ts, _ in merged_swing_lows])
                                    
                                    # Convert to minute timestamps for comparison
                                    swing_low_times = final_swing_lows_df.index
                                    for swing_time in swing_low_times:
                                        swing_time_utc = swing_time.tz_convert('UTC')
                                        rounded_minute_utc = swing_time_utc.floor('min')
                                        swing_low_minutes.append(rounded_minute_utc)
                                    
                                    # Remove duplicates and sort
                                    swing_low_minutes = sorted(list(set(swing_low_minutes)))
                                    
                                    print(f"   ✅ Found {len(final_swing_lows_df)} total swing lows")
                                    print(f"   🎯 Methods breakdown:")
                                    method_counts = final_swing_lows_df['detection_method'].value_counts()
                                    for method, count in method_counts.items():
                                        print(f"      {method}: {count}")
                                else:
                                    print(f"   ❌ No swing lows detected after merging")
                            else:
                                print(f"   ❌ No swing lows detected by either method")
                        else:
                            print(f"   ❌ Required price column not found in data")
                            
                    except Exception as swing_error:
                        print(f"   ❌ Error in swing low detection: {swing_error}")
                        swing_low_minutes = []
                else:
                    print(f"   ⚠️  Insufficient data for swing low detection ({len(intra_day_df)} rows)")

                # ===== CHECK IF CURRENT TIME IS A SWING LOW =====
                ts_ny = pd.Timestamp(current_time_ny)
                if swing_low_minutes:
                    # Convert current time to UTC and round to minute for comparison
                    ts_utc = ts_ny.tz_convert('UTC') if ts_ny.tz is not None else ts_ny
                    current_minute = ts_utc.floor('min')
                    
                    # Check if current minute matches any swing low minute
                    for i, swing_minute in enumerate(swing_low_minutes):
                        # Allow small time differences (within 1 minute)
                        time_diff = abs((current_minute - swing_minute).total_seconds())
                        if time_diff <= 60:  # Within 1 minute
                            is_swing_low = True
                            swing_low_match_info = f"matched_swing_low_{i+1}_diff_{time_diff}s"
                            break
                    
                    if not is_swing_low:
                        swing_low_match_info = f"no_match_checked_{len(swing_low_minutes)}_swing_lows"
                else:
                    swing_low_match_info = "no_swing_lows_detected"

                # ===== TECHNICAL ASSESSMENT WITH SWING LOW REQUIREMENT =====
                # First run the basic technical conditions
                initial_assessment_result, initial_assessment_reason = meets_basic_buy_conditions(features, ts_ny)
                
                # Enhanced logging with swing low status
                print(f"\n🔍 ASSESSMENT FOR {symbol} at {ts_ny}:")
                print(f"   📊 Initial technical conditions: {initial_assessment_result} ({initial_assessment_reason})")
                print(f"   🎯 Swing low status: {is_swing_low} ({swing_low_match_info})")

                # ===== CRITICAL: REQUIRE SWING LOW FOR TECHNICAL CONDITIONS TO PASS =====
                if initial_assessment_result and not is_swing_low:
                    print(f"   ❌ REJECTED: Technical conditions met but NOT at a swing low")
                    assessment_result = False
                    assessment_reason = "rejected_not_at_swing_low"
                elif initial_assessment_result and is_swing_low:
                    print(f"   ✅ APPROVED: Technical conditions met AND at a swing low!")
                    assessment_result = True
                    assessment_reason = f"{initial_assessment_reason}_at_swing_low"
                elif not initial_assessment_result and is_swing_low:
                    print(f"   ❌ At swing low but technical conditions failed: {initial_assessment_reason}")
                    assessment_result = False
                    assessment_reason = f"at_swing_low_but_{initial_assessment_reason}"
                else:
                    print(f"   ❌ Neither swing low nor technical conditions met")
                    assessment_result = False
                    assessment_reason = f"no_swing_low_and_{initial_assessment_reason}"

                print(f"   🎯 FINAL ASSESSMENT: {assessment_result} ({assessment_reason})")
                
                # Model inference (only if technical assessment passes - which now requires swing low)
                model_result = {}
                if assessment_result:
                    print(f"   🚀 Running model inference for swing low signal...")
                    model_result = self._run_model_inference(symbol, features, intra_day_df, real_time_df, daily_df)
                else:
                    print(f"   ⏸️  Skipping model inference - requirements not met")

                # Add swing low information to the result
                enhanced_features = features.copy() if features else {}
                enhanced_features['is_swing_low'] = is_swing_low
                enhanced_features['swing_low_info'] = swing_low_match_info
                enhanced_features['swing_lows_detected'] = len(swing_low_minutes)
                enhanced_features['initial_technical_passed'] = initial_assessment_result
                                
                results.append(ResultBuilder.build_result(
                    symbol, technical_assessment=assessment_result,
                    assessment_reason=assessment_reason, features=enhanced_features,
                    model_result=model_result, symbol_data=symbol_data))

                
            except Exception as e:
                print(f"❌ Error processing {symbol}: {e}")
                results.append(ResultBuilder.build_error_result(symbol, e))
        
        
        # In your results processing loop - now only swing low signals get through
        for result in results:

            if (
                    result['technical_assessment']  # This now REQUIRES swing low
                    and result.get('model_result', {}).get('prediction', 0) == 1
                    and result.get('model_result', {}).get('probability', 0) > 0.7
                ):

                # At this point, we know it's DEFINITELY at a swing low
                swing_low_info = result.get('features', {}).get('swing_low_info', 'unknown')
                
                print(f"🎯 SWING LOW BUY SIGNAL for {result['symbol']}! ({swing_low_info})")
                
                # BUY signal detected - guaranteed to be at swing low
                current_cash = position_manager.buy(
                    simulation_id=simulation_id,
                    symbol=result['symbol'],
                    current_price=result['symbol_data']['last_price'],
                    features=result['features'],
                    simulation_time_utc=simulated_now_utc,
                    current_cash=current_cash
                )
            elif result.get('features', {}).get('initial_technical_passed', False):
                # This means technical conditions were good but not at swing low
                print(f"⚠️  Skipped {result['symbol']}: Good technicals but not at swing low")
            
            # Always check for sell conditions
            current_cash = position_manager.sell(
                symbol=result['symbol'],
                simulation_time_utc=simulated_now_utc,
                current_cash=current_cash,
                simulation_id=simulation_id,
                current_price=result['symbol_data']['last_price']
            )
                
        
        return results
    
    def _fetch_latest_prices(self, symbols, simulated_now_utc):
        """Fetch latest prices for all symbols"""
        data = []
        for symbol in symbols:
            try:
                intra_day_df = self.data_manager.fetch_intraday_data(symbol, simulated_now_utc)
                if not intra_day_df.empty:
                    latest = intra_day_df.iloc[-1]
                    data.append({
                        'symbol': symbol, 'last_price': latest['close_1min'],
                        'bid_price': latest.get('bidPrice'), 'ask_price': latest.get('askPrice'),
                        'bid_size': latest.get('bidSize'), 'ask_size': latest.get('askSize'),
                        'volume': latest['volume_1min'], 'last_update': latest['ts_event_clean']
                    })
                else:
                    data.append({k: None for k in ['last_price', 'bid_price', 'ask_price', 'bid_size', 'ask_size', 'volume', 'last_update']})
                    data[-1]['symbol'] = symbol
            except Exception:
                data.append({k: None for k in ['last_price', 'bid_price', 'ask_price', 'bid_size', 'ask_size', 'volume', 'last_update']})
                data[-1]['symbol'] = symbol
        return data
    
    def _run_model_inference(self, symbol, features, intra_day_df, real_time_df, daily_df):
        """Run model inference for a symbol"""
        try:
            # Apply feature optimization
            optimized_features = optimize_features(features, intra_day_df=intra_day_df)
            
            # Get model and prepare features
            model = self.model_adapter.get_model(symbol)
            model_features = model.features
            
            feature_vector = []
            missing_features = []
            for feat in model_features:
                if feat in optimized_features:
                    feature_vector.append(optimized_features[feat])
                else:
                    feature_vector.append(0.0)
                    missing_features.append(feat)
            
            X = pd.DataFrame([feature_vector], columns=model_features)
            
            # Run prediction
            threshold = 0.9
            prediction, probability = model.predict(X, threshold=threshold)
            
            return {
                'prediction': int(prediction[0]), 'probability': float(probability[0]),
                'missing_features': missing_features, 'threshold': threshold
            }
        except Exception as e:
            return {'prediction': 0, 'probability': 0.0, 'missing_features': [str(e)], 'threshold': 0.9, 'error': str(e)}

class DisplayManager:
    def __init__(self, time_manager):
        self.time_manager = time_manager
    
    def display_results(self, results, simulated_now_ny):
        """Display results in table format"""
        global current_cash
        table = []
        for result in results:
            symbol = result['symbol']
            symbol_data = result.get('symbol_data', {})
            model_result = result.get('model_result', {})
            
            prediction_text = "BUY" if model_result.get('prediction', 0) == 1 else "NO BUY"
            tech_status = "✅ PASS" if result['technical_assessment'] else f"❌ {result['assessment_reason']}"
            
            # Get volume percentile
            volume_percentile = "N/A"
            if 'features' in result:
                volume_percentile = f"{result['features'].get('Volume_Percentile_Intraday', 0):.3f}"
            
            # Format last update time
            last_update = symbol_data.get('last_update')
            formatted_time = self._format_time(last_update) if last_update else 'N/A'
            
            table.append([
                symbol,
                f"${symbol_data.get('last_price', 0):.2f}" if symbol_data.get('last_price') else 'N/A',
                formatted_time,
                tech_status,
                prediction_text if result['technical_assessment'] else 'N/A',
                f"{model_result.get('probability', 0):.2f}" if result['technical_assessment'] and 'probability' in model_result else 'N/A',
                volume_percentile,
                len(model_result.get('missing_features', [])) if result['technical_assessment'] else 'N/A',
                f"{model_result.get('threshold', 0):.2f}" if result['technical_assessment'] and 'threshold' in model_result else 'N/A'
            ])
        
        print("(Historical Simulation Mode)")
        print(tabulate(table, headers=["Symbol", "Last Price", "Last Update", "Tech Assessment", 
                                     "Prediction", "Probability", "Vol %", "Missing Features", "Threshold"], 
                      tablefmt="fancy_grid"))
        
        self._print_summary(results, simulated_now_ny)

        position_manager.print_position_summary(simulation_id, current_cash)
    
    def _format_time(self, timestamp):
        """Format timestamp for display"""
        try:
            ny_time = self.time_manager.convert_time(pd.to_datetime(timestamp, utc=True), 'NY')
            return ny_time.strftime('%H:%M:%S %Z')
        except:
            return 'N/A'
    
    def _print_summary(self, results, simulated_now_ny):
        """Print summary section"""
        buy_signals = [r['symbol'] for r in results 
                      if r['technical_assessment'] and r.get('model_result', {}).get('prediction', 0) == 1]
        num_tech_pass = sum(1 for r in results if r['technical_assessment'])
        
        print("\n================ SUMMARY ================")
        print(f"BUY SIGNALS: {', '.join(buy_signals) if buy_signals else 'None'}")
        print(f"Number of symbols passing technical assessment: {num_tech_pass}")
        print(f"Number of BUY signals: {len(buy_signals)}")
        print(f"Historical simulation mode: Exact date ({simulated_now_ny.strftime('%Y-%m-%d %H:%M:%S %Z')})")
        print(f"Market hours: {TimeManager.is_market_hours(simulated_now_ny)}")
        print("========================================\n")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Trading Simulation with Exact Date (Historical Mode Only)")
    parser.add_argument('--simulation_date', type=str, required=True,
                       help='Simulation date and time in format YYYYMMDDHH24MI')
    args = parser.parse_args()

    # Initialize components
    time_manager = TimeManager()
    data_manager = DataManager(DB_PATH)
    model_adapter = ModelInferenceAdapter(model_index_path="model_artifacts/model_index.csv")
    processor = TradingProcessor(data_manager, model_adapter)
    display_manager = DisplayManager(time_manager)
    
    # Parse simulation time
    simulated_now_ny = time_manager.parse_simulation_date(args.simulation_date)
    simulated_now_utc = time_manager.convert_time(simulated_now_ny, 'UTC')
    
    # Display startup info
    print(f"\n🕐 Trading Simulation (HISTORICAL MODE - EXACT DATE)")
    print("=" * 80)
    print(f"Simulated time New York: {simulated_now_ny.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Simulated time UTC: {simulated_now_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Market hours check: {time_manager.is_market_hours(simulated_now_ny)}")
    print(f"Symbols: {', '.join(SYMBOLS)}")
    print("Using historical intraday data for simulation. Press Ctrl+C to stop.\n")

    # Main simulation loop
    while True:
        try:

            # Convert simulated_now_utc to NY time for market check
            current_time_ny = time_manager.convert_time(simulated_now_utc, 'NY')
            if current_time_ny.hour > 16 or (current_time_ny.hour == 15 and current_time_ny.minute > 55):
                print(f"\n🛑 Exiting: Simulated time {current_time_ny.strftime('%Y-%m-%d %H:%M:%S %Z')} is approching NYSE close (16:00).")
                break
            # Process all symbols
            results = processor.process_symbols(SYMBOLS, simulated_now_utc)
            
            # Display results
            current_time_ny = time_manager.convert_time(simulated_now_utc, 'NY')
            display_manager.display_results(results, current_time_ny)
            
            # Advance time
            time.sleep(FETCH_INTERVAL)
            simulated_now_utc += timedelta(seconds=55)
            
        except KeyboardInterrupt:
            print("\n🛑 Simulation stopped by user.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(FETCH_INTERVAL)
            exit()

if __name__ == "__main__":
    main()