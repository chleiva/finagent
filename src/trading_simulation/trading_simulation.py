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
from inference.model_inference_adapter import ModelInferenceAdapter
from feature_engineering.feature_optimizer import optimize_features

DB_PATH = 'database/realtime_market_data.db'
SYMBOLS = ["NVDA", "MSFT", "AAPL", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "NFLX", "COST"]
FETCH_INTERVAL = 5

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
        """Process all symbols - technical assessment and model inference combined"""
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
                
                # Technical assessment
                ts_ny = pd.Timestamp(current_time_ny)
                assessment_result, assessment_reason = meets_basic_buy_conditions(features, ts_ny)
                
                # Model inference (only if technical assessment passes)
                model_result = {}
                if assessment_result:
                    model_result = self._run_model_inference(symbol, features, intra_day_df, real_time_df, daily_df)
                
                results.append(ResultBuilder.build_result(
                    symbol, technical_assessment=assessment_result,
                    assessment_reason=assessment_reason, features=features,
                    model_result=model_result, symbol_data=symbol_data))
                
            except Exception as e:
                results.append(ResultBuilder.build_error_result(symbol, e))
        
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
            threshold = 0.5
            prediction, probability = model.predict(X, threshold=threshold)
            
            return {
                'prediction': int(prediction[0]), 'probability': float(probability[0]),
                'missing_features': missing_features, 'threshold': threshold
            }
        except Exception as e:
            return {'prediction': 0, 'probability': 0.0, 'missing_features': [str(e)], 'threshold': 0.5, 'error': str(e)}

class DisplayManager:
    def __init__(self, time_manager):
        self.time_manager = time_manager
    
    def display_results(self, results, simulated_now_ny):
        """Display results in table format"""
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
                f"{model_result.get('probability', 0):.4f}" if result['technical_assessment'] and 'probability' in model_result else 'N/A',
                volume_percentile,
                len(model_result.get('missing_features', [])) if result['technical_assessment'] else 'N/A',
                f"{model_result.get('threshold', 0):.3f}" if result['technical_assessment'] and 'threshold' in model_result else 'N/A'
            ])
        
        print("(Historical Simulation Mode)")
        print(tabulate(table, headers=["Symbol", "Last Price", "Last Update", "Tech Assessment", 
                                     "Prediction", "Probability", "Vol %", "Missing Features", "Threshold"], 
                      tablefmt="fancy_grid"))
        
        self._print_summary(results, simulated_now_ny)
    
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

if __name__ == "__main__":
    main()