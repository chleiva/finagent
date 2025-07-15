import pandas as pd
import numpy as np
from datetime import datetime, time
from typing import Dict, Optional, Tuple

def meets_basic_buy_conditions(features: Dict, current_time: pd.Timestamp) -> Tuple[bool, str]:
    """
    Check if basic conditions are met for considering a buy signal.
    
    Args:
        features: Dictionary containing calculated features
        current_time: Current timestamp
        
    Returns:
        Tuple of (condition_met: bool, reason: str)
    """
    # Market hours (9:30 AM to 4:00 PM ET)
    market_open = time(9, 30)
    market_close = time(16, 0)
    
    current_time_only = current_time.time()
    
    # Check if it's during market hours
    if current_time_only < market_open or current_time_only > market_close:
        return False, "outside_market_hours"
    
    # Must be at least 30 minutes after market open
    min_time_after_open = time(10, 0)  # 9:30 + 30 min = 10:00
    if current_time_only < min_time_after_open:
        return False, "too_early_after_open"
    
    # Must be at least 1 hour before market close
    max_time_before_close = time(15, 0)  # 16:00 - 60 min = 15:00
    if current_time_only > max_time_before_close:
        return False, "too_close_to_close"
    
    # Check if we have valid R-levels
    required_fields = ['Current_Price', '1R_Stop_Loss', '1R_Target', '2R_Target', 'ATR_Distance']
    for field in required_fields:
        if field not in features or pd.isna(features[field]) or features[field] <= 0:
            return False, f"missing_or_invalid_{field.lower()}"
    
    # Basic technical conditions
    current_price = features['Current_Price']
    stop_loss = features['1R_Stop_Loss']
    target_1r = features['1R_Target']
    
    # Ensure stop loss is below current price and target is above
    if stop_loss >= current_price:
        return False, "stop_loss_above_price"
    if target_1r <= current_price:
        return False, "target_below_price"

    # Volume should be at least 40th percentile of the day
    if features.get('Volume_Percentile_Intraday', 0) < 0.4:
        return False, "volume_percentile_too_low"
    
    # Check for reasonable ATR distance (not too tight or too wide)
    atr_distance = features['ATR_Distance']
    atr_pct = (atr_distance / current_price) * 100
    
    # ATR should be between 0.01% and 3% of current price
    # Lower bound: 0.01% allows for very liquid stocks like AAPL
    # Upper bound: 3% allows for more volatile stocks
    if atr_pct < 0.01:
        return False, "atr_too_tight"
    if atr_pct > 3.0:
        return False, "atr_too_wide"
    
    # Optional: Add more technical conditions
    # Example: Check if we have volume data and it's reasonable
    if 'volume' in features and features['volume'] > 0:
        # Volume should be reasonable (not too low)
        if features['volume'] < 100:  # Minimum volume threshold
            return False, "volume_too_low"
    
    # Optional: Check spread conditions
    if 'bidPrice' in features and 'askPrice' in features:
        bid = features['bidPrice']
        ask = features['askPrice']
        if bid > 0 and ask > 0:
            spread = ask - bid
            spread_pct = (spread / current_price) * 100
            # Spread should be reasonable (not too wide)
            if spread_pct > 0.5:  # Max 0.5% spread
                return False, "spread_too_wide"
    
    return True, "conditions_met"

def evaluate_buy_signal(features: Dict, real_time_df: pd.DataFrame, intra_day_df: pd.DataFrame, daily_df: pd.DataFrame, full_intra_day_df: pd.DataFrame = None, max_time_window_minutes: int = 240) -> Dict:
    """
    Evaluate if current minute represents a good buy opportunity based on future price action.
    
    Args:
        features: Dictionary containing calculated features for current minute
        real_time_df: Current minute's real-time data
        intra_day_df: Filtered intraday data used for features (up to current minute)
        daily_df: Daily historical data
        full_intra_day_df: Complete unfiltered intraday data for the entire day (contains future prices)
        max_time_window_minutes: Maximum time window to check for targets (default: 240 minutes = 4 hours)
        
    Returns:
        Dictionary with buy signal results: {buy: bool, R: int, time_minutes: int, reason: str}
    """
    # Initialize result
    result = {
        'buy': False,
        'R': 0,
        'time_minutes': 0,
        'reason': 'no_signal'
    }
    
    # Get current time from real_time_df
    current_time = real_time_df['timestamp'].iloc[0] if 'timestamp' in real_time_df.columns else pd.Timestamp.now()
    
    # Check basic conditions first
    conditions_met, condition_reason = meets_basic_buy_conditions(features, current_time)
    if not conditions_met:
        result['reason'] = condition_reason
        return result
    
    # Check if we're too close to market close (30 minutes before)
    market_close = time(16, 0)
    close_cutoff = time(15, 30)  # 30 minutes before close
    
    if current_time.time() > close_cutoff:
        result['reason'] = 'market_near_close'
        return result
    
    # Get R-levels from features
    current_price = features['Current_Price']
    stop_loss = features['1R_Stop_Loss']
    target_1r = features['1R_Target']
    target_2r = features['2R_Target']
    target_3r = features.get('3R_Target', target_2r + (target_2r - target_1r))
    target_4r = features.get('4R_Target', target_3r + (target_2r - target_1r))
    
    # Use full_intra_day_df if provided, otherwise fall back to intra_day_df
    data_for_future = full_intra_day_df if full_intra_day_df is not None else intra_day_df
    
    # Extract future prices from the complete intraday data
    # Determine the time column name
    time_col = 'ts_event_clean' if 'ts_event_clean' in data_for_future.columns else 'timestamp'
    price_col = 'close_1min' if 'close_1min' in data_for_future.columns else 'lastPrice'
    
    if time_col not in data_for_future.columns:
        result['reason'] = 'no_timestamp_column'
        return result
    
    if price_col not in data_for_future.columns:
        result['reason'] = 'no_price_column'
        return result
    
    # Get future data (after current minute) - THIS IS THE KEY DIFFERENCE
    future_data = data_for_future[data_for_future[time_col].dt.floor('min') > current_time].copy()
    
    if future_data.empty:
        result['reason'] = 'no_future_data'
        return result
    
    # Sort by timestamp to ensure chronological order
    future_data = future_data.sort_values(time_col)
    
    # Group by minute and get the close price for each minute
    future_data['minute'] = future_data[time_col].dt.floor('min')
    future_prices = future_data.groupby('minute')[price_col].last().reset_index()
    future_prices.columns = ['timestamp', 'price']
    
    # Apply time window limit if specified
    if max_time_window_minutes > 0:
        max_time = current_time + pd.Timedelta(minutes=max_time_window_minutes)
        future_prices = future_prices[future_prices['timestamp'] <= max_time]
        
        if future_prices.empty:
            result['reason'] = 'no_data_in_time_window'
            return result
    
    # Track what happens minute by minute
    reached_2r = False
    reached_3r = False
    reached_4r = False
    hit_stop = False
    
    max_r_achieved = 0
    time_to_target = 0
    max_time_reached = False
    
    for idx, row in future_prices.iterrows():
        price = row['price']
        minutes_elapsed = (row['timestamp'] - current_time).total_seconds() / 60
        
        # Check if we've exceeded the maximum time window
        if max_time_window_minutes > 0 and minutes_elapsed > max_time_window_minutes:
            max_time_reached = True
            break
        
        # Check if we hit stop loss (-1R)
        if price <= stop_loss:
            hit_stop = True
            result['reason'] = 'reached_negative_1r'
            break
        
        # Check R-levels achieved
        if not reached_2r and price >= target_2r:
            reached_2r = True
            max_r_achieved = 2
            time_to_target = minutes_elapsed
            
        elif reached_2r and not reached_3r and price >= target_3r:
            reached_3r = True
            max_r_achieved = 3
            time_to_target = minutes_elapsed
            
        elif reached_3r and not reached_4r and price >= target_4r:
            reached_4r = True
            max_r_achieved = 4
            time_to_target = minutes_elapsed
        
        # Stop if we've reached 4R or if we're close to market close
        if reached_4r or row['timestamp'].time() >= close_cutoff:
            break
    
    # Determine final result
    if hit_stop and not reached_2r:
        # Hit stop loss before reaching 2R (reason already set above)
        result['buy'] = False
    elif reached_2r and not hit_stop:
        # Successfully reached at least 2R without hitting stop
        result['buy'] = True
        result['R'] = max_r_achieved
        result['time_minutes'] = int(time_to_target)
        result['reason'] = f'reached_{max_r_achieved}r_success'
    elif not reached_2r and not hit_stop:
        # Neither reached 2R nor hit stop
        if max_time_reached:
            result['reason'] = 'timeout_no_2r'
        else:
            result['reason'] = 'end_of_day_no_2r'
    
    return result

def evaluate_all_buy_signals_batch(features_df: pd.DataFrame, intra_day_df: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate buy signals for all minutes in the features dataframe (batch processing).
    
    Args:
        features_df: DataFrame with calculated features for each minute
        intra_day_df: Complete intraday data for the symbol/date
        daily_df: Daily historical data
        
    Returns:
        DataFrame with original features plus buy signal evaluation results
    """
    results = []
    
    print(f"Evaluating buy signals for {len(features_df)} minutes...")
    
    for idx, row in features_df.iterrows():
        # Convert row to dictionary for easier handling
        features = row.to_dict()
        
        # Create a mock real_time_df for this minute
        real_time_df = pd.DataFrame({
            'timestamp': [pd.to_datetime(row['timestamp'])],
            'bidPrice': [features.get('bidPrice', 0)],
            'askPrice': [features.get('askPrice', 0)],
            'lastPrice': [features.get('lastPrice', features.get('Current_Price', 0))],
            'volume': [features.get('volume', 0)]
        })
        
        # Evaluate buy signal
        signal_result = evaluate_buy_signal(features, real_time_df, intra_day_df, daily_df)
        
        # Combine original features with signal results
        combined_result = {**features, **signal_result}
        results.append(combined_result)
        
        # Print progress for every 100 minutes
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(features_df)} minutes...")
    
    return pd.DataFrame(results)

def prepare_price_data_from_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare price data from features dataframe for buy signal evaluation.
    
    Args:
        features_df: DataFrame with features including Current_Price and timestamp
        
    Returns:
        DataFrame with timestamp and price columns
    """
    price_data = features_df[['timestamp', 'Current_Price']].copy()
    price_data.columns = ['timestamp', 'price']
    price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
    
    return price_data.sort_values('timestamp')

# Example usage function
def run_buy_signal_analysis(features_file: str, output_file: str = None):
    """
    Run complete buy signal analysis on a features file.
    
    Args:
        features_file: Path to CSV file with calculated features
        output_file: Optional path to save results (default: adds _with_signals suffix)
    """
    print(f"Loading features from {features_file}...")
    
    # Load features
    features_df = pd.read_csv(features_file)
    
    # Prepare price data
    price_data = prepare_price_data_from_features(features_df)
    
    # Evaluate buy signals
    results_df = evaluate_all_buy_signals(features_df, price_data)
    
    # Generate output filename if not provided
    if output_file is None:
        base_name = features_file.replace('.csv', '')
        output_file = f"{base_name}_with_signals.csv"
    
    # Save results
    results_df.to_csv(output_file, index=False)
    
    # Print summary
    total_minutes = len(results_df)
    buy_signals = results_df[results_df['buy'] == True]
    
    print(f"\n=== BUY SIGNAL ANALYSIS SUMMARY ===")
    print(f"Total minutes analyzed: {total_minutes}")
    print(f"Buy signals generated: {len(buy_signals)}")
    print(f"Buy signal rate: {len(buy_signals)/total_minutes*100:.1f}%")
    
    if len(buy_signals) > 0:
        print(f"\nR-level distribution:")
        r_dist = buy_signals['R'].value_counts().sort_index()
        for r_level, count in r_dist.items():
            print(f"  {r_level}R: {count} signals ({count/len(buy_signals)*100:.1f}%)")
        
        print(f"\nAverage time to target: {buy_signals['time_minutes'].mean():.1f} minutes")
        print(f"Median time to target: {buy_signals['time_minutes'].median():.1f} minutes")
    
    print(f"\nResults saved to: {output_file}")
    
    return results_df