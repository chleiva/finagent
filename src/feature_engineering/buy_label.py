import pandas as pd
import numpy as np
from datetime import datetime, time
from typing import Dict, Optional, Tuple

# Pre-define constants to avoid recreating them
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
MIN_TIME_AFTER_OPEN = time(10, 0)
MAX_TIME_BEFORE_CLOSE = time(15, 0)
CLOSE_CUTOFF = time(15, 30)
REQUIRED_FIELDS = frozenset(['Current_Price', '1R_Stop_Loss', '1R_Target', '2R_Target', 'ATR_Distance'])

def meets_basic_buy_conditions(features: Dict, current_time: pd.Timestamp) -> Tuple[bool, str]:
    """
    Check if basic conditions are met for considering a buy signal.
    
    Args:
        features: Dictionary containing calculated features
        current_time: Current timestamp
        
    Returns:
        Tuple of (condition_met: bool, reason: str)
    """
    #current_time_only = current_time.time()
    
    # Check if it's during market hours
    #if current_time_only < MARKET_OPEN or current_time_only > MARKET_CLOSE:
    #    return False, "outside_market_hours"
    
    # Must be at least 30 minutes after market open
    #if current_time_only < MIN_TIME_AFTER_OPEN:
    #    return False, "too_early_after_open"
    
    # Must be at least 1 hour before market close
    #if current_time_only > MAX_TIME_BEFORE_CLOSE:
    #    return False, "too_close_to_close"
    
    # Check if we have valid R-levels
    for field in REQUIRED_FIELDS:
        value = features.get(field)
        if value is None or pd.isna(value) or value <= 0:
            return False, f"missing_or_invalid_{field.lower()}"
    
    # Basic technical conditions
    current_price = features['Current_Price']
    stop_loss = features['1R_Stop_Loss']
    target_1r = features['1R_Target']
    
    # Ensure stop loss is below current price and target is above
    #if stop_loss >= current_price:
    #    return False, "stop_loss_above_price"
    #if target_1r <= current_price:
    #    return False, "target_below_price"

    # Volume should be at least 40th percentile of the day
    #if features.get('Volume_Percentile_Intraday', 0) < 0.4:
    #    return False, "volume_percentile_too_low"
    
    # Check for reasonable ATR distance (not too tight or too wide)
    #atr_pct = (features['ATR_Distance'] / current_price) * 100
    
    # ATR should be between 0.01% and 3% of current price
    #if atr_pct < 0.01:
    #    return False, "atr_too_tight"
    #if atr_pct > 3.0:
    #    return False, "atr_too_wide"
    
    # Optional: Add more technical conditions
    # Example: Check if we have volume data and it's reasonable
    #volume = features.get('volume', 0)
    #if volume > 0 and volume < 100:  # Minimum volume threshold
    #    return False, "volume_too_low"
    
    # Optional: Check spread conditions
    bid = features.get('bidPrice', 0)
    ask = features.get('askPrice', 0)
    if bid > 0 and ask > 0:
        spread_pct = ((ask - bid) / current_price) * 100
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
    #if current_time.time() > CLOSE_CUTOFF:
    #    result['reason'] = 'market_near_close'
     #   return result
    
    # Get R-levels from features
    current_price = features['Current_Price']
    stop_loss = features['1R_Stop_Loss']
    target_1r = features['1R_Target']
    target_2r = features['2R_Target']
    r_increment = target_2r - target_1r
    target_3r = features.get('3R_Target', target_2r + r_increment)
    target_4r = features.get('4R_Target', target_3r + r_increment)
    
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
    future_data = data_for_future[data_for_future[time_col].dt.floor('min') > current_time]
    
    if future_data.empty:
        result['reason'] = 'no_future_data'
        return result
    
    # Sort by timestamp to ensure chronological order
    if not future_data[time_col].is_monotonic_increasing:
        future_data = future_data.sort_values(time_col)
    
    # Group by minute and get the close price for each minute
    future_data['minute'] = future_data[time_col].dt.floor('min')
    future_prices = future_data.groupby('minute')[price_col].last()
    

    # Apply time window limit if specified
    if max_time_window_minutes > 0:
        max_time = current_time + pd.Timedelta(minutes=max_time_window_minutes)
        future_prices = future_prices[future_prices.index <= max_time]
        
        if future_prices.empty:
            result['reason'] = 'no_data_in_time_window'
            return result
    
    # Track what happens minute by minute
    max_r_achieved = 0
    time_to_target = 0
    
    # Convert to numpy arrays for faster iteration
    timestamps = future_prices.index.to_numpy()
    prices = future_prices.values

    max_time_window_minutes = 40
    
    for i in range(len(prices)):
        price = prices[i]
        minutes_elapsed = (timestamps[i] - current_time).total_seconds() / 60
        
        # Check if we've exceeded the maximum time window
        if max_time_window_minutes > 0 and minutes_elapsed > max_time_window_minutes:
            break
        
        # Check if we hit stop loss (-1R), if already reached max_r_achieved ignore
        if max_r_achieved == 0 and price <= stop_loss:
            result['reason'] = f'reached_negative_1r'
            return result
        
        # Check R-levels achieved
        if price >= target_4r and max_r_achieved < 4:
            max_r_achieved = 4
            time_to_target = minutes_elapsed
            break
        elif price >= target_3r and max_r_achieved < 3:
            max_r_achieved = 3
            time_to_target = minutes_elapsed
        elif price >= target_2r and max_r_achieved < 2:
            max_r_achieved = 2
            time_to_target = minutes_elapsed
        
    
        # Determine final result
        if max_r_achieved >= 3:
            # Successfully reached at least 2R without hitting stop
            result['buy'] = True
            result['R'] = max_r_achieved
            result['time_minutes'] = int(time_to_target)
            result['reason'] = f'reached_{max_r_achieved}r_success'
            return result


    # Determine final result
    if max_r_achieved >= 2:
        # Successfully reached at least 2R without hitting stop
        result['buy'] = True
        result['R'] = max_r_achieved
        result['time_minutes'] = int(time_to_target)
        result['reason'] = f'reached_{max_r_achieved}r_success'
        return result

    
    result['reason'] = 'timeout_no_2r'
    
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
    
    # Pre-create timestamp column if not exists
    if 'timestamp' not in features_df.columns:
        features_df['timestamp'] = pd.to_datetime(features_df.index)
    
    # Convert to records for faster iteration
    records = features_df.to_dict('records')
    
    for idx, features in enumerate(records):
        # Create a mock real_time_df for this minute
        timestamp = pd.to_datetime(features['timestamp'])
        real_time_df = pd.DataFrame({
            'timestamp': [timestamp],
            'bidPrice': [features.get('bidPrice', 0)],
            'askPrice': [features.get('askPrice', 0)],
            'lastPrice': [features.get('lastPrice', features.get('Current_Price', 0))],
            'volume': [features.get('volume', 0)]
        })
        
        # Evaluate buy signal
        signal_result = evaluate_buy_signal(features, real_time_df, intra_day_df, daily_df)
        
        # Combine original features with signal results
        features.update(signal_result)
        results.append(features)
        
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
    
    # Evaluate buy signals - Note: This should be evaluate_all_buy_signals_batch
    results_df = evaluate_all_buy_signals_batch(features_df, price_data, pd.DataFrame())
    
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