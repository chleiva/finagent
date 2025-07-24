import pandas as pd
import numpy as np
from scipy.signal import argrelextrema


def strict_higher_swing_lows(df, price_col='close', window=4, max_gap_minutes=40, 
                            vwap_col='vwap', vwap_tolerance=0.5, debug=False):
    """
    Detect strict higher swing lows in intraday market data.
    
    A strict higher swing low is identified when:
    1. Current low > Previous low (higher low condition)
    2. Last high > Previous high (higher high condition)  
    3. Both conditions must be satisfied simultaneously
    4. Optionally filters by proximity to VWAP
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with datetime index containing intraday market data
    price_col : str, default 'close'
        Column name containing price data for swing detection
    window : int, default 4
        Window size for local minima/maxima detection
    max_gap_minutes : int, default 40
        Maximum time gap (in minutes) between consecutive swing lows to include
    vwap_col : str, default 'vwap'
        Column name containing VWAP values (optional filter)
    vwap_tolerance : float, default 0.5
        Maximum distance from VWAP to consider valid swing low
    debug : bool, default False
        Print detailed debugging information
        
    Returns:
    --------
    pandas.DataFrame
        Filtered DataFrame containing only the strict higher swing low points
    """
    
    def _detect_strict_swing_lows(prices, window=7, debug=False):
        """Core logic for detecting strict higher swing lows"""
        lows = argrelextrema(prices.values, np.less, order=window)[0]
        highs = argrelextrema(prices.values, np.greater, order=window)[0]
        points = [(i, 'low') for i in lows] + [(i, 'high') for i in highs]
        points = sorted(points, key=lambda x: x[0])
        
        valid_lows = []
        prev_low_idx = None
        prev_high_idx = None
        last_high_idx = None
        
        for idx, kind in points:
            if kind == 'high':
                prev_high_idx, last_high_idx = last_high_idx, idx
                if debug and last_high_idx is not None:
                    print(f"High at {last_high_idx}: price = {prices.iloc[last_high_idx]:.2f}")
                    
            elif kind == 'low':
                current_low_price = prices.iloc[idx]
                if debug:
                    print(f"\nEvaluating Low at {idx}: price = {current_low_price:.2f}")
                
                # Need previous low and two previous highs for comparison
                if prev_low_idx is not None and prev_high_idx is not None and last_high_idx is not None:
                    prev_low_price = prices.iloc[prev_low_idx]
                    prev_high_price = prices.iloc[prev_high_idx] 
                    last_high_price = prices.iloc[last_high_idx]
                    
                    # Check BOTH conditions:
                    # 1. Current low > Previous low (higher low)
                    # 2. Last high > Previous high (higher high)
                    condition1 = current_low_price > prev_low_price
                    condition2 = last_high_price > prev_high_price
                    
                    if debug:
                        print(f"  Previous low at {prev_low_idx}: {prev_low_price:.2f}")
                        print(f"  Previous high at {prev_high_idx}: {prev_high_price:.2f}")
                        print(f"  Last high at {last_high_idx}: {last_high_price:.2f}")
                        print(f"  Condition 1 (current low > prev low): {current_low_price:.2f} > {prev_low_price:.2f} = {condition1}")
                        print(f"  Condition 2 (last high > prev high): {last_high_price:.2f} > {prev_high_price:.2f} = {condition2}")
                    
                    if condition1 and condition2:
                        valid_lows.append(idx)
                        if debug:
                            print(f"  -> MARKED as valid higher swing low")
                    else:
                        if debug:
                            print(f"  -> NOT MARKED (failed condition)")
                else:
                    if debug:
                        print(f"  -> NOT MARKED (insufficient previous points)")
                
                # Update previous low for next iteration
                prev_low_idx = idx
        
        return valid_lows
    
    # Ensure we have a proper DataFrame with datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if debug:
            print("Warning: DataFrame index is not DatetimeIndex, attempting conversion")
        try:
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            if debug:
                print(f"Error converting index to datetime: {e}")
            return pd.DataFrame()
    
    # Detect initial swing lows
    strict_lows_idx = _detect_strict_swing_lows(df[price_col], window=window, debug=debug)
    
    if not strict_lows_idx:
        if debug:
            print("No strict higher swing lows detected")
        return pd.DataFrame()
    
    # Convert indices to timestamps - FIX: Handle the index properly
    try:
        strict_lows_times = df.index[strict_lows_idx]
    except IndexError as e:
        if debug:
            print(f"Error accessing swing low timestamps: {e}")
        return pd.DataFrame()
    
    # Apply time gap filter (consecutive swing lows within max_gap_minutes)
    max_gap = pd.Timedelta(minutes=max_gap_minutes)
    final_strict_lows_idx = []
    
    if len(strict_lows_idx) > 0:
        final_strict_lows_idx = [strict_lows_idx[0]]
        last_time = strict_lows_times[0]
        
        for idx, ts in zip(strict_lows_idx[1:], strict_lows_times[1:]):
            try:
                # FIX: Ensure both timestamps are comparable
                if isinstance(ts, pd.Timestamp) and isinstance(last_time, pd.Timestamp):
                    time_diff = ts - last_time
                    if time_diff <= max_gap:
                        final_strict_lows_idx.append(idx)
                else:
                    if debug:
                        print(f"Warning: Timestamp comparison issue - ts: {type(ts)}, last_time: {type(last_time)}")
                last_time = ts  # Always update last_time
            except Exception as e:
                if debug:
                    print(f"Error in time gap calculation: {e}")
                continue
    
    # Create boolean mask for final swing lows
    df_copy = df.copy()
    df_copy["is_strict_swing_low"] = False
    if final_strict_lows_idx:
        try:
            df_copy.iloc[final_strict_lows_idx, df_copy.columns.get_loc("is_strict_swing_low")] = True
        except Exception as e:
            if debug:
                print(f"Error setting swing low flags: {e}")
            return pd.DataFrame()
    
    # Filter by VWAP proximity if VWAP column exists
    if vwap_col and vwap_col in df_copy.columns and vwap_tolerance is not None:
        try:
            vwap_filter = np.abs(df_copy[price_col] - df_copy[vwap_col]) < vwap_tolerance
            strict_swing_lows = df_copy[df_copy["is_strict_swing_low"] & vwap_filter]
        except Exception as e:
            if debug:
                print(f"Error applying VWAP filter: {e}")
            strict_swing_lows = df_copy[df_copy["is_strict_swing_low"]]
    else:
        strict_swing_lows = df_copy[df_copy["is_strict_swing_low"]]
    
    if debug:
        print(f"\nFinal Results:")
        print(f"Initial swing lows detected: {len(strict_lows_idx)}")
        print(f"After time gap filter: {len(final_strict_lows_idx)}")
        print(f"After VWAP filter: {len(strict_swing_lows)}")
    
    return strict_swing_lows


def adaptive_higher_swing_lows(df, price_col='close', window=4, max_gap_minutes=40, 
                              vwap_col='vwap', vwap_tolerance=0.5, debug=False):
    """
    Detect adaptive higher swing lows in intraday market data using volatility-adjusted windows.
    
    An adaptive higher swing low is identified when:
    1. Current low > Previous low (first condition)
    2. Current low > Second previous low (second condition)
    3. Uses adaptive window based on recent vs long-term volatility
    4. Optionally filters by proximity to VWAP
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with datetime index containing intraday market data
    price_col : str, default 'close'
        Column name containing price data for swing detection
    window : int, default 4
        Base window size for local minima detection (will be adapted based on volatility)
    max_gap_minutes : int, default 40
        Maximum time gap (in minutes) between consecutive swing lows to include
    vwap_col : str, default 'vwap'
        Column name containing VWAP values (optional filter)
    vwap_tolerance : float, default 0.5
        Maximum distance from VWAP to consider valid swing low
    debug : bool, default False
        Print detailed debugging information
        
    Returns:
    --------
    pandas.DataFrame
        Filtered DataFrame containing only the adaptive higher swing low points
    """
    
    def _calculate_adaptive_window(prices, base_window=4, debug=False):
        """Calculate adaptive window based on recent volatility"""
        try:
            # Calculate recent volatility (20-period ATR)
            high_vals = prices.rolling(window=20).max()
            low_vals = prices.rolling(window=20).min()
            recent_atr = (high_vals - low_vals).rolling(window=20).mean()
            
            # Calculate longer-term volatility (100-period)
            longer_atr = (high_vals - low_vals).rolling(window=100).mean()
            
            # Get most recent values
            current_recent_atr = recent_atr.iloc[-1] if not recent_atr.empty and not pd.isna(recent_atr.iloc[-1]) else 0
            current_longer_atr = longer_atr.iloc[-1] if not longer_atr.empty and not pd.isna(longer_atr.iloc[-1]) else 1
            
            # Avoid division by zero
            volatility_ratio = current_recent_atr / current_longer_atr if current_longer_atr != 0 else 1.0
            
            # Adjust window based on volatility
            if volatility_ratio > 1.5:
                # High volatility → larger window
                adaptive_window = min(base_window + 2, 8)
            elif volatility_ratio < 0.7:
                # Low volatility → smaller window
                adaptive_window = max(base_window - 1, 2)
            else:
                # Normal volatility → base window
                adaptive_window = base_window
                
            if debug:
                print(f"Volatility ratio: {volatility_ratio:.2f}")
                print(f"Adaptive window: {adaptive_window} (base: {base_window})")
                
            return adaptive_window
        except Exception as e:
            if debug:
                print(f"Error calculating adaptive window: {e}")
            return base_window
    
    def _detect_adaptive_swing_lows(prices, base_window=4, debug=False):
        """Core logic for detecting adaptive higher swing lows"""
        try:
            # Calculate adaptive window
            adaptive_window = _calculate_adaptive_window(prices, base_window, debug)
            
            # Find swing lows using adaptive window
            lows = argrelextrema(prices.values, np.less, order=adaptive_window)[0]
            
            valid_lows = []
            
            for i, current_idx in enumerate(lows):
                current_low_price = prices.iloc[current_idx]
                
                if debug:
                    print(f"\nEvaluating Adaptive Low at {current_idx}: price = {current_low_price:.2f}")
                
                # Need at least 2 previous lows to compare
                if i >= 2:
                    prev_low_1_idx = lows[i-1]  # Most recent previous low
                    prev_low_2_idx = lows[i-2]  # Second most recent previous low
                    
                    prev_low_1_price = prices.iloc[prev_low_1_idx]
                    prev_low_2_price = prices.iloc[prev_low_2_idx]
                    
                    # Check if current low is higher than BOTH previous lows
                    condition1 = current_low_price > prev_low_1_price
                    condition2 = current_low_price > prev_low_2_price
                    
                    if debug:
                        print(f"  Previous low 1 at {prev_low_1_idx}: {prev_low_1_price:.2f}")
                        print(f"  Previous low 2 at {prev_low_2_idx}: {prev_low_2_price:.2f}")
                        print(f"  Condition 1 (current > prev_1): {current_low_price:.2f} > {prev_low_1_price:.2f} = {condition1}")
                        print(f"  Condition 2 (current > prev_2): {current_low_price:.2f} > {prev_low_2_price:.2f} = {condition2}")
                    
                    if condition1 and condition2:
                        valid_lows.append(current_idx)
                        if debug:
                            print(f"  -> MARKED as adaptive higher swing low (window={adaptive_window})")
                    else:
                        if debug:
                            print(f"  -> NOT MARKED (current low not higher than both previous lows)")
                else:
                    if debug:
                        print(f"  -> NOT MARKED (need at least 2 previous lows)")
            
            return valid_lows
        except Exception as e:
            if debug:
                print(f"Error in adaptive swing low detection: {e}")
            return []
    
    # Ensure we have a proper DataFrame with datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if debug:
            print("Warning: DataFrame index is not DatetimeIndex, attempting conversion")
        try:
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            if debug:
                print(f"Error converting index to datetime: {e}")
            return pd.DataFrame()
    
    # Detect initial swing lows
    adaptive_lows_idx = _detect_adaptive_swing_lows(df[price_col], base_window=window, debug=debug)
    
    if not adaptive_lows_idx:
        if debug:
            print("No adaptive higher swing lows detected")
        return pd.DataFrame()
    
    # Convert indices to timestamps - FIX: Handle the index properly
    try:
        adaptive_lows_times = df.index[adaptive_lows_idx]
    except IndexError as e:
        if debug:
            print(f"Error accessing swing low timestamps: {e}")
        return pd.DataFrame()
    
    # Apply time gap filter (consecutive swing lows within max_gap_minutes)
    max_gap = pd.Timedelta(minutes=max_gap_minutes)
    final_adaptive_lows_idx = []
    
    if len(adaptive_lows_idx) > 0:
        final_adaptive_lows_idx = [adaptive_lows_idx[0]]
        last_time = adaptive_lows_times[0]
        
        for idx, ts in zip(adaptive_lows_idx[1:], adaptive_lows_times[1:]):
            try:
                # FIX: Ensure both timestamps are comparable
                if isinstance(ts, pd.Timestamp) and isinstance(last_time, pd.Timestamp):
                    time_diff = ts - last_time
                    if time_diff <= max_gap:
                        final_adaptive_lows_idx.append(idx)
                else:
                    if debug:
                        print(f"Warning: Timestamp comparison issue - ts: {type(ts)}, last_time: {type(last_time)}")
                last_time = ts  # Always update last_time
            except Exception as e:
                if debug:
                    print(f"Error in time gap calculation: {e}")
                continue
    
    # Create boolean mask for final swing lows
    df_copy = df.copy()
    df_copy["is_adaptive_swing_low"] = False
    if final_adaptive_lows_idx:
        try:
            df_copy.iloc[final_adaptive_lows_idx, df_copy.columns.get_loc("is_adaptive_swing_low")] = True
        except Exception as e:
            if debug:
                print(f"Error setting swing low flags: {e}")
            return pd.DataFrame()
    
    # Filter by VWAP proximity if VWAP column exists
    if vwap_col and vwap_col in df_copy.columns and vwap_tolerance is not None:
        try:
            vwap_filter = np.abs(df_copy[price_col] - df_copy[vwap_col]) < vwap_tolerance
            adaptive_swing_lows = df_copy[df_copy["is_adaptive_swing_low"] & vwap_filter]
        except Exception as e:
            if debug:
                print(f"Error applying VWAP filter: {e}")
            adaptive_swing_lows = df_copy[df_copy["is_adaptive_swing_low"]]
    else:
        adaptive_swing_lows = df_copy[df_copy["is_adaptive_swing_low"]]
    
    if debug:
        print(f"\nFinal Results:")
        print(f"Initial adaptive swing lows detected: {len(adaptive_lows_idx)}")
        print(f"After time gap filter: {len(final_adaptive_lows_idx)}")
        print(f"After VWAP filter: {len(adaptive_swing_lows)}")
    
    return adaptive_swing_lows