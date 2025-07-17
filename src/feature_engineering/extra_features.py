import pandas as pd
import numpy as np
from typing import Dict
import logging
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema




logger = logging.getLogger(__name__)

def get_daily_open_price(daily_df: pd.DataFrame) -> float:
    """
    Get the official daily opening price from daily data.
    
    Args:
        daily_df: Daily OHLCV data DataFrame
        
    Returns:
        Official daily opening price
    """
    try:
        if daily_df.empty:
            logger.warning("No daily data available for opening price")
            return 0.0
            
        # Get the most recent daily open (today's open)
        daily_open = float(daily_df['open'].iloc[-1])
        return daily_open if daily_open > 0 else 0.0
    except Exception as e:
        logger.error(f"Error getting daily open price: {e}")
        return 0.0

def calculate_additional_features(real_time_df: pd.DataFrame, intra_day_df: pd.DataFrame, daily_df: pd.DataFrame) -> Dict:
    """
    Calculate additional ML features that can improve model performance.
    
    Args:
        real_time_df: Current minute's real-time data
        intra_day_df: Historical intraday data up to current minute
        daily_df: Daily historical data
        
    Returns:
        Dictionary with additional feature values
    """
    features = {}
    
    if intra_day_df.empty:
        # Return default values if no data
        return {
            'Volume_Percentile_Intraday': 0.5,
            'RSI_Regime': 'neutral',
            'RSI_Regime_Numeric': 0,
            'Momentum_Acceleration': 0,
            'Order_Book_Stability': 0.5,
            'Price_Action_Quality': 0.5,
            'Volume_Zscore_Intraday': 0,
            'RSI_Distance_From_50': 0,
            'Momentum_Consistency': 0.5,
            'Return_Since_Open_Enhanced': 0.0,
            'Intraday_Performance_Percentile': 0.5
        }
    
    # Get current values
    current_volume = real_time_df['volume'].iloc[0] if 'volume' in real_time_df.columns else 0
    current_price = real_time_df['lastPrice'].iloc[0] if 'lastPrice' in real_time_df.columns else 0
    current_rsi = None
    current_momentum = None
    current_imbalance = None


    # More comprehensive debug (add this right before current_volume calculation):
    print("=" * 50)
    print("🔍 REAL_TIME_DF FULL DEBUG:")
    print(f"Shape: {real_time_df.shape}")
    print(f"Columns: {list(real_time_df.columns)}")
    print(f"Index: {real_time_df.index.tolist()}")
    print("\nFull DataFrame:")
    print(real_time_df.to_string())
    print("\nColumn dtypes:")
    print(real_time_df.dtypes)
    if not real_time_df.empty:
        print("\nFirst row as dict:")
        print(real_time_df.iloc[0].to_dict())
    print("=" * 50)



    # Try to get current technical indicators from the last calculation
    if not intra_day_df.empty:
        # Look for volume column
        volume_col = 'volume_1min' if 'volume_1min' in intra_day_df.columns else 'volume'
        
        # 1. VOLUME_PERCENTILE_INTRADAY
        if volume_col in intra_day_df.columns:
            day_volumes = intra_day_df[volume_col].dropna()
            if len(day_volumes) > 0:
                # Calculate percentile of current volume vs all volumes today
                volumes_including_current = np.append(day_volumes.values, current_volume)

               
               
                #Genius formula to calculate percentile
                current_percentile = (volumes_including_current <= current_volume).mean()
                features['Volume_Percentile_Intraday'] = current_percentile

                greater = sum(volumes_including_current > current_volume)
                smaller = sum(volumes_including_current < current_volume)
     
                print(f"VOLUME PERCENTILE DEBUG > g:{greater} s:{smaller} cur_vol:{current_volume} calc_perc:{current_percentile}")

                # Bonus: Volume Z-score (how many standard deviations from mean)
                if len(day_volumes) > 2:
                    volume_mean = day_volumes.mean()
                    volume_std = day_volumes.std()
                    if volume_std > 0:
                        features['Volume_Zscore_Intraday'] = (current_volume - volume_mean) / volume_std
                    else:
                        features['Volume_Zscore_Intraday'] = 0
                else:
                    features['Volume_Zscore_Intraday'] = 0
            else:
                features['Volume_Percentile_Intraday'] = 0.5
                features['Volume_Zscore_Intraday'] = 0
        else:
            features['Volume_Percentile_Intraday'] = 0.5
            features['Volume_Zscore_Intraday'] = 0

    # 2. RSI_REGIME (requires RSI calculation)
    # You'll need to calculate RSI from price data
    price_col = 'close_1min' if 'close_1min' in intra_day_df.columns else 'lastPrice'
    if price_col in intra_day_df.columns and len(intra_day_df) >= 14:
        prices = intra_day_df[price_col].dropna()
        if len(prices) >= 14:
            current_rsi = calculate_rsi(prices, period=14)
            
            # RSI Regime Classification (Fixed thresholds)
            if current_rsi >= 70:
                features['RSI_Regime'] = 'overbought'
                features['RSI_Regime_Numeric'] = 1
            elif current_rsi <= 30:
                features['RSI_Regime'] = 'oversold'
                features['RSI_Regime_Numeric'] = -1
            elif current_rsi >= 60:
                features['RSI_Regime'] = 'bullish'
                features['RSI_Regime_Numeric'] = 0.5
            elif current_rsi <= 40:
                features['RSI_Regime'] = 'bearish'
                features['RSI_Regime_Numeric'] = -0.5
            else:
                features['RSI_Regime'] = 'neutral'
                features['RSI_Regime_Numeric'] = 0
            
            # Distance from neutral (50)
            features['RSI_Distance_From_50'] = abs(current_rsi - 50) / 50
        else:
            features['RSI_Regime'] = 'neutral'
            features['RSI_Regime_Numeric'] = 0
            features['RSI_Distance_From_50'] = 0
    else:
        features['RSI_Regime'] = 'neutral'
        features['RSI_Regime_Numeric'] = 0
        features['RSI_Distance_From_50'] = 0
    
    # 3. MOMENTUM_ACCELERATION
    # Look at momentum change over recent periods
    if price_col in intra_day_df.columns and len(intra_day_df) >= 6:
        prices = intra_day_df[price_col].tail(6).values  # Last 6 minutes
        if len(prices) >= 6:
            # Calculate momentum for last 3 periods vs previous 3 periods
            recent_momentum = (prices[-1] - prices[-3]) / prices[-3] if prices[-3] != 0 else 0
            earlier_momentum = (prices[-3] - prices[-6]) / prices[-6] if prices[-6] != 0 else 0
            
            # Acceleration = change in momentum
            features['Momentum_Acceleration'] = recent_momentum - earlier_momentum
            
            # Momentum consistency (how consistent is the direction)
            returns = np.diff(prices) / prices[:-1]
            returns = returns[returns != 0]  # Remove zero returns
            if len(returns) > 0:
                # Measure consistency as % of returns in same direction as overall trend
                overall_trend = 1 if prices[-1] > prices[0] else -1
                same_direction = np.sum(np.sign(returns) == overall_trend)
                features['Momentum_Consistency'] = same_direction / len(returns)
            else:
                features['Momentum_Consistency'] = 0.5
        else:
            features['Momentum_Acceleration'] = 0
            features['Momentum_Consistency'] = 0.5
    else:
        features['Momentum_Acceleration'] = 0
        features['Momentum_Consistency'] = 0.5
    
    # 4. ORDER_BOOK_STABILITY
    # Measure how stable the order book imbalance has been
    if 'bid_sz_00' in intra_day_df.columns and 'ask_sz_00' in intra_day_df.columns:
        recent_data = intra_day_df.tail(10)  # Last 10 minutes
        if len(recent_data) >= 3:
            bid_sizes = recent_data['bid_sz_00'].values
            ask_sizes = recent_data['ask_sz_00'].values
            
            # Calculate imbalances for recent periods
            imbalances = []
            for i in range(len(bid_sizes)):
                if bid_sizes[i] + ask_sizes[i] > 0:
                    imbalance = (bid_sizes[i] - ask_sizes[i]) / (bid_sizes[i] + ask_sizes[i])
                    imbalances.append(imbalance)
            
            if len(imbalances) > 1:
                # Stability = 1 - (coefficient of variation)
                imbalance_std = np.std(imbalances)
                imbalance_mean = np.mean(np.abs(imbalances))
                if imbalance_mean > 0.01:  # Only calculate if meaningful imbalances
                    cv = imbalance_std / imbalance_mean
                    features['Order_Book_Stability'] = max(0, min(1, 1 - cv))  # Clamp between 0-1
                else:
                    features['Order_Book_Stability'] = 0.5  # Neutral if no meaningful imbalances
            else:
                features['Order_Book_Stability'] = 0.5
        else:
            features['Order_Book_Stability'] = 0.5
    else:
        features['Order_Book_Stability'] = 0.5
    
    # 5. PRICE_ACTION_QUALITY
    # Measure how "clean" the price action is (trend vs chop)
    if price_col in intra_day_df.columns and len(intra_day_df) >= 10:
        recent_prices = intra_day_df[price_col].tail(10).values
        if len(recent_prices) >= 10:
            # Calculate price action quality metrics
            
            # A) Trend strength: R-squared of linear regression
            x = np.arange(len(recent_prices))
            if len(recent_prices) > 1:
                correlation = np.corrcoef(x, recent_prices)[0, 1]
                r_squared = correlation ** 2 if not np.isnan(correlation) else 0
            else:
                r_squared = 0
            
            # B) Noise ratio: ratio of total movement to net movement
            total_movement = np.sum(np.abs(np.diff(recent_prices)))
            net_movement = abs(recent_prices[-1] - recent_prices[0])
            if total_movement > 0:
                noise_ratio = net_movement / total_movement
            else:
                noise_ratio = 0
            
            # C) Combine metrics (higher = cleaner trend)
            features['Price_Action_Quality'] = (r_squared + noise_ratio) / 2
        else:
            features['Price_Action_Quality'] = 0.5
    else:
        features['Price_Action_Quality'] = 0.5

    # 6. FIXED: Return Since Open Enhanced (using correct daily opening price)
    if current_price > 0 and not daily_df.empty:
        try:
            # CRITICAL FIX: Use official daily opening price, NOT minute-level opens
            official_open_price = get_daily_open_price(daily_df)
            
            if official_open_price > 0:
                return_since_open = ((current_price - official_open_price) / official_open_price) * 100
                features['Return_Since_Open_Enhanced'] = np.clip(return_since_open, -20.0, 20.0)
                
                # Intraday performance percentile vs historical same-time-of-day performance
                if 'timestamp' in real_time_df.columns or 'ts_event_clean' in intra_day_df.columns:
                    # Could be enhanced to compare vs historical performance at same time of day
                    # For now, use simple percentile vs today's range
                    if len(intra_day_df) > 10:
                        all_prices = intra_day_df[price_col].values
                        day_high = np.max(all_prices)
                        day_low = np.min(all_prices)
                        
                        if day_high > day_low:
                            price_position = (current_price - day_low) / (day_high - day_low)
                            features['Intraday_Performance_Percentile'] = np.clip(price_position, 0.0, 1.0)
                        else:
                            features['Intraday_Performance_Percentile'] = 0.5
                    else:
                        features['Intraday_Performance_Percentile'] = 0.5
                else:
                    features['Intraday_Performance_Percentile'] = 0.5
            else:
                logger.warning("Invalid official opening price for Return_Since_Open_Enhanced")
                features['Return_Since_Open_Enhanced'] = 0.0
                features['Intraday_Performance_Percentile'] = 0.5
        except Exception as e:
            logger.error(f"Error calculating Return_Since_Open_Enhanced: {e}")
            features['Return_Since_Open_Enhanced'] = 0.0
            features['Intraday_Performance_Percentile'] = 0.5
    else:
        features['Return_Since_Open_Enhanced'] = 0.0
        features['Intraday_Performance_Percentile'] = 0.5
    
    return features

def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    """
    Calculate RSI (Relative Strength Index).
    
    Args:
        prices: Series of prices
        period: RSI period (default 14)
        
    Returns:
        Current RSI value
    """
    if len(prices) < period + 1:
        return 50.0  # Neutral RSI if not enough data
    
    # Calculate price changes
    delta = prices.diff()
    
    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    
    # Calculate average gains and losses using Wilder's smoothing
    avg_gains = gains.ewm(alpha=1/period, adjust=False).mean()
    avg_losses = losses.ewm(alpha=1/period, adjust=False).mean()
    
    # Calculate RS and RSI
    current_avg_gain = avg_gains.iloc[-1]
    current_avg_loss = avg_losses.iloc[-1]
    
    if current_avg_loss == 0:
        return 100.0
        
    rs = current_avg_gain / current_avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return float(np.clip(rsi, 0.0, 100.0))

def add_regime_features(features: Dict) -> Dict:
    """
    Add additional regime-based features based on existing features.
    
    Args:
        features: Existing feature dictionary
        
    Returns:
        Updated feature dictionary with regime features
    """
    # Market regime based on multiple indicators
    rsi = features.get('RSI_14', 50)
    volume_percentile = features.get('Volume_Percentile_Intraday', 0.5)
    momentum_acceleration = features.get('Momentum_Acceleration', 0)
    return_since_open = features.get('Return_Since_Open_Enhanced', 0)  # Use the fixed version
    
    # Multi-factor regime
    regime_score = 0
    
    # RSI component
    if rsi > 70:
        regime_score += 1  # Overbought
    elif rsi < 30:
        regime_score -= 1  # Oversold
    
    # Volume component
    if volume_percentile > 0.8:
        regime_score += 0.5  # High volume
    elif volume_percentile < 0.2:
        regime_score -= 0.5  # Low volume
    
    # Momentum component
    if momentum_acceleration > 0.001:
        regime_score += 0.5  # Accelerating
    elif momentum_acceleration < -0.001:
        regime_score -= 0.5  # Decelerating
    
    # Return since open component (NEW: using corrected opening price)
    if return_since_open > 2.0:  # Strong positive performance
        regime_score += 0.5
    elif return_since_open < -2.0:  # Strong negative performance
        regime_score -= 0.5
    
    # Market regime classification
    if regime_score >= 1.5:
        features['Market_Regime'] = 'strong_bullish'
        features['Market_Regime_Numeric'] = 2
    elif regime_score >= 0.5:
        features['Market_Regime'] = 'bullish'
        features['Market_Regime_Numeric'] = 1
    elif regime_score <= -1.5:
        features['Market_Regime'] = 'strong_bearish'
        features['Market_Regime_Numeric'] = -2
    elif regime_score <= -0.5:
        features['Market_Regime'] = 'bearish'
        features['Market_Regime_Numeric'] = -1
    else:
        features['Market_Regime'] = 'neutral'
        features['Market_Regime_Numeric'] = 0
    
    return features

def validate_additional_features(features: Dict) -> Dict[str, str]:
    """
    Validate the additional features for consistency and logical bounds.
    
    Args:
        features: Feature dictionary to validate
        
    Returns:
        Dictionary of validation warnings
    """
    warnings = {}
    
    try:
        # 1. Validate percentile features are in [0,1] range
        percentile_features = ['Volume_Percentile_Intraday', 'Intraday_Performance_Percentile']
        for feature in percentile_features:
            if feature in features:
                value = features[feature]
                if not (0 <= value <= 1):
                    warnings[f'{feature}_Range'] = f"{feature} value {value:.3f} outside [0,1] range"
        
        # 2. Validate RSI distance is in [0,1] range
        if 'RSI_Distance_From_50' in features:
            rsi_distance = features['RSI_Distance_From_50']
            if not (0 <= rsi_distance <= 1):
                warnings['RSI_Distance_Range'] = f"RSI distance {rsi_distance:.3f} outside [0,1] range"
        
        # 3. Validate momentum consistency is in [0,1] range
        if 'Momentum_Consistency' in features:
            momentum_consistency = features['Momentum_Consistency']
            if not (0 <= momentum_consistency <= 1):
                warnings['Momentum_Consistency_Range'] = f"Momentum consistency {momentum_consistency:.3f} outside [0,1] range"
        
        # 4. Validate RSI regime consistency
        if 'RSI_Regime_Numeric' in features and 'RSI_Distance_From_50' in features:
            regime_numeric = features['RSI_Regime_Numeric']
            rsi_distance = features['RSI_Distance_From_50']
            
            # Strong regimes should have high distance from 50
            if abs(regime_numeric) == 1 and rsi_distance < 0.4:  # overbought/oversold
                warnings['RSI_Regime_Consistency'] = f"Strong RSI regime ({regime_numeric}) but low distance from 50 ({rsi_distance:.3f})"
        
        # 5. Validate market regime logic
        if 'Market_Regime_Numeric' in features and 'Return_Since_Open_Enhanced' in features:
            market_regime = features['Market_Regime_Numeric']
            return_enhanced = features['Return_Since_Open_Enhanced']
            
            # Strong bullish regime should generally have positive returns
            if market_regime >= 1 and return_enhanced < -1.0:
                warnings['Market_Regime_Return_Mismatch'] = f"Bullish regime ({market_regime}) with negative return ({return_enhanced:.2f}%)"
            
            # Strong bearish regime should generally have negative returns
            elif market_regime <= -1 and return_enhanced > 1.0:
                warnings['Market_Regime_Return_Mismatch'] = f"Bearish regime ({market_regime}) with positive return ({return_enhanced:.2f}%)"
        
        # 6. Validate order book stability is reasonable
        if 'Order_Book_Stability' in features:
            ob_stability = features['Order_Book_Stability']
            if not (0 <= ob_stability <= 1):
                warnings['Order_Book_Stability_Range'] = f"Order book stability {ob_stability:.3f} outside [0,1] range"
        
        # 7. Validate price action quality
        if 'Price_Action_Quality' in features:
            pa_quality = features['Price_Action_Quality']
            if not (0 <= pa_quality <= 1):
                warnings['Price_Action_Quality_Range'] = f"Price action quality {pa_quality:.3f} outside [0,1] range"
        
    except Exception as e:
        warnings['Additional_Features_Validation_Error'] = f"Validation failed: {e}"
    
    return warnings

# Example usage function to integrate with your existing code
def enhanced_calculate_features(real_time_df: pd.DataFrame, intra_day_df: pd.DataFrame, daily_df: pd.DataFrame, validate: bool = True) -> Dict:
    """
    Enhanced feature calculation that includes both original and additional features.
    
    This function would replace or supplement your existing calculate_features function.
    
    Args:
        real_time_df: Real-time market data
        intra_day_df: Intraday minute data
        daily_df: Daily OHLCV data
        validate: Whether to run validation
        
    Returns:
        Complete feature dictionary
    """
    # Calculate your existing features first
    # features = calculate_features(real_time_df, intra_day_df, daily_df)  # Your existing function
    features = {}  # Placeholder - replace with your actual function call
    
    # Add the new features
    additional_features = calculate_additional_features(real_time_df, intra_day_df, daily_df)
    features.update(additional_features)
        
    # Add regime-based features
    features = add_regime_features(features)
    
    # Run validation if requested
    if validate:
        warnings = validate_additional_features(features)
        if warnings:
            logger.warning("Additional features validation warnings:")
            for feature, warning in warnings.items():
                logger.warning(f"  {feature}: {warning}")
    
    return features

# Integration example with your main feature calculator
def integrate_with_main_calculator(real_time_df: pd.DataFrame, intra_day_df: pd.DataFrame, daily_df: pd.DataFrame) -> Dict:
    """
    Example of how to integrate additional features with your main FeatureCalculator.
    
    Args:
        real_time_df: Real-time market data
        intra_day_df: Intraday minute data  
        daily_df: Daily OHLCV data
        
    Returns:
        Complete feature set including additional features
    """
    # Import your main feature calculator
    # from your_feature_library import calculate_features
    
    # Calculate main features (uncomment when you have the import)
    # main_features = calculate_features(real_time_df, intra_day_df, daily_df, validate=True)
    main_features = {}  # Placeholder
    
    # Calculate additional features
    additional_features = calculate_additional_features(real_time_df, intra_day_df, daily_df)
    
    # Combine all features
    all_features = {**main_features, **additional_features}
    
    # Add regime features based on combined feature set
    all_features = add_regime_features(all_features)
    
    # Validate everything
    main_warnings = {}  # Would come from main calculator validation
    additional_warnings = validate_additional_features(all_features)
    
    if additional_warnings:
        logger.warning("Feature validation warnings found:")
        for feature, warning in additional_warnings.items():
            logger.warning(f"  {feature}: {warning}")
    
    return all_features