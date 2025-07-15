import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats.mstats import winsorize
from scipy.signal import argrelextrema
from sklearn.preprocessing import PowerTransformer, RobustScaler
from sklearn.feature_selection import mutual_info_classif, f_classif
import warnings
warnings.filterwarnings('ignore')

def calculate_peak_valley_vector_chris_combined(intra_day_df: pd.DataFrame) -> dict:
    """
    Calculate a 2D directional vector (impulse) from recent peaks and valleys 
    using a combined pseudo-price (HL2) over the last 30 minutes.

    Returns:
        dict: Magnitude, Direction, Duration, and Slope of the impulse vector.
    """
    if not {'high_1min', 'low_1min'}.issubset(intra_day_df.columns) or len(intra_day_df) < 30:
        return {
            "Impulse_Vector_Magnitude_chris": 0.0,
            "Impulse_Vector_Direction_chris": 0,
            "Impulse_Vector_Duration_chris": 0,
            "Impulse_Vector_Slope_chris": 0.0,
        }

    recent_df = intra_day_df.tail(30).reset_index(drop=True)
    pseudo_prices = (recent_df['high_1min'] + recent_df['low_1min']) / 2

    # Find local maxima and minima
    peaks_idx = argrelextrema(pseudo_prices.values, np.greater, order=2)[0]
    valleys_idx = argrelextrema(pseudo_prices.values, np.less, order=2)[0]

    if len(peaks_idx) == 0 or len(valleys_idx) == 0:
        return {
            "Impulse_Vector_Magnitude_chris": 0.0,
            "Impulse_Vector_Direction_chris": 0,
            "Impulse_Vector_Duration_chris": 0,
            "Impulse_Vector_Slope_chris": 0.0,
        }

    # Combine and sort by recency
    extrema = sorted(
        [(i, pseudo_prices[i], 'peak') for i in peaks_idx] +
        [(i, pseudo_prices[i], 'valley') for i in valleys_idx],
        key=lambda x: x[0],
        reverse=True
    )

    # Find last valid peak-valley or valley-peak pair
    for i in range(len(extrema) - 1):
        first, second = extrema[i + 1], extrema[i]
        if (first[2] == 'valley' and second[2] == 'peak') or (first[2] == 'peak' and second[2] == 'valley'):
            idx1, price1 = first[0], first[1]
            idx2, price2 = second[0], second[1]
            duration = abs(idx2 - idx1)
            magnitude = abs(price2 - price1) / price1 if price1 != 0 else 0
            direction = 1 if price2 > price1 else -1
            slope = magnitude / duration if duration > 0 else 0
            return {
                "Impulse_Vector_Magnitude_chris": round(magnitude, 6),
                "Impulse_Vector_Direction_chris": direction,
                "Impulse_Vector_Duration_chris": duration,
                "Impulse_Vector_Slope_chris": round(slope, 6),
            }

    return {
        "Impulse_Vector_Magnitude_chris": 0.0,
        "Impulse_Vector_Direction_chris": 0,
        "Impulse_Vector_Duration_chris": 0,
        "Impulse_Vector_Slope_chris": 0.0,
    }


def consolidate_impulse_features(impulse_features: dict) -> dict:
    """
    Consolidate impulse vector components into more ML-friendly features.
    
    Args:
        impulse_features: Dict containing the 4 impulse vector components
        
    Returns:
        dict: Consolidated impulse features
    """
    magnitude = impulse_features.get("Impulse_Vector_Magnitude_chris", 0.0)
    direction = impulse_features.get("Impulse_Vector_Direction_chris", 0)
    duration = impulse_features.get("Impulse_Vector_Duration_chris", 0)
    slope = impulse_features.get("Impulse_Vector_Slope_chris", 0.0)
    
    consolidated = {}
    
    # 1. Keep original components (normalized)
    consolidated["Impulse_Magnitude"] = np.clip(magnitude * 100, 0.0, 10.0)  # Convert to percentage, cap at 10%
    consolidated["Impulse_Direction"] = direction  # Keep as -1, 0, 1
    consolidated["Impulse_Duration"] = np.clip(duration, 0, 30)  # Cap at 30 minutes
    consolidated["Impulse_Slope"] = np.clip(slope * 1000, -5.0, 5.0)  # Scale and cap
    
    # 2. Create composite features
    if magnitude > 0 and duration > 0:
        # Impulse strength: combines magnitude and direction
        consolidated["Impulse_Strength"] = magnitude * direction * 100  # Signed strength
        
        # Impulse momentum: magnitude normalized by duration
        consolidated["Impulse_Momentum"] = (magnitude / np.sqrt(duration)) * 100
        
        # Impulse quality: how sustained the movement is
        if duration >= 3:  # At least 3 minutes
            consolidated["Impulse_Quality"] = np.clip(magnitude * np.log(duration), 0.0, 2.0)
        else:
            consolidated["Impulse_Quality"] = 0.0
    else:
        consolidated["Impulse_Strength"] = 0.0
        consolidated["Impulse_Momentum"] = 0.0
        consolidated["Impulse_Quality"] = 0.0
    
    # 3. Categorical impulse type
    if magnitude > 0.002:  # More than 0.2% movement
        if duration <= 5:
            consolidated["Impulse_Type"] = 1  # Quick impulse
        elif duration <= 15:
            consolidated["Impulse_Type"] = 2  # Medium impulse
        else:
            consolidated["Impulse_Type"] = 3  # Sustained impulse
    else:
        consolidated["Impulse_Type"] = 0  # No significant impulse
    
    return consolidated


def optimize_features_enhanced(features_dict, intra_day_df=None, target_value=None, 
                              apply_normalization=True, create_interactions=True, 
                              include_impulse=True):
    """
    Enhanced feature optimization that includes impulse vector features.
    
    Args:
        features_dict (dict): Dictionary of calculated features
        intra_day_df (pd.DataFrame, optional): Intraday data for impulse calculation
        target_value (bool/int, optional): Target value for feature selection
        apply_normalization (bool): Whether to apply smart normalization
        create_interactions (bool): Whether to create interaction features
        include_impulse (bool): Whether to calculate and include impulse features
    
    Returns:
        dict: Optimized features dictionary with impulse features
    """
    
    # Start with the base optimization
    optimized_features = optimize_features_base(features_dict, target_value, 
                                              apply_normalization, create_interactions)
    
    # Add impulse vector features if requested and data is available
    if include_impulse and intra_day_df is not None:
        try:
            # Calculate impulse vector
            impulse_features = calculate_peak_valley_vector_chris_combined(intra_day_df)
            
            # Consolidate impulse features
            consolidated_impulse = consolidate_impulse_features(impulse_features)
            
            # Add to optimized features
            optimized_features.update(consolidated_impulse)
            
            # Create impulse-based interactions with existing features
            if create_interactions:
                optimized_features.update(create_impulse_interactions(optimized_features))
                
        except Exception as e:
            print(f"Warning: Impulse feature calculation failed: {e}")
    
    return optimized_features


def create_impulse_interactions(features: dict) -> dict:
    """
    Create interaction features between impulse and other key features.
    
    Args:
        features: Dictionary of features including impulse features
        
    Returns:
        dict: New interaction features
    """
    interactions = {}
    
    # Get key features for interactions
    impulse_strength = features.get('Impulse_Strength', 0)
    impulse_momentum = features.get('Impulse_Momentum', 0)
    impulse_direction = features.get('Impulse_Direction', 0)
    
    # Interaction 1: Impulse + Volume
    volume_vs_sma = features.get('Volume_vs_SMA10', 1)
    if abs(impulse_strength) > 0.1 and volume_vs_sma > 0:
        interactions['Impulse_Volume_Confirmation'] = impulse_strength * np.log(volume_vs_sma)
    else:
        interactions['Impulse_Volume_Confirmation'] = 0.0
    
    # Interaction 2: Impulse + RSI regime
    rsi = features.get('RSI_14', 50)
    if abs(impulse_strength) > 0.1:
        # Strong impulse in oversold/overbought territory
        if rsi < 30:
            interactions['Impulse_RSI_Oversold'] = abs(impulse_strength) * (30 - rsi) / 30
        elif rsi > 70:
            interactions['Impulse_RSI_Overbought'] = abs(impulse_strength) * (rsi - 70) / 30
        else:
            interactions['Impulse_RSI_Oversold'] = 0.0
            interactions['Impulse_RSI_Overbought'] = 0.0
    else:
        interactions['Impulse_RSI_Oversold'] = 0.0
        interactions['Impulse_RSI_Overbought'] = 0.0
    
    # Interaction 3: Impulse + Support/Resistance
    support_proximity = features.get('Support_Resistance_Proximity', 5.0)
    if abs(impulse_strength) > 0.1 and support_proximity < 2.0:  # Close to S/R level
        # Strong impulse near support/resistance
        interactions['Impulse_Near_Support_Resistance'] = abs(impulse_strength) / support_proximity
    else:
        interactions['Impulse_Near_Support_Resistance'] = 0.0
    
    # Interaction 4: Impulse alignment with existing momentum
    mid_momentum = features.get('Mid_Price_Momentum', 0)
    if abs(impulse_strength) > 0.1 and abs(mid_momentum) > 0.1:
        # Check if impulse direction aligns with mid-price momentum
        momentum_direction = 1 if mid_momentum > 0 else -1
        if impulse_direction == momentum_direction:
            interactions['Impulse_Momentum_Alignment'] = abs(impulse_strength) * abs(mid_momentum)
        else:
            interactions['Impulse_Momentum_Alignment'] = -abs(impulse_strength) * abs(mid_momentum)
    else:
        interactions['Impulse_Momentum_Alignment'] = 0.0
    
    # Apply reasonable bounds to interactions
    for key, value in interactions.items():
        if isinstance(value, (int, float)) and not np.isnan(value):
            interactions[key] = np.clip(value, -10.0, 10.0)
    
    return interactions


def optimize_features_base(features_dict, target_value=None, apply_normalization=True, create_interactions=True):
    """
    Base optimization function (your existing optimization logic).
    """
    # Create a copy to avoid modifying original
    optimized_features = features_dict.copy()
    
    # 1. REMOVE PROBLEMATIC FEATURES
    problematic_features = [
        'Breakout_Confirmation',  # 95.76% same values
        'VWAP',  # Keep only normalized version
        'Bid_Ask_Spread',  # Replace with percentage version if available
        'ATR_14',  # Replace with percentage version if available
        'Bollinger_Bands_20_2',  # Keep only width
        'Volatility_Regime_Shift',  # Very skewed, replace with normalized version
    ]
    
    for feature in problematic_features:
        if feature in optimized_features:
            del optimized_features[feature]
    
    # 2. HANDLE EXTREME OUTLIERS WITH WINSORIZATION
    outlier_features = [
        'MACD_12_26_9', 'Return_5min', 'Price_vs_SMA50', 'Volume_vs_SMA10',
        'Mid_Price_Momentum', 'Price_Acceleration', 'Volume_Zscore_Intraday', 
        'Momentum_Acceleration', 'Price_Impact', 'Volume_Rate_of_Change'
    ]
    
    for feature in outlier_features:
        if feature in optimized_features:
            value = optimized_features[feature]
            if isinstance(value, (int, float)) and not np.isnan(value):
                # Apply simple bounds based on feature type
                if 'Return' in feature or 'MACD' in feature:
                    optimized_features[feature] = np.clip(value, -10.0, 10.0)
                elif 'Volume' in feature and 'vs' in feature:
                    optimized_features[feature] = np.clip(value, 0.0, 50.0)
                elif 'Price_vs' in feature:
                    optimized_features[feature] = np.clip(value, -20.0, 20.0)
                elif 'Momentum' in feature:
                    optimized_features[feature] = np.clip(value, -5.0, 5.0)
    
    # 3. RESOLVE HIGH CORRELATION PAIRS
    correlation_removals = [
        'RSI_Regime_Numeric',  # Keep RSI_14 instead
        'Volume_Zscore_Intraday',  # Keep Volume_Percentile_Intraday instead
    ]
    
    for feature in correlation_removals:
        if feature in optimized_features:
            del optimized_features[feature]
    
    # 4. CREATE INTERACTION FEATURES
    if create_interactions:
        # Risk-Return relationship
        if 'Max_Drawdown_30' in optimized_features and 'Return_Since_Open_Enhanced' in optimized_features:
            dd = optimized_features['Max_Drawdown_30']
            ret = optimized_features['Return_Since_Open_Enhanced']
            if isinstance(dd, (int, float)) and isinstance(ret, (int, float)):
                if not (np.isnan(dd) or np.isnan(ret)):
                    optimized_features['Max_Drawdown_Return_Interaction'] = dd * ret
        
        # Support/Resistance with Momentum
        if 'Support_Resistance_Proximity' in optimized_features and 'Mid_Price_Momentum' in optimized_features:
            support = optimized_features['Support_Resistance_Proximity']
            momentum = optimized_features['Mid_Price_Momentum']
            if isinstance(support, (int, float)) and isinstance(momentum, (int, float)):
                if not (np.isnan(support) or np.isnan(momentum)):
                    optimized_features['Support_Momentum_Interaction'] = support * momentum
        
        # RSI momentum signal
        if 'RSI_14' in optimized_features and 'RSI_Distance_From_50' in optimized_features:
            rsi = optimized_features['RSI_14']
            rsi_dist = optimized_features['RSI_Distance_From_50']
            if isinstance(rsi, (int, float)) and isinstance(rsi_dist, (int, float)):
                if not (np.isnan(rsi) or np.isnan(rsi_dist)):
                    optimized_features['RSI_Momentum_Signal'] = (rsi - 50) * rsi_dist
    
    # 5. APPLY SMART NORMALIZATION (simplified for base function)
    if apply_normalization:
        # Percentage features to 0-1 range
        percentage_features = [
            'Spread_Percentile_20', 'Volatility_Percentile_100', 
            'Volume_Percentile_Intraday', 'Intraday_Performance_Percentile'
        ]
        
        for feature in percentage_features:
            if feature in optimized_features:
                value = optimized_features[feature]
                if isinstance(value, (int, float)) and not np.isnan(value):
                    if value >= 0 and value <= 100:
                        optimized_features[feature] = value / 100.0
                    else:
                        optimized_features[feature] = np.clip(value, 0.0, 1.0)
    
    # 6. ENSURE FEATURE CONSISTENCY
    feature_bounds = {
        'RSI_14': (0.0, 100.0),
        'MACD_12_26_9': (-2.0, 2.0),
        'Return_5min': (-5.0, 5.0),
        'Position_In_Day_Range': (0.0, 1.0),
        'Price_Position_In_Spread': (0.0, 1.0),
        'Order_Book_Imbalance': (-1.0, 1.0),
        'Max_Drawdown_30': (-50.0, 0.0)
    }
    
    for feature, (min_val, max_val) in feature_bounds.items():
        if feature in optimized_features:
            value = optimized_features[feature]
            if isinstance(value, (int, float)) and not np.isnan(value):
                optimized_features[feature] = np.clip(value, min_val, max_val)
    
    # 7. REMOVE INVALID VALUES
    features_to_remove = []
    for feature, value in optimized_features.items():
        if isinstance(value, (int, float)):
            if np.isnan(value) or np.isinf(value):
                features_to_remove.append(feature)
        elif value is None:
            features_to_remove.append(feature)
    
    for feature in features_to_remove:
        del optimized_features[feature]
    
    return optimized_features


# Wrapper function to maintain compatibility
def optimize_features(features_dict, intra_day_df=None, target_value=None, 
                     apply_normalization=True, create_interactions=True):
    """
    Main optimization function that includes impulse vector features.
    """
    return optimize_features_enhanced(
        features_dict, 
        intra_day_df=intra_day_df,
        target_value=target_value,
        apply_normalization=apply_normalization,
        create_interactions=create_interactions,
        include_impulse=True
    )


# Example integration for your workflow
def example_usage():
    """
    Example of how to integrate the enhanced optimization into your workflow.
    """
    print("Enhanced integration example:")
    print("# Your existing code:")
    print("features = calculate_features(real_time_df, intra_day_df, daily_df)")
    print("extra_features = calculate_additional_features(real_time_df, intra_day_df, daily_df)")
    print("features.update(extra_features)")
    print("signal_result = evaluate_buy_signal(...)")
    print("features.update(signal_result)")
    print("")
    print("# Enhanced optimization:")
    print("if (len(intra_day_df) >= 15 and signal_result.get('reason') in valid_reasons):")
    print("    optimized_features = optimize_features(features, intra_day_df=intra_day_df)")
    print("    features_list.append(optimized_features)")
    print("    processed_count += 1")


if __name__ == "__main__":
    # Test the enhanced optimization
    example_features = {
        'RSI_14': 65.5,
        'MACD_12_26_9': 0.015,
        'Return_5min': 0.5,
        'Max_Drawdown_30': -0.15,
        'Volume_vs_SMA10': 1.8,
        'Support_Resistance_Proximity': 1.5,
        'Mid_Price_Momentum': 0.1,
        'buy': True
    }
    
    # Create sample intraday data
    sample_intra = pd.DataFrame({
        'high_1min': np.random.normal(100, 1, 35),
        'low_1min': np.random.normal(99, 1, 35)
    })
    
    print("Testing enhanced optimization with impulse features...")
    optimized = optimize_features(example_features, intra_day_df=sample_intra)
    
    # Show impulse features
    impulse_features = {k: v for k, v in optimized.items() if 'Impulse' in k}
    print(f"\nImpulse features added: {len(impulse_features)}")
    for k, v in impulse_features.items():
        print(f"  {k}: {v}")