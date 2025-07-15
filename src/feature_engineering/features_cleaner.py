import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import PowerTransformer, RobustScaler
from sklearn.feature_selection import mutual_info_classif, f_classif
import warnings
warnings.filterwarnings('ignore')

# Enhanced Feature Cleanup and Validation

def clean_and_validate_features(features_dict):
    """
    Clean and validate features, removing problematic, invalid, or useless columns.
    
    Args:
        features_dict (dict): Raw features dictionary
        
    Returns:
        dict: Cleaned features dictionary
    """
    
    # Start with a copy
    cleaned_features = features_dict.copy()
    
    # 1. REMOVE KNOWN PROBLEMATIC FEATURES
    problematic_features = {
        # High correlation features (keep the better one)
        'Max_Drawdown_Return_Interaction',  # Remove (keep Return_Since_Open_Enhanced)
        'RSI_Momentum_Signal',              # Remove (keep RSI_14)
        'RSI_Regime_Numeric',              # Remove (keep RSI_14)
        'Volume_Zscore_Intraday',          # Remove (keep Volume_Percentile_Intraday)
        'Normalized_Price_VWAP',           # Remove (keep Intraday_Performance_Percentile)
        
        # Low variance / mostly same values
        'Breakout_Confirmation',           # 95.76% same values
        
        # Redundant absolute values (keep only relative/normalized versions)
        'VWAP',                           # Keep only Normalized_Price_VWAP
        'Bid_Ask_Spread',                 # Keep only percentage version if available
        'ATR_14',                         # Keep only ATR_Stop_Risk_Pct
        'Bollinger_Bands_20_2',          # Keep only Bollinger_Band_Width
        
        # Very skewed features that don't normalize well
        'Volatility_Regime_Shift',        # Keep normalized version if available
        
        # Missing or unreliable features
        'Time_Of_Day_Cyclical',          # Often missing
        'Trade_count_per_minute',         # Often unreliable
        'Volume_weighted_order_book_imbalance',  # Redundant with Order_Book_Imbalance
    }
    
    for feature in problematic_features:
        if feature in cleaned_features:
            del cleaned_features[feature]
    
    # 2. REMOVE INVALID VALUES
    invalid_features = []
    
    for feature, value in cleaned_features.items():
        # Skip target variable
        if feature in ['buy', 'sell', 'reason', 'entry_time', 'exit_time']:
            continue
            
        # Check for invalid values
        if value is None:
            invalid_features.append(feature)
        elif isinstance(value, (int, float)):
            if np.isnan(value) or np.isinf(value):
                invalid_features.append(feature)
        elif isinstance(value, str):
            # Remove string features that aren't categorical targets
            if feature not in ['reason', 'symbol', 'date']:
                invalid_features.append(feature)
    
    # Remove invalid features
    for feature in invalid_features:
        del cleaned_features[feature]
    
    # 3. REMOVE FEATURES WITH ZERO VARIANCE
    zero_variance_features = []
    
    for feature, value in cleaned_features.items():
        # Skip non-numeric features
        if not isinstance(value, (int, float)):
            continue
            
        # Check if feature has meaningful variation
        if isinstance(value, (int, float)):
            # For boolean-like features
            if value in [0, 1] and feature.endswith('_confirmation'):
                # These might be low-variance boolean features
                continue
            # For percentage features that are always 0 or 100
            elif feature.endswith('_percentile') and value in [0.0, 100.0]:
                # Might indicate no variation
                continue
    
    # 4. VALIDATE FEATURE RANGES AND APPLY BOUNDS
    feature_bounds = {
        # Technical indicators
        'RSI_14': (0.0, 100.0),
        'MACD_12_26_9': (-5.0, 5.0),
        'Bollinger_Band_Width': (0.0, 1.0),
        
        # Returns (reasonable intraday bounds)
        'Return_5min': (-10.0, 10.0),
        'Return_since_open': (-25.0, 25.0),
        'Return_Since_Open_Enhanced': (-25.0, 25.0),
        
        # Position indicators
        'Position_in_day_range': (0.0, 1.0),
        'Price_Position_In_Spread': (0.0, 1.0),
        'Intraday_Performance_Percentile': (0.0, 1.0),
        
        # Percentile features
        'Spread_Percentile_20': (0.0, 1.0),
        'Volatility_Percentile_100': (0.0, 100.0),
        'Volume_Percentile_Intraday': (0.0, 1.0),
        
        # Volume ratios
        'Volume_vs_SMA10': (0.0, 100.0),
        
        # Price relationships
        'Price_vs_SMA50': (-50.0, 50.0),
        
        # Order book
        'Order_Book_Imbalance': (-1.0, 1.0),
        
        # Risk metrics
        'ATR_Stop_Risk_Pct': (0.0, 50.0),
        'Max_Drawdown_30': (-100.0, 0.0),
        
        # Distance/proximity features
        'Support_Resistance_Proximity': (0.0, 20.0),
        
        # Momentum features
        'Mid_Price_Momentum': (-10.0, 10.0),
        'Price_Acceleration': (-20.0, 20.0),
        'Momentum_Acceleration': (-10.0, 10.0),
        'Momentum_Consistency': (0.0, 1.0),
        
        # Quality metrics
        'Order_Book_Stability': (0.0, 1.0),
        'Price_Action_Quality': (0.0, 1.0),
    }
    
    # Apply bounds and validate
    for feature, value in list(cleaned_features.items()):
        if feature in feature_bounds and isinstance(value, (int, float)):
            min_val, max_val = feature_bounds[feature]
            
            # Check if value is completely out of reasonable range
            if value < min_val * 2 or value > max_val * 2:  # 2x tolerance
                # Value is extremely unreasonable, remove the feature
                del cleaned_features[feature]
                continue
            
            # Apply bounds
            cleaned_features[feature] = np.clip(value, min_val, max_val)
    
    # 5. REMOVE REDUNDANT FEATURES (keep the better version)
    redundancy_groups = [
        # Keep normalized versions over absolute
        (['Normalized_Price_VWAP', 'VWAP'], 'Normalized_Price_VWAP'),
        (['ATR_Stop_Risk_Pct', 'ATR_14'], 'ATR_Stop_Risk_Pct'),
        (['Bollinger_Band_Width', 'Bollinger_Bands_20_2'], 'Bollinger_Band_Width'),
        
        # Keep enhanced versions over basic
        (['Return_Since_Open_Enhanced', 'Return_since_open'], 'Return_Since_Open_Enhanced'),
        
        # Keep percentile versions over z-score
        (['Volume_Percentile_Intraday', 'Volume_Zscore_Intraday'], 'Volume_Percentile_Intraday'),
        
        # ✅ NEW: Remove high correlation features (keep the more informative one)
        (['Return_Since_Open_Enhanced', 'Max_Drawdown_Return_Interaction'], 'Return_Since_Open_Enhanced'),
        (['RSI_14', 'RSI_Momentum_Signal'], 'RSI_14'),
    ]
    
    for group, preferred in redundancy_groups:
        # Check which features exist in this group
        existing = [f for f in group if f in cleaned_features]
        
        if len(existing) > 1 and preferred in existing:
            # Remove all except the preferred
            for feature in existing:
                if feature != preferred:
                    if feature in cleaned_features:
                        del cleaned_features[feature]
    
    # 6. ENSURE MINIMUM VIABLE FEATURE SET
    # Check that we have key feature categories
    required_categories = {
        'momentum': ['RSI_14', 'MACD_12_26_9', 'Mid_Price_Momentum'],
        'returns': ['Return_5min', 'Return_since_open', 'Return_Since_Open_Enhanced'],
        'risk': ['Max_Drawdown_30', 'ATR_Stop_Risk_Pct'],
        'volume': ['Volume_vs_SMA10', 'Volume_Percentile_Intraday'],
        'pattern': ['Support_Resistance_Proximity', 'Price_vs_SMA50']
    }
    
    missing_categories = []
    for category, features in required_categories.items():
        if not any(f in cleaned_features for f in features):
            missing_categories.append(category)
    
    if missing_categories:
        print(f"⚠️ Warning: Missing feature categories: {missing_categories}")
    
    # 7. FINAL VALIDATION
    final_feature_count = len([k for k in cleaned_features.keys() 
                              if k not in ['buy', 'sell', 'reason', 'entry_time', 'exit_time']])
    
    if final_feature_count < 10:
        print(f"⚠️ Warning: Only {final_feature_count} features remaining after cleanup")
    
    return cleaned_features


# Enhanced integration for your workflow
def process_features_enhanced():
    """
    Example of enhanced feature processing workflow
    """
    
    # Define valid reasons (same as your analysis)
    valid_reasons = {
        "end_of_day_no_2r",
        "reached_2r_success",
        "reached_3r_success", 
        "reached_4r_success",
        "reached_negative_1r"
    }

    # Your existing code with enhancements:
    """
    # Add signal results to features
    features.update(signal_result)
    
    # Check both conditions: sufficient data AND valid reason
    if (len(intra_day_df) >= 15 and 
        signal_result.get('reason') in valid_reasons):
        
        # 🚀 ENHANCED: Clean features first, then optimize
        cleaned_features = clean_and_validate_features(features)
        optimized_features = optimize_features(cleaned_features, intra_day_df=intra_day_df)
        features_list.append(optimized_features)
        processed_count += 1
    """
    
    return "Enhanced workflow template"


# Batch cleaning function for existing datasets
def clean_existing_dataset(csv_file_path, target_column='buy'):
    """
    Clean an existing dataset CSV file by removing problematic features.
    
    Args:
        csv_file_path (str): Path to the CSV file
        target_column (str): Name of the target column
        
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    
    print(f"🔄 Loading dataset from {csv_file_path}...")
    df = pd.read_csv(csv_file_path)
    print(f"   Original shape: {df.shape}")
    
    # Features to remove based on analysis
    features_to_remove = {
        'Max_Drawdown_Return_Interaction',
        'RSI_Momentum_Signal', 
        'RSI_Regime_Numeric',
        'Volume_Zscore_Intraday',
        'Normalized_Price_VWAP',
        'Breakout_Confirmation',
        'VWAP',
        'Bid_Ask_Spread',
        'ATR_14',
        'Bollinger_Bands_20_2',
        'Volatility_Regime_Shift',
        'Time_Of_Day_Cyclical',
        'Trade_count_per_minute',
        'Volume_weighted_order_book_imbalance'
    }
    
    # Remove features that exist
    existing_removals = [f for f in features_to_remove if f in df.columns]
    if existing_removals:
        df = df.drop(columns=existing_removals)
        print(f"   Removed features: {existing_removals}")
    
    # Check for features with too many missing values
    missing_threshold = 0.5  # Remove features with >50% missing
    high_missing = df.isnull().mean()
    high_missing_features = high_missing[high_missing > missing_threshold].index.tolist()
    
    if high_missing_features:
        df = df.drop(columns=high_missing_features)
        print(f"   Removed high-missing features: {high_missing_features}")
    
    # Check for zero-variance features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    zero_var_features = []
    
    for col in numeric_cols:
        if col != target_column and df[col].nunique() <= 1:
            zero_var_features.append(col)
    
    if zero_var_features:
        df = df.drop(columns=zero_var_features)
        print(f"   Removed zero-variance features: {zero_var_features}")
    
    # Fill remaining missing values
    if df.isnull().sum().sum() > 0:
        # Use median for numeric, mode for categorical
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ['int64', 'float64']:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'unknown', inplace=True)
        print(f"   Filled remaining missing values")
    
    print(f"✅ Cleaned dataset shape: {df.shape}")
    
    # Save cleaned version
    cleaned_file = csv_file_path.replace('.csv', '_cleaned.csv')
    df.to_csv(cleaned_file, index=False)
    print(f"💾 Cleaned dataset saved to: {cleaned_file}")
    
    return df


# Example usage
if __name__ == "__main__":
    # Test the cleaning function
    sample_features = {
        'RSI_14': 65.5,
        'MACD_12_26_9': 0.015,
        'Return_5min': 0.5,
        'Max_Drawdown_30': -0.15,
        'Volume_vs_SMA10': 1.8,
        'Support_Resistance_Proximity': 1.5,
        'Breakout_Confirmation': 0.0,  # Will be removed
        'VWAP': 150.25,  # Will be removed (prefer normalized)
        'Normalized_Price_VWAP': 1.002,  # Will be kept
        'Invalid_Feature': None,  # Will be removed
        'buy': True
    }
    
    print("Original features:")
    for k, v in sample_features.items():
        print(f"  {k}: {v}")
    
    cleaned = clean_and_validate_features(sample_features)
    
    print(f"\nCleaned features:")
    for k, v in cleaned.items():
        print(f"  {k}: {v}")
    
    removed = set(sample_features.keys()) - set(cleaned.keys())
    print(f"\nRemoved features: {removed}")