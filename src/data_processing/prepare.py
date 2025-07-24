import os
import joblib
import pandas as pd
import numpy as np
from utils.utils import (
    load_data_optimized, treat_outliers, impute_missing, remove_low_variance_features,
    ensemble_feature_ranking, conservative_normalization, intelligent_sampling
)
from sklearn.model_selection import train_test_split

def prepare_data(df_path, output_dir="prepared_data", top_n_features=20, sample_size=200000, sampling_strategy='balanced', fast=False):
    if fast:
        print("   ⚡ FAST MODE: Reduced sample size and features for debugging")
        sample_size = 50000
        top_n_features = 10
    
    # Define the high-impact features that must be used
    high_impact_features = selected_features = [
    "Time_Of_Day_Cyclical_Sin", "Time_Of_Day_Cyclical_Cos",
    "Bid_Ask_Spread_Pct", "Volume_Weighted_Order_Book_Imbalance",
    "Volume_Percentile_Intraday", "Volume_vs_SMA10",  # ✅ Add-back
    "Volatility_Regime_Shift_Norm", "Max_Drawdown_Return_Interaction",
    "Support_Momentum_Interaction",
    "Max_Drawdown_30", "Return_Since_Open_Enhanced", "ATR_Stop_Risk_Pct",
    "Price_vs_SMA50", "Support_Resistance_Proximity",
    "RSI_14", "Return_5min", "Mid_Price_Momentum",   # ✅ Add-back
    "Price_Acceleration", "RSI_Distance_From_50", "Price_Action_Quality"
    ]


    
    print("\n📊 1. LOADING DATA WITH M3 OPTIMIZATIONS")
    df = load_data_optimized(df_path)
    print(f"   ✅ Data loaded: {df.shape[0]:,} samples, {df.shape[1]} features")
    print(f"   💾 Memory usage: {df.memory_usage(deep=True).sum() / 1024**3:.2f} GB")

    # 2. Filter and prepare
    valid_reasons = [
        "reached_2r_success", "reached_3r_success",
        "reached_4r_success", "reached_negative_1r"
    ]
    df_filtered = df[df['reason'].isin(valid_reasons)].copy()
    print(f"   ✅ Filtered data: {df_filtered.shape[0]:,} samples")

    # 3. Clean column names
    df_filtered.columns = df_filtered.columns.str.strip().str.replace(" ", "_").str.replace("-", "_")

    # 4. Impute missing values
    df_imputed = impute_missing(df_filtered)
    print("   ✅ Missing values imputed")

    # 5. Outlier treatment (winsorization)
    feature_cols = [col for col in df_imputed.columns if col not in ['buy', 'reason']]
    df_outlier = df_imputed.copy()
    df_outlier[feature_cols] = treat_outliers(df_imputed[feature_cols])
    print("   ✅ Outliers treated")

    # 6. Remove low-variance features
    reduced_features = remove_low_variance_features(df_outlier[feature_cols])
    feature_cols = list(reduced_features.columns)
    df_lv = df_outlier[feature_cols + [col for col in ['buy', 'reason'] if col in df_outlier.columns]]
    print("   ✅ Low-variance features removed")

    # 7. Select only high-impact features that exist in the data
    available_features = [f for f in high_impact_features if f in df_lv.columns]
    missing_features = [f for f in high_impact_features if f not in df_lv.columns]
    
    if missing_features:
        print(f"   ⚠️ Missing high-impact features: {missing_features}")
    
    print(f"   ✅ Using {len(available_features)} high-impact features: {available_features}")

    # 8. Filter to only numeric features for normalization
    X_selected = df_lv[available_features]
    numeric_features = X_selected.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_features = [f for f in available_features if f not in numeric_features]
    
    if non_numeric_features:
        print(f"   ⚠️ Excluding non-numeric features from normalization: {non_numeric_features}")
        available_features = numeric_features
    
    print(f"   ✅ Using {len(available_features)} numeric features for normalization")

    # 9. Conservative normalization
    X_selected = df_lv[available_features]
    X_norm = conservative_normalization(X_selected)
    print("   ✅ Conservative normalization applied")

    # 10. Recombine with target for sampling
    df_final = pd.concat([X_norm, df_lv['buy']], axis=1)

    # 11. Intelligent sampling
    df_sampled = intelligent_sampling(df_final, target='buy', target_size=sample_size, strategy=sampling_strategy)
    print(f"   ✅ Intelligent sampling complete: {df_sampled.shape[0]:,} samples")

    # 12. Split data
    X = df_sampled[available_features]
    y = df_sampled['buy']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    print(f"\n📊 Data splits:")
    print(f"   Training: {X_train.shape[0]:,} samples")
    print(f"   Validation: {X_val.shape[0]:,} samples")
    print(f"   Test: {X_test.shape[0]:,} samples")

    # 13. Save splits and feature rankings
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump({
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'available_features': available_features
    }, os.path.join(output_dir, 'data_splits.pkl'))
    
    # Create a simple feature rankings dataframe for compatibility
    feature_rankings = pd.DataFrame({
        'Feature': available_features,
        'Rank': range(1, len(available_features) + 1),
        'Score': [1.0] * len(available_features)  # Equal importance since we're using fixed features
    })
    feature_rankings.to_csv(os.path.join(output_dir, 'feature_rankings.csv'), index=False)
    print(f"   ✅ Data splits and feature rankings saved to {output_dir}/")

