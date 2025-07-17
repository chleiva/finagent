#!/usr/bin/env python3
"""
Model Inference Adapter
======================

Adapter for real-time inference using trained models and live dataframes.

Usage:
    from src.inference.model_inference_adapter import ModelInferenceAdapter
    adapter = ModelInferenceAdapter()
    output = adapter.infer(symbol, real_time_df, intra_day_df, daily_df)
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any

# Import feature calculation functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'feature_engineering'))
from features import calculate_features
from extra_features import calculate_additional_features

# Import model wrapper
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from model_wrapper import ModelWrapper

class ModelInferenceAdapter:
    """
    Adapter for real-time model inference given live dataframes.
    """
    def __init__(self, model_index_path: str = "model_artifacts/model_index.csv"):
        self.model_index_path = model_index_path
        self.models = {}  # Cache loaded models

    def get_model(self, symbol: str) -> ModelWrapper:
        symbol = symbol.upper()
        if symbol not in self.models:
            self.models[symbol] = ModelWrapper(symbol, model_index_path=self.model_index_path)
        return self.models[symbol]

    def compute_features(self, real_time_df: pd.DataFrame, intra_day_df: pd.DataFrame, daily_df: pd.DataFrame) -> Dict[str, Any]:
        # Calculate base and extra features
        features = calculate_features(real_time_df, intra_day_df, daily_df)
        features.update(calculate_additional_features(real_time_df, intra_day_df, daily_df))
        return features

    def infer(self, symbol: str, real_time_df: pd.DataFrame, intra_day_df: pd.DataFrame, daily_df: pd.DataFrame, threshold: float = None) -> Dict[str, Any]:
        """
        Run inference for a given symbol and data snapshot.
        Args:
            symbol: Stock symbol
            real_time_df: Real-time DataFrame (single row)
            intra_day_df: Intraday DataFrame (up to current minute)
            daily_df: Daily DataFrame (up to current day)
            threshold: Optional threshold for classification
        Returns:
            Dict with prediction, probability, features, and model info
        """
        # 1. Compute features
        features = self.compute_features(real_time_df, intra_day_df, daily_df)

        # 2. Load model
        model = self.get_model(symbol)
        model_features = model.features

        # 3. Prepare feature vector in correct order
        feature_vector = []
        missing_features = []
        for feat in model_features:
            if feat in features:
                feature_vector.append(features[feat])
            else:
                feature_vector.append(0.0)
                missing_features.append(feat)
        X = pd.DataFrame([feature_vector], columns=model_features)

        # 4. Run model prediction
        if threshold is None:
            threshold = model.get_optimal_threshold()
        prediction, probability = model.predict(X, threshold=threshold)

        # 5. Return results
        return {
            'symbol': symbol,
            'prediction': int(prediction[0]),
            'probability': float(probability[0]),
            'features': features,
            'missing_features': missing_features,
            'model_info': model.get_info(),
            'threshold': threshold
        }

# Example usage
def _example():
    # Dummy data for demonstration
    import numpy as np
    import pandas as pd
    symbol = "AAPL"
    # Create dummy dataframes with required columns
    real_time_df = pd.DataFrame({
        'bidPrice': [100], 'bidSize': [10], 'askPrice': [101], 'askSize': [12],
        'lastPrice': [100.5], 'lastSize': [5], 'volume': [1000], 'timestamp': [pd.Timestamp.now()]
    })
    intra_day_df = pd.DataFrame({
        'close_1min': np.random.normal(100, 1, 30),
        'open_1min': np.random.normal(100, 1, 30),
        'volume_1min': np.random.randint(100, 200, 30),
        'ts_event_clean': pd.date_range(end=pd.Timestamp.now(), periods=30, freq='min'),
        'symbol_price': [symbol]*30
    })
    daily_df = pd.DataFrame({
        'date': pd.date_range(end=pd.Timestamp.now(), periods=10, freq='D'),
        'open': np.random.normal(100, 2, 10),
        'high': np.random.normal(102, 2, 10),
        'low': np.random.normal(98, 2, 10),
        'close': np.random.normal(100, 2, 10),
        'volume': np.random.randint(10000, 20000, 10),
        'symbol': [symbol]*10
    })
    adapter = ModelInferenceAdapter()
    result = adapter.infer(symbol, real_time_df, intra_day_df, daily_df)
    print("\n=== INFERENCE RESULT ===")
    for k, v in result.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    _example() 