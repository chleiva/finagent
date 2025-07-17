#!/usr/bin/env python3
"""
Model Wrapper for Easy Access to Trained Models
==============================================

This module provides a simple interface to access trained model artifacts
and make predictions with them.

Usage:
    from src.utils.model_wrapper import ModelWrapper
    
    # Load a model for AAPL
    model = ModelWrapper("AAPL")
    
    # Make predictions
    predictions = model.predict(new_data)
    
    # Get model info
    info = model.get_info()
"""

import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import json

class ModelWrapper:
    """
    A wrapper class for easy access to trained model artifacts.
    
    This class provides a simple interface to:
    - Load trained models
    - Make predictions
    - Access model metadata
    - Get feature information
    """
    
    def __init__(self, symbol: str, model_index_path: str = "model_artifacts/model_index.csv"):
        """
        Initialize the ModelWrapper for a specific stock symbol.
        
        Args:
            symbol: Stock symbol (e.g., "AAPL", "MSFT")
            model_index_path: Path to the model index CSV file
        """
        self.symbol = symbol.upper()
        self.model_index_path = model_index_path
        self.model_data = None
        self.model_package = None
        self.features = None
        self.best_model = None
        self.best_model_name = None
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Load the model artifacts for the given symbol."""
        if not os.path.exists(self.model_index_path):
            raise FileNotFoundError(f"Model index not found: {self.model_index_path}")
        
        # Load model index
        index_df = pd.read_csv(self.model_index_path)
        
        # Find the most recent model for this symbol
        symbol_models = index_df[index_df['description'].str.contains(self.symbol, case=False, na=False)]
        
        if symbol_models.empty:
            raise ValueError(f"No models found for symbol: {self.symbol}")
        
        # Get the most recent model
        symbol_models['date_created'] = pd.to_datetime(symbol_models['date_created'])
        latest_model = symbol_models.loc[symbol_models['date_created'].idxmax()]
        
        self.model_data = latest_model.to_dict()
        
        # Load the model package
        model_package_path = os.path.join(latest_model['artifact_path'], 'models', 'model_package.pkl')
        if not os.path.exists(model_package_path):
            raise FileNotFoundError(f"Model package not found: {model_package_path}")
        
        self.model_package = joblib.load(model_package_path)
        self.best_model_name = latest_model['best_model']
        self.best_model = self.model_package[self.best_model_name]
        self.features = self.model_package['features']
        
        print(f"✅ Loaded {self.symbol} model: {self.best_model_name}")
        print(f"   📁 Path: {latest_model['artifact_path']}")
        print(f"   📊 Test AUC: {latest_model['test_auc']:.4f}")
        print(f"   🎯 Test F1: {latest_model['test_f1']:.4f}")
    
    def get_info(self) -> Dict:
        """
        Get comprehensive information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        if self.model_data is None:
            raise ValueError("No model loaded")
        
        return {
            'symbol': self.symbol,
            'model_id': self.model_data['model_id'],
            'description': self.model_data['description'],
            'date_created': self.model_data['date_created'],
            'best_model': self.best_model_name,
            'test_auc': self.model_data['test_auc'],
            'test_f1': self.model_data['test_f1'],
            'best_precision': self.model_data['best_precision'],
            'best_signals': self.model_data['best_signals'],
            'artifact_path': self.model_data['artifact_path'],
            'num_features': len(self.features) if self.features is not None else 0,
            'features': self.features.tolist() if hasattr(self.features, 'tolist') else list(self.features) if self.features is not None else []
        }
    
    def predict(self, data: Union[pd.DataFrame, np.ndarray], 
                threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using the loaded model.
        
        Args:
            data: Input data (DataFrame or numpy array)
            threshold: Classification threshold (default: 0.5)
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.best_model is None:
            raise ValueError("No model loaded")
        
        # Convert to DataFrame if needed
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=self.features)
        
        # Ensure we have the right features
        if self.features is not None:
            missing_features = set(self.features) - set(data.columns)
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            
            # Select only the features used by the model
            data = data[self.features]
        
        # Make predictions
        probabilities = self.best_model.predict_proba(data)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
        
        return predictions, probabilities
    
    def predict_proba(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            data: Input data
            
        Returns:
            Array of probabilities
        """
        _, probabilities = self.predict(data, threshold=0.0)
        return probabilities
    
    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance scores
        """
        if self.best_model is None or self.features is None:
            raise ValueError("No model loaded")
        
        if not hasattr(self.best_model, 'feature_importances_'):
            raise ValueError("Model does not have feature importances")
        
        importance_df = pd.DataFrame({
            'Feature': self.features,
            'Importance': self.best_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def get_optimal_threshold(self) -> float:
        """
        Get the optimal threshold for this model.
        
        Returns:
            Optimal threshold value
        """
        if self.model_data is None:
            raise ValueError("No model loaded")
        
        # Try to get from model data
        if 'best_threshold' in self.model_data:
            return self.model_data['best_threshold']
        
        # Default threshold
        return 0.5
    
    def get_trading_metrics(self) -> Dict:
        """
        Get trading-specific metrics.
        
        Returns:
            Dictionary with trading metrics
        """
        if self.model_data is None:
            raise ValueError("No model loaded")
        
        return {
            'best_precision': self.model_data['best_precision'],
            'best_signals': self.model_data['best_signals'],
            'best_expected_value': self.model_data.get('best_expected_value', 0.0),
            'optimal_threshold': self.get_optimal_threshold()
        }
    
    def save_predictions(self, data: Union[pd.DataFrame, np.ndarray], 
                        output_path: str, 
                        threshold: float = None) -> str:
        """
        Make predictions and save to CSV file.
        
        Args:
            data: Input data
            output_path: Path to save predictions
            threshold: Classification threshold (uses optimal if None)
            
        Returns:
            Path to saved predictions file
        """
        if threshold is None:
            threshold = self.get_optimal_threshold()
        
        predictions, probabilities = self.predict(data, threshold)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'prediction': predictions,
            'probability': probabilities,
            'threshold': threshold
        })
        
        # Save to CSV
        results_df.to_csv(output_path, index=False)
        
        print(f"✅ Predictions saved to: {output_path}")
        print(f"   📊 Predictions: {predictions.sum()} buy signals out of {len(predictions)} total")
        print(f"   🎯 Threshold: {threshold:.3f}")
        
        return output_path

def list_available_models(model_index_path: str = "model_artifacts/model_index.csv") -> pd.DataFrame:
    """
    List all available trained models.
    
    Args:
        model_index_path: Path to model index
        
    Returns:
        DataFrame with model information
    """
    if not os.path.exists(model_index_path):
        raise FileNotFoundError(f"Model index not found: {model_index_path}")
    
    index_df = pd.read_csv(model_index_path)
    index_df['date_created'] = pd.to_datetime(index_df['date_created'])
    
    return index_df.sort_values('date_created', ascending=False)

def get_latest_model_for_symbol(symbol: str, 
                               model_index_path: str = "model_artifacts/model_index.csv") -> Optional[Dict]:
    """
    Get information about the latest model for a specific symbol.
    
    Args:
        symbol: Stock symbol
        model_index_path: Path to model index
        
    Returns:
        Dictionary with model information or None if not found
    """
    index_df = list_available_models(model_index_path)
    
    symbol_models = index_df[index_df['description'].str.contains(symbol.upper(), case=False, na=False)]
    
    if symbol_models.empty:
        return None
    
    return symbol_models.iloc[0].to_dict()

# Example usage
if __name__ == "__main__":
    # Example: Load AAPL model and make predictions
    try:
        model = ModelWrapper("AAPL")
        
        # Get model info
        info = model.get_info()
        print(f"\n📊 Model Info:")
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        # Get feature importance
        importance = model.get_feature_importance(top_n=5)
        print(f"\n🔝 Top 5 Features:")
        print(importance)
        
        # Get trading metrics
        trading_metrics = model.get_trading_metrics()
        print(f"\n💰 Trading Metrics:")
        for key, value in trading_metrics.items():
            print(f"   {key}: {value}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have trained models available!") 