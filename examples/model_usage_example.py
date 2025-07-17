#!/usr/bin/env python3
"""
Example Usage of ModelWrapper
=============================

This script demonstrates how to use the ModelWrapper class to:
1. Load trained models
2. Make predictions
3. Access model information
4. Get feature importance

Usage:
    python examples/model_usage_example.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.model_wrapper import ModelWrapper, list_available_models, get_latest_model_for_symbol
import pandas as pd
import numpy as np

def main():
    """Main example function"""
    print("🤖 MODEL WRAPPER USAGE EXAMPLE")
    print("=" * 50)
    
    # 1. List all available models
    print("\n📋 1. LISTING ALL AVAILABLE MODELS")
    print("-" * 40)
    try:
        models_df = list_available_models()
        print(f"Found {len(models_df)} trained models:")
        for _, row in models_df.head(5).iterrows():
            print(f"   • {row['description']} (AUC: {row['test_auc']:.4f})")
        if len(models_df) > 5:
            print(f"   ... and {len(models_df) - 5} more models")
    except FileNotFoundError:
        print("❌ No models found. Please train some models first!")
        return
    
    # 2. Get latest model for a specific symbol
    print("\n🔍 2. GETTING LATEST MODEL FOR AAPL")
    print("-" * 40)
    try:
        aapl_info = get_latest_model_for_symbol("AAPL")
        if aapl_info:
            print(f"✅ Latest AAPL model:")
            print(f"   📝 Description: {aapl_info['description']}")
            print(f"   📊 Test AUC: {aapl_info['test_auc']:.4f}")
            print(f"   🎯 Test F1: {aapl_info['test_f1']:.4f}")
            print(f"   📁 Path: {aapl_info['artifact_path']}")
        else:
            print("❌ No AAPL model found")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # 3. Load a model and get information
    print("\n📊 3. LOADING MODEL AND GETTING INFO")
    print("-" * 40)
    try:
        model = ModelWrapper("AAPL")
        
        # Get comprehensive model info
        info = model.get_info()
        print(f"✅ Model loaded successfully!")
        print(f"   🏷️ Symbol: {info['symbol']}")
        print(f"   🤖 Best Model: {info['best_model']}")
        print(f"   📊 Test AUC: {info['test_auc']:.4f}")
        print(f"   🎯 Test F1: {info['test_f1']:.4f}")
        print(f"   📈 Best Precision: {info['best_precision']:.1%}")
        print(f"   🔢 Features: {info['num_features']}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # 4. Get feature importance
    print("\n🔝 4. FEATURE IMPORTANCE")
    print("-" * 40)
    try:
        importance = model.get_feature_importance(top_n=10)
        print("Top 10 most important features:")
        for i, (_, row) in enumerate(importance.iterrows(), 1):
            print(f"   {i:2d}. {row['Feature']:<30} {row['Importance']:.4f}")
    except Exception as e:
        print(f"❌ Error getting feature importance: {e}")
    
    # 5. Get trading metrics
    print("\n💰 5. TRADING METRICS")
    print("-" * 40)
    try:
        trading_metrics = model.get_trading_metrics()
        print("Trading-specific metrics:")
        for key, value in trading_metrics.items():
            if isinstance(value, float):
                if 'precision' in key.lower() or 'threshold' in key.lower():
                    print(f"   {key}: {value:.1%}")
                else:
                    print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")
    except Exception as e:
        print(f"❌ Error getting trading metrics: {e}")
    
    # 6. Example prediction (with dummy data)
    print("\n🎯 6. EXAMPLE PREDICTION")
    print("-" * 40)
    try:
        # Create dummy data with the same features as the model
        features = model.features
        if features is not None:
            # Create random data with the right features
            dummy_data = pd.DataFrame(
                np.random.randn(10, len(features)),
                columns=features
            )
            
            # Make predictions
            predictions, probabilities = model.predict(dummy_data)
            
            print(f"✅ Made predictions on dummy data:")
            print(f"   📊 Buy signals: {predictions.sum()}/{len(predictions)}")
            print(f"   🎯 Average probability: {probabilities.mean():.3f}")
            print(f"   📈 Max probability: {probabilities.max():.3f}")
            print(f"   📉 Min probability: {probabilities.min():.3f}")
            
            # Show first few predictions
            print(f"\n   First 5 predictions:")
            for i in range(min(5, len(predictions))):
                signal = "BUY" if predictions[i] else "HOLD"
                print(f"      {i+1}: {signal} (prob: {probabilities[i]:.3f})")
        else:
            print("❌ No features available for prediction")
            
    except Exception as e:
        print(f"❌ Error making predictions: {e}")
    
    # 7. Save predictions example
    print("\n💾 7. SAVING PREDICTIONS")
    print("-" * 40)
    try:
        if features is not None:
            # Create more dummy data
            dummy_data = pd.DataFrame(
                np.random.randn(100, len(features)),
                columns=features
            )
            
            # Save predictions to file
            output_path = "example_predictions.csv"
            model.save_predictions(dummy_data, output_path)
            
            # Clean up
            if os.path.exists(output_path):
                os.remove(output_path)
                print(f"   🗑️ Cleaned up example file")
                
    except Exception as e:
        print(f"❌ Error saving predictions: {e}")
    
    print("\n🎉 EXAMPLE COMPLETED!")
    print("=" * 50)
    print("💡 You can now use ModelWrapper in your own scripts!")
    print("📚 Check the documentation for more advanced usage.")

if __name__ == "__main__":
    main() 