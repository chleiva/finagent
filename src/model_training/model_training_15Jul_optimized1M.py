# Orchestrator for the full M3-optimized pipeline: prepare, train, evaluate
# Usage: python model_training_15Jul_optimized1M.py [input_csv] [--fast] [--test-csv TEST_CSV] [--description "Model description"]

import os
import sys
import argparse
import time
import datetime
import pandas as pd
import joblib
from pathlib import Path
import numpy as np # Added for np.argmax

# Import the main functions from the modular scripts
from prepare import prepare_data
from train import train_model
from evaluate import evaluate_model

def log_model_run(training_file, test_file, num_features, num_samples, duration_seconds, 
                  evaluation_results, model_description, error_message=None, fast_mode=False):
    """
    Log model training run to CSV file with comprehensive metrics
    """
    log_file = "model_training_log.csv"
    
    # Prepare the log entry
    log_entry = {
        'Date_Start': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Model_Description': model_description,
        'Training_File': training_file,
        'Test_File': test_file if test_file else 'None',
        'Fast_Mode': fast_mode,
        'Num_Features': num_features,
        'Num_Samples': num_samples,
        'Duration_Seconds': duration_seconds,
        'Status': 'Success' if error_message is None else 'Error',
        'Error_Message': error_message if error_message else 'None'
    }
    
    # Add evaluation metrics if available
    if evaluation_results:
        # Best model metrics
        best_model = evaluation_results.get('best_model', 'None')
        log_entry.update({
            'Best_Model': best_model,
            'Test_AUC': evaluation_results.get('test_auc', 0.0),
            'Test_Accuracy': evaluation_results.get('test_accuracy', 0.0),
            'Test_Precision': evaluation_results.get('test_precision', 0.0),
            'Test_Recall': evaluation_results.get('test_recall', 0.0),
            'Test_F1': evaluation_results.get('test_f1', 0.0),
            'Best_Threshold': evaluation_results.get('best_threshold', 0.0),
            'Best_Precision': evaluation_results.get('best_precision', 0.0),
            'Best_Signals': evaluation_results.get('best_signals', 0),
            'Best_Expected_Value': evaluation_results.get('best_expected_value', 0.0),
            'Top_Feature_1': evaluation_results.get('top_feature_1', 'None'),
            'Top_Feature_2': evaluation_results.get('top_feature_2', 'None'),
            'Top_Feature_3': evaluation_results.get('top_feature_3', 'None'),
            'Low_Impact_Features': evaluation_results.get('low_impact_features', 'None'),
            'Model_Performance_Notes': evaluation_results.get('performance_notes', 'None')
        })
    else:
        # Fill with defaults if no evaluation results
        log_entry.update({
            'Best_Model': 'None',
            'Test_AUC': 0.0,
            'Test_Accuracy': 0.0,
            'Test_Precision': 0.0,
            'Test_Recall': 0.0,
            'Test_F1': 0.0,
            'Best_Threshold': 0.0,
            'Best_Precision': 0.0,
            'Best_Signals': 0,
            'Best_Expected_Value': 0.0,
            'Top_Feature_1': 'None',
            'Top_Feature_2': 'None',
            'Top_Feature_3': 'None',
            'Low_Impact_Features': 'None',
            'Model_Performance_Notes': 'None'
        })
    
    # Load existing log or create new one
    if os.path.exists(log_file):
        log_df = pd.read_csv(log_file)
    else:
        log_df = pd.DataFrame()
    
    # Add new entry
    log_df = pd.concat([log_df, pd.DataFrame([log_entry])], ignore_index=True)
    
    # Save updated log
    log_df.to_csv(log_file, index=False)
    print(f"\n📊 Model run logged to {log_file}")
    
    return log_entry

def extract_evaluation_results(output_dir="evaluation_results"):
    """
    Extract key metrics from evaluation results
    """
    results = {}
    
    try:
        # Load test AUC comparison
        auc_file = os.path.join(output_dir, 'test_auc_comparison.csv')
        if os.path.exists(auc_file):
            auc_df = pd.read_csv(auc_file)
            best_model_row = auc_df.loc[auc_df['Test_AUC'].idxmax()]
            results['best_model'] = best_model_row['Model']
            results['test_auc'] = best_model_row['Test_AUC']
        
        # Load trading strategies
        strategy_file = os.path.join(output_dir, 'trading_strategies.csv')
        if os.path.exists(strategy_file):
            strategy_df = pd.read_csv(strategy_file)
            if not strategy_df.empty:
                # Best F1 score
                best_f1_row = strategy_df.loc[strategy_df['F1'].idxmax()]
                results['best_threshold'] = best_f1_row['Threshold']
                results['test_f1'] = best_f1_row['F1']
                results['best_signals'] = best_f1_row['Signals']
                
                # Best expected value
                best_ev_row = strategy_df.loc[strategy_df['Expected_Value'].idxmax()]
                results['best_expected_value'] = best_ev_row['Expected_Value']
                
                # Best precision with >20 signals
                high_signal_df = strategy_df[strategy_df['Signals'] >= 20]
                if not high_signal_df.empty:
                    best_precision_row = high_signal_df.loc[high_signal_df['Precision'].idxmax()]
                    results['best_precision'] = best_precision_row['Precision']
        
        # Load feature importance analysis
        feature_file = os.path.join(output_dir, 'feature_importance_analysis.csv')
        if os.path.exists(feature_file):
            feature_df = pd.read_csv(feature_file)
            if not feature_df.empty:
                # Top 3 features
                top_features = feature_df.sort_values('Mean_Importance', ascending=False).head(3)
                results['top_feature_1'] = top_features.iloc[0]['Feature'] if len(top_features) > 0 else 'None'
                results['top_feature_2'] = top_features.iloc[1]['Feature'] if len(top_features) > 1 else 'None'
                results['top_feature_3'] = top_features.iloc[2]['Feature'] if len(top_features) > 2 else 'None'
                
                # Low impact features
                mean_importance = feature_df['Mean_Importance'].mean()
                std_importance = feature_df['Mean_Importance'].std()
                low_impact = feature_df[feature_df['Mean_Importance'] < (mean_importance - std_importance)]
                results['low_impact_features'] = ', '.join(low_impact['Feature'].tolist()) if not low_impact.empty else 'None'
        
        # Performance notes
        if results.get('test_auc', 0) > 0.7:
            results['performance_notes'] = 'Excellent performance'
        elif results.get('test_auc', 0) > 0.6:
            results['performance_notes'] = 'Good performance'
        elif results.get('test_auc', 0) > 0.55:
            results['performance_notes'] = 'Moderate performance'
        else:
            results['performance_notes'] = 'Poor performance - needs improvement'
            
    except Exception as e:
        print(f"⚠️ Warning: Could not extract all evaluation results: {e}")
    
    return results

def get_model_description():
    """
    Prompt user for model description if not provided
    """
    print("\n📝 MODEL DESCRIPTION")
    print("-" * 40)
    print("Please provide a description for this model run to help identify it later.")
    print("Examples:")
    print("  - 'AAPL 2024-12 with 22 features'")
    print("  - 'High-impact features only'")
    print("  - 'Experiment with reduced feature set'")
    print("  - 'Testing new preprocessing pipeline'")
    print()
    
    while True:
        description = input("Model description: ").strip()
        if description:
            return description
        else:
            print("❌ Description cannot be empty. Please try again.")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='M3-optimized ML pipeline for trading data')
    parser.add_argument('input_csv', nargs='?', default="monthly_AAPL_2024-12.csv", 
                       help='Input CSV file path (default: monthly_AAPL_2024-12.csv)')
    parser.add_argument('--fast', action='store_true', 
                       help='Fast mode: skip expensive optimizations for debugging')
    parser.add_argument('--test-csv', type=str, 
                       help='Test CSV file path for evaluation (should have "buy" or "buy_signal" column for full metrics)')
    parser.add_argument('--description', type=str,
                       help='Model description/name to identify this training run')
    
    args = parser.parse_args()
    
    # Get model description
    model_description = args.description
    if not model_description:
        model_description = get_model_description()
    
    # Start timing
    start_time = time.time()
    error_message = None
    evaluation_results = None
    
    print(f"🚀 Running pipeline with {'FAST' if args.fast else 'FULL'} mode")
    print(f"📊 Input file: {args.input_csv}")
    print(f"📝 Model description: {model_description}")
    if args.test_csv:
        print(f"🧪 Test file: {args.test_csv} (will look for 'buy' or 'buy_signal' column)")
    print("=" * 60)

    try:
        print("\n=== [1/3] PREPARING DATA ===")
        prepare_data(args.input_csv, fast=args.fast)
        
        # Get data statistics
        data_splits = joblib.load("prepared_data/data_splits.pkl")
        num_features = len(data_splits['available_features'])
        num_samples = len(data_splits['X_train']) + len(data_splits['X_val']) + len(data_splits['X_test'])

        print("\n=== [2/3] TRAINING MODEL ===")
        train_model("prepared_data/data_splits.pkl", fast=args.fast)

        print("\n=== [3/3] EVALUATING MODEL ===")
        evaluate_model(
            model_package_path="trained_model/model_package.pkl", 
            prepared_data_path="prepared_data/data_splits.pkl", 
            test_csv=args.test_csv, 
            fast=args.fast
        )
        
        # Extract evaluation results
        evaluation_results = extract_evaluation_results()
        
    except Exception as e:
        error_message = str(e)
        print(f"\n❌ Pipeline failed with error: {error_message}")
    
    # Calculate duration
    duration_seconds = time.time() - start_time
    
    # Log the model run
    log_entry = log_model_run(
        training_file=args.input_csv,
        test_file=args.test_csv,
        num_features=num_features if 'num_features' in locals() else 0,
        num_samples=num_samples if 'num_samples' in locals() else 0,
        duration_seconds=duration_seconds,
        evaluation_results=evaluation_results,
        model_description=model_description,
        error_message=error_message,
        fast_mode=args.fast
    )
    
    if error_message is None:
        print("\n✅ Pipeline completed successfully!")
        print(f"⏱️ Total duration: {duration_seconds:.2f} seconds")
    else:
        print(f"\n❌ Pipeline failed after {duration_seconds:.2f} seconds")
