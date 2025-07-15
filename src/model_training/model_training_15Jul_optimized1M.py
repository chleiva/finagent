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

# Add src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Import the main functions from the modular scripts
from data_processing.prepare import prepare_data
from evaluation.evaluate import evaluate_model

# Import train_model function from the archived file (we'll copy it here)
def train_model(prepared_data_path="prepared_data/data_splits.pkl", output_dir="trained_model", n_trials=50, cv_folds=5, fast=False):
    """
    Train model function - copied from archived train.py
    """
    import os
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.ensemble import GradientBoostingClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    import optuna

    def check_for_leakage(X, y):
        corr = X.corrwith(y).abs()
        leaks = corr[corr >= 0.999]
        if not leaks.empty:
            print("\n🚨 Potential leakage detected in features:")
            print(leaks.sort_values(ascending=False))
        else:
            print("✅ No perfect correlation with target detected.")

    def remove_leakage_features(X):
        leakage_cols = ['R', 'time_minutes', 'future_return', 'label_debug']
        return X.drop(columns=[col for col in leakage_cols if col in X.columns], errors='ignore')

    if fast:
        print("   ⚡ FAST MODE: Reduced optimization trials and CV folds for debugging")
        n_trials = 5
        cv_folds = 3
    
    # Load prepared data
    data = joblib.load(prepared_data_path)
    X_train = data['X_train']
    X_val = data['X_val']
    y_train = data['y_train']
    y_val = data['y_val']
    available_features = data['available_features']

    # Combine train and val for cross-validation
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])

    # Remove leakage features
    X_trainval = remove_leakage_features(X_trainval)

    # Sanity check
    check_for_leakage(X_trainval, y_trainval)

    print("\n🚀 [1/3] OPTUNA HYPERPARAMETER OPTIMIZATION (XGBoost)")
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 1, 2),
            'random_state': 42,
            'verbosity': 0,
            'n_jobs': -1
        }
        model = XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_trainval, y_trainval, cv=cv, scoring='roc_auc', n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best_params = study.best_params
    print(f"   ✅ Best XGBoost params: {best_params}")
    print(f"   ✅ Best CV AUC: {study.best_value:.4f}")

    # Save Optuna study
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(study, os.path.join(output_dir, 'optuna_study.pkl'))

    print("\n🚀 [2/3] TRAINING FINAL MODELS")
    # Train XGBoost with best params
    xgb_model = XGBClassifier(**best_params)
    xgb_model.fit(X_trainval, y_trainval)
    # Get actual features used by XGBoost
    trained_features = xgb_model.get_booster().feature_names

    # Train LightGBM
    lgbm_params = {
        'n_estimators': 500,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'verbosity': -1,
        'n_jobs': -1
    }
    lgbm_model = LGBMClassifier(**lgbm_params)
    lgbm_model.fit(X_trainval, y_trainval)

    # Train GradientBoosting
    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    )
    gb_model.fit(X_trainval, y_trainval)

    print("\n🚀 [3/3] CROSS-VALIDATION MODEL COMPARISON")
    models = {
        'XGBoost_Optimized': xgb_model,
        'LightGBM': lgbm_model,
        'GradientBoosting': gb_model
    }
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    results = {}
    for name, model in models.items():
        print(f"   🤖 {name}:")
        scores = cross_val_score(model, X_trainval, y_trainval, cv=cv, scoring='roc_auc', n_jobs=-1)
        results[name] = {
            'cv_scores': scores,
            'cv_mean': scores.mean(),
            'cv_std': scores.std()
        }
        print(f"      CV AUC: {scores.mean():.4f} (±{scores.std():.4f})")

    # Save all models and results
    joblib.dump({
        'XGBoost_Optimized': xgb_model,
        'LightGBM': lgbm_model,
        'GradientBoosting': gb_model,
        'features': trained_features,
        'best_params': best_params,
        'cv_results': results
    }, os.path.join(output_dir, 'model_package.pkl'))
    pd.DataFrame([
        {'Model': k, 'CV_AUC_Mean': v['cv_mean'], 'CV_AUC_Std': v['cv_std']} for k, v in results.items()
    ]).to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
    print(f"\n✅ All models and results saved to {output_dir}/")

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
