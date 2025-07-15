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

def train_model(prepared_data_path="prepared_data/data_splits.pkl", output_dir="trained_model", n_trials=50, cv_folds=5, fast=False):
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

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        prepared_data_path = sys.argv[1]
    else:
        prepared_data_path = "prepared_data/data_splits.pkl"
    train_model(prepared_data_path)
