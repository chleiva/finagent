import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
import os

def evaluate_model(model_package_path="trained_model/model_package.pkl", prepared_data_path="prepared_data/data_splits.pkl", output_dir="evaluation_results", test_csv=None, fast=False):
    if fast:
        print("   ⚡ FAST MODE: Skipping SHAP analysis and reducing plots for debugging")
    
    # Load models and data
    model_package = joblib.load(model_package_path)
    features = model_package['features']
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle custom test CSV if provided
    if test_csv:
        print(f"\n📄 Loading custom test data from {test_csv}")
        try:
            # Load and prepare the custom test data
            test_data = pd.read_csv(test_csv)
            
            # Check if target column exists (either 'buy_signal' or 'buy')
            target_column = None
            if 'buy_signal' in test_data.columns:
                target_column = 'buy_signal'
            elif 'buy' in test_data.columns:
                target_column = 'buy'
                
            if target_column:
                X_test = test_data.drop(target_column, axis=1)
                y_test = test_data[target_column]
                print(f"   ✅ Loaded test data with {len(X_test)} rows and target column '{target_column}'")
            else:
                print(f"   ⚠️ Target column not found in test CSV. Using all columns as features.")
                X_test = test_data
                y_test = None
                print(f"   ⚠️ Without target column, only predictions will be generated (no evaluation metrics).")
        except Exception as e:
            print(f"   ❌ Error loading test CSV: {e}")
            print(f"   ⚠️ Falling back to prepared data splits.")
            data = joblib.load(prepared_data_path)
            X_test = data['X_test']
            y_test = data['y_test']
    else:
        # Load the prepared data splits
        data = joblib.load(prepared_data_path)
        X_test = data['X_test']
        y_test = data['y_test']

    # Debug: Check feature consistency
    print(f"\n🔍 DEBUG: Feature consistency check")
    print(f"   Model features ({len(features)}): {features}")
    print(f"   X_test columns ({len(X_test.columns)}): {list(X_test.columns)}")
    
    # Ensure test set uses only the training features, in the same order
    missing_features = [f for f in features if f not in X_test.columns]
    extra_features = [f for f in X_test.columns if f not in features]
    
    if missing_features:
        print(f"   ⚠️ Missing features in X_test: {missing_features}")
    if extra_features:
        print(f"   ⚠️ Extra features in X_test: {extra_features}")
    
    # Select only the features used in training, in the correct order
    # Handle missing features by adding them with zeros
    for feat in missing_features:
        X_test[feat] = 0
    
    X_test_eval = X_test[features]
    print(f"   ✅ X_test_eval shape: {X_test_eval.shape}")
    print(f"   ✅ X_test_eval columns: {list(X_test_eval.columns)}")

    models = {k: v for k, v in model_package.items() if k in ['XGBoost_Optimized', 'LightGBM', 'GradientBoosting']}
    results = {}
    
    # If no target column is available in the test data, just generate predictions
    if y_test is None:
        print("\n📊 MODEL PREDICTIONS ON TEST SET (NO EVALUATION POSSIBLE)")
        print("=" * 60)
        predictions_df = pd.DataFrame()
        predictions_df['timestamp'] = X_test.index if isinstance(X_test.index, pd.DatetimeIndex) else range(len(X_test))
        
        for name, model in models.items():
            print(f"\n   🤖 {name}:")
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test_eval)[:, 1]
            else:
                y_pred_proba = model.predict(X_test_eval, num_iteration=getattr(model, 'best_iteration', None))
            
            predictions_df[f'{name}_probability'] = y_pred_proba
            predictions_df[f'{name}_prediction'] = (y_pred_proba >= 0.5).astype(int)
            
            print(f"      Generated predictions for {len(X_test_eval)} samples")
            if hasattr(model, 'feature_importances_'):
                fi = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
                fi = fi.sort_values('Importance', ascending=False)
                print("      Top 5 Feature Importances:")
                for _, row in fi.head(5).iterrows():
                    print(f"         {row['Feature']}: {row['Importance']:.4f}")
        
        # Save predictions to CSV
        predictions_path = os.path.join(output_dir, 'test_predictions.csv')
        predictions_df.to_csv(predictions_path, index=False)
        print(f"\n✅ Predictions saved to {predictions_path}")
        return
    
    # Continue with regular evaluation when target column is available
    print("\n📊 MODEL EVALUATION ON TEST SET")
    print("=" * 60)
    for name, model in models.items():
        print(f"\n   🤖 {name}:")
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test_eval)[:, 1]
        else:
            y_pred_proba = model.predict(X_test_eval, num_iteration=getattr(model, 'best_iteration', None))
        y_pred = (y_pred_proba >= 0.5).astype(int)
        auc = roc_auc_score(y_test, y_pred_proba)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        results[name] = {
            'auc': auc,
            'report': report,
            'cm': cm,
            'fpr': fpr,
            'tpr': tpr,
            'y_pred_proba': y_pred_proba,
            'y_pred': y_pred
        }
        print(f"      Test AUC: {auc:.4f}")
        print("      Classification Report:")
        print(classification_report(y_test, y_pred))
        print("      Confusion Matrix:")
        print(cm)
        # Print top 5 feature importances if available
        if hasattr(model, 'feature_importances_'):
            fi = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
            fi = fi.sort_values('Importance', ascending=False)
            print("      Top 5 Feature Importances:")
            for _, row in fi.head(5).iterrows():
                print(f"         {row['Feature']}: {row['Importance']:.4f}")

    # After model evaluation, add comprehensive feature importance analysis
    print("\n📊 COMPREHENSIVE FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)
    
    # Collect feature importances from all models
    feature_importance_dict = {}
    
    for name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            # Normalize importances to percentages
            importances = (importances / importances.sum()) * 100
            feature_importance_dict[name] = dict(zip(features, importances))
    
    # Create a DataFrame with all models' feature importances
    importance_df = pd.DataFrame(feature_importance_dict)
    
    # Calculate aggregate metrics
    importance_df['Mean_Importance'] = importance_df.mean(axis=1)
    importance_df['Std_Importance'] = importance_df.std(axis=1)
    importance_df['Min_Importance'] = importance_df.min(axis=1)
    importance_df['Max_Importance'] = importance_df.max(axis=1)
    importance_df['CV_Importance'] = importance_df['Std_Importance'] / importance_df['Mean_Importance']
    
    # Sort by mean importance
    importance_df = importance_df.sort_values('Mean_Importance', ascending=False)
    
    # Save detailed feature importance analysis
    importance_df.to_csv(os.path.join(output_dir, 'feature_importance_analysis.csv'))
    
    # Print feature importance analysis
    print("\n📈 FEATURE IMPORTANCE RANKING")
    print("-" * 40)
    print(f"{'Feature':<30} {'Mean %':>8} {'Min %':>8} {'Max %':>8} {'CV':>8}")
    print("-" * 70)
    for idx, (feature, row) in enumerate(importance_df.iterrows(), 1):
        print(f"{idx:2d}. {feature:<27} {row['Mean_Importance']:8.2f} {row['Min_Importance']:8.2f} {row['Max_Importance']:8.2f} {row['CV_Importance']:8.2f}")
    
    # Group features by importance
    mean_importance = importance_df['Mean_Importance'].mean()
    std_importance = importance_df['Mean_Importance'].std()
    
    high_impact = importance_df[importance_df['Mean_Importance'] > (mean_importance + std_importance)]
    medium_impact = importance_df[(importance_df['Mean_Importance'] >= (mean_importance - std_importance)) & 
                                (importance_df['Mean_Importance'] <= (mean_importance + std_importance))]
    low_impact = importance_df[importance_df['Mean_Importance'] < (mean_importance - std_importance)]
    
    print("\n🎯 FEATURE IMPACT GROUPS")
    print("-" * 40)
    print("\n🔴 High Impact Features (Consider keeping):")
    for feature in high_impact.index:
        print(f"   • {feature:<30} {high_impact.loc[feature, 'Mean_Importance']:6.2f}%")
    
    print("\n🟡 Medium Impact Features (Review case by case):")
    for feature in medium_impact.index:
        print(f"   • {feature:<30} {medium_impact.loc[feature, 'Mean_Importance']:6.2f}%")
    
    print("\n🔵 Low Impact Features (Consider removing):")
    for feature in low_impact.index:
        print(f"   • {feature:<30} {low_impact.loc[feature, 'Mean_Importance']:6.2f}%")
    
    # Save results as CSV
    pd.DataFrame([
        {'Model': k, 'Test_AUC': v['auc']} for k, v in results.items()
    ]).to_csv(os.path.join(output_dir, 'test_auc_comparison.csv'), index=False)

    # Plot ROC curves
    plt.figure(figsize=(8, 6))
    for name, res in results.items():
        plt.plot(res['fpr'], res['tpr'], label=f"{name} (AUC={res['auc']:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'))
    plt.close()

    # Find best model by AUC
    best_model_name = max(results.keys(), key=lambda x: results[x]['auc'])
    best_model = models[best_model_name]
    best_prob = results[best_model_name]['y_pred_proba']
    best_pred = results[best_model_name]['y_pred']

    print("\n🏆 BEST MODEL ANALYSIS")
    print("-" * 40)
    print(f"   🥇 Best Model: {best_model_name}")
    print(f"      Test AUC: {results[best_model_name]['auc']:.4f}")
    # Print top 10 feature importances if available
    if hasattr(best_model, 'feature_importances_'):
        fi = pd.DataFrame({'Feature': features, 'Importance': best_model.feature_importances_})
        fi = fi.sort_values('Importance', ascending=False)
        print("      Top 10 Feature Importances:")
        for _, row in fi.head(10).iterrows():
            print(f"         {row['Feature']}: {row['Importance']:.4f}")

    # Trading strategy threshold analysis
    print("\n💰 TRADING STRATEGY THRESHOLD ANALYSIS")
    print("-" * 40)
    thresholds = np.arange(0.1, 0.95, 0.05)
    strategy_results = []
    for threshold in thresholds:
        predictions = (best_prob >= threshold).astype(int)
        if predictions.sum() > 0:
            tp = ((predictions == 1) & (y_test == 1)).sum()
            fp = ((predictions == 1) & (y_test == 0)).sum()
            fn = ((predictions == 0) & (y_test == 1)).sum()
            tn = ((predictions == 0) & (y_test == 0)).sum()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            hit_rate = precision
            expected_value = hit_rate * 1.0 - (1 - hit_rate) * 1.0
            strategy_results.append({
                'Threshold': threshold,
                'Signals': predictions.sum(),
                'Precision': precision,
                'Recall': recall,
                'Accuracy': accuracy,
                'F1': f1,
                'Hit_Rate': hit_rate,
                'Expected_Value': expected_value
            })
    strategy_df = pd.DataFrame(strategy_results)
    strategy_df.to_csv(os.path.join(output_dir, 'trading_strategies.csv'), index=False)
    # Print best strategies
    if not strategy_df.empty:
        best_f1_idx = strategy_df['F1'].idxmax()
        best_ev_idx = strategy_df['Expected_Value'].idxmax()
        best_precision_idx = strategy_df[strategy_df['Signals'] >= 20]['Precision'].idxmax() if (strategy_df['Signals'] >= 20).any() else None
        print(f"   Best F1-Score (threshold {strategy_df.loc[best_f1_idx, 'Threshold']:.2f}):")
        print(f"      Signals: {strategy_df.loc[best_f1_idx, 'Signals']}")
        print(f"      Precision: {strategy_df.loc[best_f1_idx, 'Precision']:.1%}")
        print(f"      F1: {strategy_df.loc[best_f1_idx, 'F1']:.3f}")
        print(f"   Best Expected Value (threshold {strategy_df.loc[best_ev_idx, 'Threshold']:.2f}):")
        print(f"      Signals: {strategy_df.loc[best_ev_idx, 'Signals']}")
        print(f"      Expected Value: {strategy_df.loc[best_ev_idx, 'Expected_Value']:.3f}")
        if best_precision_idx is not None:
            print(f"   Best Precision >20 signals (threshold {strategy_df.loc[best_precision_idx, 'Threshold']:.2f}):")
            print(f"      Signals: {strategy_df.loc[best_precision_idx, 'Signals']}")
            print(f"      Precision: {strategy_df.loc[best_precision_idx, 'Precision']:.1%}")

    # SHAP analysis (if available and not in fast mode)
    if not fast:
        print("\n🧠 SHAP VALUE ANALYSIS")
        print("-" * 40)
        try:
            import shap
            if best_model_name in ['XGBoost_Optimized', 'LightGBM']:
                explainer = shap.TreeExplainer(best_model)
                shap_sample_size = min(100, len(X_test_eval))
                X_test_sample = X_test_eval.iloc[:shap_sample_size]
                shap_values = explainer.shap_values(X_test_sample)
                shap_importance = pd.DataFrame({
                    'Feature': features,
                    'SHAP_Importance': np.abs(shap_values).mean(0)
                }).sort_values('SHAP_Importance', ascending=False)
                print(f"   Top 5 Features by SHAP Importance:")
                for _, row in shap_importance.head(5).iterrows():
                    print(f"      {row['Feature']}: {row['SHAP_Importance']:.4f}")
                print("   SHAP summary plot and CSV saved.")
        except ImportError:
            print("   ⚠️ SHAP not installed. Skipping SHAP analysis.")
        except Exception as e:
            print(f"   ⚠️ SHAP analysis failed: {e}")
    else:
        print("\n🧠 SHAP VALUE ANALYSIS")
        print("-" * 40)
        print("   ⚡ FAST MODE: Skipping SHAP analysis")

    print(f"\n✅ Evaluation complete. Results saved to {output_dir}/")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        model_package_path = sys.argv[1]
        prepared_data_path = sys.argv[2]
        test_csv = sys.argv[3] if len(sys.argv) > 3 else None
    else:
        model_package_path = "trained_model/model_package.pkl"
        prepared_data_path = "prepared_data/data_splits.pkl"
        test_csv = None
    evaluate_model(model_package_path, prepared_data_path, test_csv=test_csv) 